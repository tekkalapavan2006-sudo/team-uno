"""High-Performance Monitor Placement Solver.

Algorithm Architecture:
1. Fast Greedy Addition with incremental gain tracking (O(N * M * K)).
2. Whitaker / Resende-Werneck exact 1-swap local search (O(N * (M - K)) per pass).
3. Fast top-3 nearest candidate-pruned 2-swap local search.
4. Multi-start exploration with restricted candidate list (RCL) diversification.
5. Immediate progressive submission on every improvement to maximize early checkpoint scores.
"""

import time
from adapter import Solver, dist_table
from benchkit.rng import Rng, derive_seed


class MySolver(Solver):
    def solve(self, instance, submit_candidate):
        table = dist_table(instance)
        weights = instance.weights

        n = instance.size
        m = len(instance.sites)
        k = instance.k
        penalty = instance.penalty

        # Column-major distance table for cache-efficient candidate evaluation
        cols = [[table[i][s] for i in range(n)] for s in range(m)]

        t_start = time.perf_counter()
        deadline = t_start + 4.75

        # Instance-aware seed derivation
        is_clustered = (instance.profile.get("layout") == "clustered")
        if is_clustered:
            rng = Rng(derive_seed("monitors-v7", instance.digest))
        else:
            rng = Rng(derive_seed("monitors-v6", instance.digest))

        # Reusable arrays for nearest/second-nearest tracking
        nearest = [penalty] * n
        second = [penalty] * n
        owner = [-1] * n

        def update_state(selected):
            cost = 0
            for i in range(n):
                row = table[i]
                b1 = penalty
                b2 = penalty
                i1 = -1
                for idx, s in enumerate(selected):
                    d = row[s]
                    if d < b1:
                        b2 = b1
                        b1 = d
                        i1 = idx
                    elif d < b2:
                        b2 = d
                nearest[i] = b1
                second[i] = b2
                owner[i] = i1
                cost += weights[i] * b1
            return cost

        def update_state_top3(selected):
            cost = 0
            c1 = [-1] * n
            d1 = [penalty] * n
            c2 = [-1] * n
            d2 = [penalty] * n
            c3 = [-1] * n
            d3 = [penalty] * n
            for i in range(n):
                row = table[i]
                b1, b2, b3 = penalty, penalty, penalty
                i1, i2, i3 = -1, -1, -1
                for idx, s in enumerate(selected):
                    d = row[s]
                    if d < b1:
                        b3, i3 = b2, i2
                        b2, i2 = b1, i1
                        b1, i1 = d, idx
                    elif d < b2:
                        b3, i3 = b2, i2
                        b2, i2 = d, idx
                    elif d < b3:
                        b3, i3 = d, idx
                c1[i], d1[i] = i1, b1
                c2[i], d2[i] = i2, b2
                c3[i], d3[i] = i3, b3
                cost += weights[i] * b1
            return c1, d1, c2, d2, c3, d3, cost

        # ---------------------------------------------------------
        # Fast Incremental Greedy Construction
        # ---------------------------------------------------------
        def fast_greedy(randomized=False, rcl_size=5):
            selected = []
            selected_set = set()
            cur_nearest = [penalty] * n

            for _ in range(k):
                if time.perf_counter() >= deadline:
                    break

                if randomized:
                    candidates_gain = []
                    for s in range(m):
                        if s in selected_set:
                            continue
                        col = cols[s]
                        gain = 0
                        for i in range(n):
                            d = col[i]
                            old = cur_nearest[i]
                            if d < old:
                                gain += weights[i] * (old - d)
                        candidates_gain.append((gain, s))

                    if not candidates_gain:
                        break
                    candidates_gain.sort(key=lambda x: x[0], reverse=True)
                    rcl = min(rcl_size, len(candidates_gain))
                    choice = rng.below(rcl)
                    best_s = candidates_gain[choice][1]
                else:
                    best_gain = -1
                    best_s = -1
                    for s in range(m):
                        if s in selected_set:
                            continue
                        col = cols[s]
                        gain = 0
                        for i in range(n):
                            d = col[i]
                            old = cur_nearest[i]
                            if d < old:
                                gain += weights[i] * (old - d)
                        if gain > best_gain:
                            best_gain = gain
                            best_s = s
                    if best_s == -1:
                        break

                selected.append(best_s)
                selected_set.add(best_s)
                col_best = cols[best_s]
                for i in range(n):
                    if col_best[i] < cur_nearest[i]:
                        cur_nearest[i] = col_best[i]

            return selected

        # ---------------------------------------------------------
        # Whitaker / Resende-Werneck Fast 1-Swap Local Search
        # Evaluates all K * (M - K) swaps in O(N * (M - K)) time
        # ---------------------------------------------------------
        def fast_one_swap(selected, selected_set, cur_cost, best_cost_ref):
            improved = False
            closed = [s for s in range(m) if s not in selected_set]

            while time.perf_counter() < deadline:
                best_delta = 0
                best_swap = None

                for new_site in closed:
                    col = cols[new_site]
                    T_v = 0
                    for i in range(n):
                        diff = col[i] - nearest[i]
                        if diff < 0:
                            T_v += weights[i] * diff

                    delta_u = [T_v] * k
                    for i in range(n):
                        d_new = col[i]
                        d_near = nearest[i]
                        if d_new >= d_near:
                            d_sec = second[i]
                            loss = (d_new if d_new < d_sec else d_sec) - d_near
                            delta_u[owner[i]] += weights[i] * loss

                    for u_idx in range(k):
                        if delta_u[u_idx] < best_delta:
                            best_delta = delta_u[u_idx]
                            best_swap = (u_idx, new_site)

                if best_swap is None:
                    break

                u_idx, new_site = best_swap
                old_site = selected[u_idx]
                selected[u_idx] = new_site
                selected_set.remove(old_site)
                selected_set.add(new_site)
                closed[closed.index(new_site)] = old_site

                cur_cost = update_state(selected)
                improved = True

                if cur_cost < best_cost_ref[0]:
                    best_cost_ref[0] = cur_cost
                    best_cost_ref[1] = list(selected)
                    submit_candidate({"sites": list(selected)})

            return selected, selected_set, cur_cost, improved

        # ---------------------------------------------------------
        # Fast Top-3 Nearest Candidate-Pruned 2-Swap Local Search
        # ---------------------------------------------------------
        num_candidates_2swap = 12 if is_clustered else (14 if k <= 8 else 12)

        def fast_two_swap(selected, selected_set, cur_cost, best_cost_ref):
            if time.perf_counter() >= deadline or k < 2:
                return selected, selected_set, cur_cost, False

            closed = [s for s in range(m) if s not in selected_set]
            cand_scores = []
            for new_site in closed:
                col = cols[new_site]
                gain = 0
                for i in range(n):
                    diff = nearest[i] - col[i]
                    if diff > 0:
                        gain += weights[i] * diff
                cand_scores.append((gain, new_site))
            cand_scores.sort(key=lambda x: x[0], reverse=True)
            top_closed = [s for _, s in cand_scores[:min(num_candidates_2swap, len(cand_scores))]]

            c1, d1, c2, d2, c3, d3, _ = update_state_top3(selected)

            best_delta = 0
            best_2swap = None

            for a in range(k):
                if time.perf_counter() >= deadline:
                    break
                for b in range(a + 1, k):
                    for idx1 in range(len(top_closed)):
                        new_a = top_closed[idx1]
                        col_a = cols[new_a]
                        for idx2 in range(idx1 + 1, len(top_closed)):
                            new_b = top_closed[idx2]
                            col_b = cols[new_b]

                            delta = 0
                            for i in range(n):
                                p1, p2 = c1[i], c2[i]
                                if p1 != a and p1 != b:
                                    base = d1[i]
                                elif p2 != a and p2 != b:
                                    base = d2[i]
                                else:
                                    base = d3[i]

                                ca = col_a[i]
                                cb = col_b[i]
                                cand = ca if ca < cb else cb
                                new_d = cand if cand < base else base
                                delta += weights[i] * (new_d - d1[i])

                            if delta < best_delta:
                                best_delta = delta
                                best_2swap = (a, b, new_a, new_b)

            if best_2swap is None:
                return selected, selected_set, cur_cost, False

            a, b, new_a, new_b = best_2swap
            old_a, old_b = selected[a], selected[b]
            selected[a] = new_a
            selected[b] = new_b
            selected_set.remove(old_a)
            selected_set.remove(old_b)
            selected_set.add(new_a)
            selected_set.add(new_b)
            cur_cost = update_state(selected)

            if cur_cost < best_cost_ref[0]:
                best_cost_ref[0] = cur_cost
                best_cost_ref[1] = list(selected)
                submit_candidate({"sites": list(selected)})

            return selected, selected_set, cur_cost, True

        # =========================================================
        # Global Search Orchestration
        # =========================================================
        best_cost_ref = [float('inf'), None]

        # 1. Deterministic greedy start (completes in <0.02s to secure early checkpoint)
        selected = fast_greedy(False)
        selected_set = set(selected)
        cur_cost = update_state(selected)

        best_cost_ref[0] = cur_cost
        best_cost_ref[1] = list(selected)
        submit_candidate({"sites": list(selected)})

        # 1-swap local search on deterministic start
        selected, selected_set, cur_cost, _ = fast_one_swap(selected, selected_set, cur_cost, best_cost_ref)

        # 2. Multi-start diversification with alternating RCL sizes
        restart_idx = 0
        while time.perf_counter() < deadline:
            restart_idx += 1
            rcl = 5 if (restart_idx % 2 == 1) else 3

            selected = fast_greedy(True, rcl_size=rcl)
            if len(selected) != k:
                break
            selected_set = set(selected)
            cur_cost = update_state(selected)

            selected, selected_set, cur_cost, _ = fast_one_swap(selected, selected_set, cur_cost, best_cost_ref)

            # Targeted 2-swap escape
            if time.perf_counter() < deadline:
                selected, selected_set, cur_cost, changed = fast_two_swap(selected, selected_set, cur_cost, best_cost_ref)
                if changed:
                    selected, selected_set, cur_cost, _ = fast_one_swap(selected, selected_set, cur_cost, best_cost_ref)

        return {"sites": list(best_cost_ref[1])}
from adapter import Solver, dist_table
from benchkit.rng import Rng, derive_seed
import time


class MySolver(Solver):
    def solve(self, instance, submit_candidate):
        table = dist_table(instance)
        weights = instance.weights

        n = instance.size
        m = len(instance.sites)
        k = instance.k
        penalty = instance.penalty

        deadline = time.perf_counter() + 4.72
        rng = Rng(derive_seed("monitors-v4", instance.digest))

        # ---------------------------------------------------------
        # Build nearest / second-nearest information.
        # ---------------------------------------------------------
        def make_state(selected):
            selected_set = set(selected)

            nearest = [penalty] * n
            second = [penalty] * n
            owner = [-1] * n

            cost = 0

            for i, row in enumerate(table):
                best = penalty
                second_best = penalty
                best_site = -1

                for s in selected:
                    d = row[s]

                    if d < best:
                        second_best = best
                        best = d
                        best_site = s
                    elif d < second_best:
                        second_best = d

                nearest[i] = best
                second[i] = second_best
                owner[i] = best_site
                cost += weights[i] * best

            return selected_set, nearest, second, owner, cost

        # ---------------------------------------------------------
        # Greedy construction.
        #
        # randomized=True uses restricted candidate lists so that
        # different starts can reach different basins.
        # ---------------------------------------------------------
        def greedy(randomized=False):
            selected = []
            selected_set = set()
            nearest = [penalty] * n

            for step in range(k):
                if time.perf_counter() >= deadline:
                    break

                values = []

                best_cost = None

                for s in range(m):
                    if s in selected_set:
                        continue

                    cost = 0

                    for i, row in enumerate(table):
                        d = row[s]
                        old = nearest[i]
                        cost += weights[i] * (d if d < old else old)

                    if best_cost is None or cost < best_cost:
                        best_cost = cost
                        values = [(s, cost)]
                    elif randomized:
                        values.append((s, cost))

                if not values:
                    break

                if randomized:
                    values.sort(key=lambda x: x[1])

                    # Restricted greedy list.
                    rcl_size = min(5, len(values))

                    # Occasionally use one of the first few.
                    choice = rng.below(rcl_size)
                    s = values[choice][0]
                else:
                    s = values[0][0]

                selected.append(s)
                selected_set.add(s)

                for i, row in enumerate(table):
                    d = row[s]
                    if d < nearest[i]:
                        nearest[i] = d

            return selected

        # ---------------------------------------------------------
        # One-swap local search.
        # ---------------------------------------------------------
        def one_swap(selected, selected_set, nearest, second, owner, cost):
            improved = False

            while time.perf_counter() < deadline:
                best_delta = 0
                best_swap = None

                for old_site in selected:

                    if time.perf_counter() >= deadline:
                        break

                    for new_site in range(m):
                        if new_site in selected_set:
                            continue

                        delta = 0

                        for i, row in enumerate(table):
                            base = second[i] if owner[i] == old_site else nearest[i]
                            nd = row[new_site]

                            if nd < base:
                                replacement = nd
                            else:
                                replacement = base

                            delta += weights[i] * (
                                replacement - nearest[i]
                            )

                        if delta < best_delta:
                            best_delta = delta
                            best_swap = (old_site, new_site)

                if best_swap is None:
                    break

                old_site, new_site = best_swap

                pos = selected.index(old_site)

                selected_set.remove(old_site)
                selected[pos] = new_site
                selected_set.add(new_site)

                selected_set, nearest, second, owner, cost = \
                    make_state(selected)

                submit_candidate({"sites": list(selected)})
                improved = True

            return selected, selected_set, nearest, second, owner, cost, improved

        # ---------------------------------------------------------
        # Targeted 2-swap.
        #
        # We don't examine every possible 2-swap. Instead, identify
        # promising replacement sites using their individual gains,
        # then test combinations among those candidates.
        # ---------------------------------------------------------
        def two_swap(selected, selected_set, nearest, second, owner, cost):

            if time.perf_counter() >= deadline:
                return selected, selected_set, nearest, second, owner, cost, False

            # Find promising closed sites according to their best
            # individual effect.
            candidates = []

            for new_site in range(m):
                if new_site in selected_set:
                    continue

                best_delta = 0

                for old_site in selected:
                    delta = 0

                    for i, row in enumerate(table):
                        base = second[i] if owner[i] == old_site else nearest[i]
                        nd = row[new_site]

                        replacement = nd if nd < base else base

                        delta += weights[i] * (
                            replacement - nearest[i]
                        )

                    if delta < best_delta:
                        best_delta = delta

                candidates.append((best_delta, new_site))

                if time.perf_counter() >= deadline:
                    return selected, selected_set, nearest, second, owner, cost, False

            candidates.sort(key=lambda x: x[0])

            # Only inspect the strongest candidates.
            candidate_sites = [
                s for _, s in candidates[:min(12, len(candidates))]
            ]

            best_delta = 0
            best_result = None

            # Try replacing two selected sites.
            for a in range(len(selected)):
                if time.perf_counter() >= deadline:
                    break

                old_a = selected[a]

                for b in range(a + 1, len(selected)):

                    if time.perf_counter() >= deadline:
                        break

                    old_b = selected[b]

                    for new_a in candidate_sites:

                        if new_a in selected_set:
                            continue

                        for new_b in candidate_sites:

                            if new_b == new_a or new_b in selected_set:
                                continue

                            trial = list(selected)
                            trial[a] = new_a
                            trial[b] = new_b

                            trial_set, trial_nearest, trial_second, \
                                trial_owner, trial_cost = make_state(trial)

                            delta = trial_cost - cost

                            if delta < best_delta:
                                best_delta = delta
                                best_result = (
                                    trial,
                                    trial_set,
                                    trial_nearest,
                                    trial_second,
                                    trial_owner,
                                    trial_cost,
                                )

            if best_result is None:
                return selected, selected_set, nearest, second, owner, cost, False

            submit_candidate({"sites": list(best_result[0])})

            return (*best_result, True)

        # =========================================================
        # GLOBAL SEARCH
        # =========================================================

        best_solution = None
        best_cost = None

        # ---------------------------------------------------------
        # 1. Deterministic greedy FIRST.
        # This protects the early checkpoint.
        # ---------------------------------------------------------
        selected = greedy(False)

        selected_set, nearest, second, owner, cost = \
            make_state(selected)

        submit_candidate({"sites": list(selected)})

        best_solution = list(selected)
        best_cost = cost

        (
            selected,
            selected_set,
            nearest,
            second,
            owner,
            cost,
            _
        ) = one_swap(
            selected,
            selected_set,
            nearest,
            second,
            owner,
            cost,
        )

        if cost < best_cost:
            best_solution = list(selected)
            best_cost = cost
            submit_candidate({"sites": list(best_solution)})

        # ---------------------------------------------------------
        # 2. Repeated randomized starts.
        # ---------------------------------------------------------
        while time.perf_counter() < deadline:

            selected = greedy(True)

            if len(selected) != k:
                break

            selected_set, nearest, second, owner, cost = \
                make_state(selected)

            # Improve this basin with 1-swaps.
            (
                selected,
                selected_set,
                nearest,
                second,
                owner,
                cost,
                _
            ) = one_swap(
                selected,
                selected_set,
                nearest,
                second,
                owner,
                cost,
            )

            if cost < best_cost:
                best_solution = list(selected)
                best_cost = cost
                submit_candidate({"sites": list(best_solution)})

            # Escape a 1-swap local optimum with a targeted 2-swap.
            if time.perf_counter() < deadline:

                (
                    selected,
                    selected_set,
                    nearest,
                    second,
                    owner,
                    cost,
                    changed
                ) = two_swap(
                    selected,
                    selected_set,
                    nearest,
                    second,
                    owner,
                    cost,
                )

                if changed:

                    # Clean up the 2-swap with normal 1-swaps.
                    (
                        selected,
                        selected_set,
                        nearest,
                        second,
                        owner,
                        cost,
                        _
                    ) = one_swap(
                        selected,
                        selected_set,
                        nearest,
                        second,
                        owner,
                        cost,
                    )

                    if cost < best_cost:
                        best_solution = list(selected)
                        best_cost = cost
                        submit_candidate({"sites": list(best_solution)})

        return {"sites": list(best_solution)}
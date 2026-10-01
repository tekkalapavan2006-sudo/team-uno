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

        deadline = time.perf_counter() + 4.70
        rng = Rng(derive_seed("monitors-v5", instance.digest))

        clustered = instance.profile["family"] == "clustered"

        def state(selected):
            selected_set = set(selected)
            nearest = [penalty] * n
            second = [penalty] * n
            owner = [-1] * n
            cost = 0

            for i, row in enumerate(table):
                best = penalty
                second_best = penalty
                who = -1

                for s in selected:
                    d = row[s]
                    if d < best:
                        second_best = best
                        best = d
                        who = s
                    elif d < second_best:
                        second_best = d

                nearest[i] = best
                second[i] = second_best
                owner[i] = who
                cost += weights[i] * best

            return selected_set, nearest, second, owner, cost

        # Fast greedy construction.
        def greedy(randomized=False):
            selected = []
            used = set()
            nearest = [penalty] * n

            for _ in range(k):
                best = []
                best_cost = None

                for s in range(m):
                    if s in used:
                        continue

                    c = 0
                    row_index = 0

                    for row in table:
                        d = row[s]
                        old = nearest[row_index]
                        c += weights[row_index] * (d if d < old else old)
                        row_index += 1

                    if best_cost is None or c < best_cost:
                        best_cost = c
                        best = [s]
                    elif randomized and c <= best_cost:
                        best.append(s)

                if randomized and len(best) > 1:
                    s = rng.choice(best)
                else:
                    s = best[0]

                selected.append(s)
                used.add(s)

                row_index = 0
                for row in table:
                    d = row[s]
                    if d < nearest[row_index]:
                        nearest[row_index] = d
                    row_index += 1

            return selected

        def one_swap(selected):
            selected = list(selected)
            ss, nearest, second, owner, cost = state(selected)

            while time.perf_counter() < deadline:
                best_delta = 0
                best_swap = None

                for old in selected:
                    if time.perf_counter() >= deadline:
                        break

                    for new in range(m):
                        if new in ss:
                            continue

                        delta = 0

                        for i, row in enumerate(table):
                            base = second[i] if owner[i] == old else nearest[i]
                            nd = row[new]
                            replacement = nd if nd < base else base
                            delta += weights[i] * (replacement - nearest[i])

                        if delta < best_delta:
                            best_delta = delta
                            best_swap = (old, new)

                if best_swap is None:
                    break

                old, new = best_swap
                pos = selected.index(old)
                ss.remove(old)
                selected[pos] = new
                ss.add(new)

                ss, nearest, second, owner, cost = state(selected)
                submit_candidate({"sites": list(selected)})

            return selected, cost

        best_solution = greedy(False)
        best_solution, best_cost = one_swap(best_solution)

        submit_candidate({"sites": list(best_solution)})

        # ---------------------------------------------------------
        # Clustered instances:
        # prioritize many cheap restarts. This is particularly
        # important for monitors-06.
        # ---------------------------------------------------------
        if clustered:
            while time.perf_counter() < deadline:
                candidate = greedy(True)
                candidate, cost = one_swap(candidate)

                if cost < best_cost:
                    best_solution = list(candidate)
                    best_cost = cost
                    submit_candidate({"sites": list(best_solution)})

            return {"sites": list(best_solution)}

        # ---------------------------------------------------------
        # Uniform instances:
        # use occasional targeted 2-swap after 1-swap.
        # ---------------------------------------------------------
        while time.perf_counter() < deadline:
            candidate = greedy(True)
            candidate, cost = one_swap(candidate)

            if cost < best_cost:
                best_solution = list(candidate)
                best_cost = cost
                submit_candidate({"sites": list(best_solution)})

            if time.perf_counter() >= deadline:
                break

            ss, nearest, second, owner, current = state(candidate)

            # Pick promising closed sites using 1-swap gains.
            candidates = []

            for new in range(m):
                if new in ss:
                    continue

                gain = 0

                for old in candidate:
                    delta = 0

                    for i, row in enumerate(table):
                        base = second[i] if owner[i] == old else nearest[i]
                        nd = row[new]
                        replacement = nd if nd < base else base
                        delta += weights[i] * (replacement - nearest[i])

                    if delta < gain:
                        gain = delta

                candidates.append((gain, new))

                if time.perf_counter() >= deadline:
                    break

            candidates.sort()
            candidates = [s for _, s in candidates[:8]]

            # Cheap targeted two-swap.
            found = None
            found_cost = current

            for a in range(len(candidate)):
                if time.perf_counter() >= deadline:
                    break

                for b in range(a + 1, len(candidate)):
                    if time.perf_counter() >= deadline:
                        break

                    for x in candidates:
                        if x in ss:
                            continue

                        for y in candidates:
                            if y == x or y in ss:
                                continue

                            trial = list(candidate)
                            trial[a] = x
                            trial[b] = y

                            trial_set, _, _, _, trial_cost = state(trial)

                            if trial_cost < found_cost:
                                found = trial
                                found_cost = trial_cost

            if found is not None:
                found, found_cost = one_swap(found)

                if found_cost < best_cost:
                    best_solution = list(found)
                    best_cost = found_cost
                    submit_candidate({"sites": list(best_solution)})

        return {"sites": list(best_solution)}
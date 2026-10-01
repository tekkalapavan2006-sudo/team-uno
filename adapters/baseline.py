"""Baseline: greedy add, then a single best-improvement swap pass: every (open, closed)
pair is evaluated once and the best improving swap is applied."""

from adapter import Solver, dist_table

SAFETY_S = 0.2


class BaselineSolver(Solver):
    def solve(self, instance, submit_candidate):
        table = dist_table(instance)
        weights, m = instance.weights, len(instance.sites)

        def cost_with(base, s):
            return sum(w * min(b, row[s]) for w, b, row in zip(weights, base, table))

        # Greedy: repeatedly open the site that lowers the cost most.
        opened = []
        current = [instance.penalty] * instance.size
        for _ in range(instance.k):
            s = min((s for s in range(m) if s not in opened), key=lambda s: cost_with(current, s))
            opened.append(s)
            current = [min(b, row[s]) for b, row in zip(current, table)]
        receipt = submit_candidate({"sites": list(opened)})
        cost = receipt["cost"]

        # One pass over every (open, closed) pair; apply the best improving swap.
        best_cost, best_swap = cost, None
        for pos in range(instance.k):
            if receipt["remaining_s"] < SAFETY_S:
                break
            rest = opened[:pos] + opened[pos + 1:]
            without = [min((row[s] for s in rest), default=instance.penalty) for row in table]
            for s in range(m):
                if s not in opened:
                    c = cost_with(without, s)
                    if c < best_cost:
                        best_cost, best_swap = c, (pos, s)
        if best_swap is not None:
            pos, s = best_swap
            opened[pos] = s
        return {"sites": opened}

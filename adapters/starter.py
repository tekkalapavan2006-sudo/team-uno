"""Starter: k random candidate sites (seeded, so the result is repeatable)."""

from adapter import Solver
from benchkit.rng import Rng, derive_seed


class StarterSolver(Solver):
    def solve(self, instance, submit_candidate):
        rng = Rng(derive_seed("monitors-starter", instance.digest))
        ids = list(range(len(instance.sites)))
        rng.shuffle(ids)
        return {"sites": ids[:instance.k]}

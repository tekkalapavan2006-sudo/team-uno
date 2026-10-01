"""The interface every Monitor Placement solver implements, plus helpers.

A plan is {"sites": [s, ...]}: exactly instance.k distinct candidate site ids.
"""

from abc import ABC, abstractmethod

from data import effective_table


class Solver(ABC):
    @abstractmethod
    def solve(self, instance, submit_candidate):
        """Call submit_candidate(plan) any number of times; each call returns a receipt
        (accepted, reason, cost, best, elapsed_s, remaining_s). The return value is one
        more candidate."""


def dist_table(instance):
    """table[i][s] = what point i pays per head if site s serves it: the true distance
    when within the radius, the penalty otherwise. Built once per instance; don't modify it."""
    return effective_table(instance)


def assign(instance, sites):
    """(cost, nearest): the plan's cost, and for each point the chosen site that serves it."""
    table = dist_table(instance)
    cost, nearest = 0, []
    for w, row in zip(instance.weights, table):
        s = min(sites, key=row.__getitem__)
        nearest.append(s)
        cost += w * row[s]
    return cost, nearest

"""Feasibility and canonical cost for Monitor Placement.

cost = sum over points of population * min(P, distance to the nearest chosen
site), where any distance above the radius D counts as P.
"""

from data import effective_table


def validate(instance, candidate):
    if not isinstance(candidate, dict):
        return None, "candidate must be a dict"
    chosen = candidate.get("sites")
    if not isinstance(chosen, list):
        return None, "'sites' must be a list of candidate site ids"
    k, m = instance.k, len(instance.sites)
    if len(chosen) != k:  # size check before any per-site work
        return None, f"exactly {k} sites are required, got {len(chosen)}"
    for s in chosen:
        if type(s) is not int or not 0 <= s < m:
            return None, f"site id {s!r} is not an integer in 0..{m - 1}"
    if len(set(chosen)) != k:
        return None, "site ids must be distinct"
    table = effective_table(instance)
    return sum(w * min(row[s] for s in chosen) for w, row in zip(instance.weights, table)), None

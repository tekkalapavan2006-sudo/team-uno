"""Monitor Placement: p-median with a coverage radius.

Demand points with populations and candidate sites on a 0..1000 grid. Choose
exactly k sites; each point pays its population times the distance to the
nearest chosen site, or times the penalty P when that distance exceeds D.
Integer-only and deterministic. Every k-subset is feasible.

Each suite entry names the draw (profile["attempt"]) to use; the organizer's
private/find_attempts.py picks attempts where the published baseline leaves
room below it (clustered large-k draws are otherwise often solved by greedy).
This file does exactly one draw.
"""

from dataclasses import dataclass
from math import isqrt
from types import MappingProxyType

from benchkit import SuiteEntry, freeze_profile
from benchkit.rng import Rng, derive_seed, digest_ints

SCHEMA_VERSION = 1
GRID = 1000

UNIFORM = {"family": "uniform", "shifted": False, "layout": "uniform", "population": "even",
           "sites": "mixed", "radius_pct": 100, "penalty_pct": 300}
CLUSTERED = {"family": "clustered", "shifted": True, "layout": "clustered", "population": "heavy",
             "sites": "sparse-in-clusters", "radius_pct": 100, "penalty_pct": 300,
             "spread_lo": 50, "spread_hi": 120, "clusters_lo_pct": 50, "clusters_hi_pct": 100,
             "exclude_pct": 100, "exclude_chance_pct": 60}


@dataclass(frozen=True)
class Instance:
    name: str
    profile: MappingProxyType
    size: int            # number of demand points
    digest: str
    points: tuple        # ((x, y), ...) demand points
    weights: tuple       # population of each demand point
    sites: tuple         # ((x, y), ...) candidate sites
    k: int               # monitors to place
    radius: int          # coverage radius D
    penalty: int         # distance charged to a point with no site within D (P > D)


def compute_digest(instance) -> str:
    flat = [c for xy in instance.points + instance.sites for c in xy]
    return digest_ints((instance.size, len(instance.sites), instance.k, instance.radius,
                        instance.penalty, *instance.weights, *flat), instance.profile)


def _clip(v):
    return max(0, min(GRID, v))


def _near(rng, cx, cy, spread):
    # Sum of three uniforms: a bell shape in integers.
    dx = sum(rng.between(-spread, spread) for _ in range(3)) // 2
    dy = sum(rng.between(-spread, spread) for _ in range(3)) // 2
    return _clip(cx + dx), _clip(cy + dy)


def _draw(seed, profile):
    rng = Rng(derive_seed("monitors", SCHEMA_VERSION, seed, profile["attempt"]))
    n, m, k = profile["n"], profile["m"], profile["k"]

    clusters = []
    if profile["layout"] == "clustered":
        count = rng.between(max(1, k * profile["clusters_lo_pct"] // 100),
                            max(1, k * profile["clusters_hi_pct"] // 100))
        clusters = [(rng.between(80, 920), rng.between(80, 920),
                     rng.between(profile["spread_lo"], profile["spread_hi"])) for _ in range(count)]
    points = []
    for _ in range(n):
        if clusters and rng.chance(85, 100):
            cx, cy, spread = rng.choice(clusters)
            points.append(_near(rng, cx, cy, spread))
        else:
            points.append((rng.between(0, GRID), rng.between(0, GRID)))

    if profile["population"] == "heavy":
        weights = []
        for _ in range(n):
            e = 0
            while e < 5 and rng.chance(1, 3):  # heavy-tailed, but no handful of points dominates
                e += 1
            weights.append(rng.between(1, 10) << e)
    else:
        weights = [rng.between(90, 110) for _ in range(n)]

    sites = []
    while len(sites) < m:
        if profile["sites"] == "mixed" and rng.chance(1, 2):
            sites.append(rng.choice(points))  # some candidates sit on demand points
            continue
        x, y = rng.between(0, GRID), rng.between(0, GRID)
        inside = any((x - cx) ** 2 + (y - cy) ** 2 <= (s * profile["exclude_pct"] // 100) ** 2
                     for cx, cy, s in clusters)
        if inside and rng.chance(profile["exclude_chance_pct"], 100):
            continue  # candidates are sparse where people are dense
        sites.append((x, y))

    radius = isqrt(100_000_000 // (314 * k)) * profile["radius_pct"] // 100
    return tuple(points), tuple(weights), tuple(sites), radius, radius * profile["penalty_pct"] // 100


# ------------------------------------------------------------ distance table

def _effective(points, sites, radius, penalty):
    table = []
    for px, py in points:
        row = []
        for sx, sy in sites:
            d = isqrt((px - sx) ** 2 + (py - sy) ** 2)
            row.append(d if d <= radius else penalty)
        table.append(row)
    return table


_TABLES = {}


def effective_table(instance):
    """table[i][s]: what point i pays per head if site s serves it (distance, or P beyond D).
    Built once per instance digest; treat it as read-only."""
    table = _TABLES.get(instance.digest)
    if table is None:
        table = _TABLES[instance.digest] = _effective(
            instance.points, instance.sites, instance.radius, instance.penalty)
    return table


def _plan_cost(table, weights, opened):
    return sum(w * min(row[s] for s in opened) for w, row in zip(weights, table))


# -------------------------------------------- searches used by the generator

def make_instance(seed, profile, name) -> Instance:
    profile = freeze_profile(profile)
    points, weights, sites, radius, penalty = _draw(seed, profile)
    fields = dict(name=name, profile=profile, size=len(points), points=points, weights=weights,
                  sites=sites, k=profile["k"], radius=radius, penalty=penalty)
    return Instance(digest=compute_digest(Instance(digest="", **fields)), **fields)


def _entry(name, seed, family, n, m, k, attempt):
    return SuiteEntry(name, seed, {**family, "n": n, "m": m, "k": k, "attempt": attempt})


PUBLIC_SUITE = (
    _entry("monitors-01", 2101, UNIFORM, 120, 30, 4, 197),
    _entry("monitors-02", 2102, UNIFORM, 200, 50, 6, 0),
    _entry("monitors-03", 2103, UNIFORM, 300, 80, 8, 5),
    _entry("monitors-04", 2104, UNIFORM, 400, 110, 10, 3),
    _entry("monitors-05", 2105, CLUSTERED, 250, 70, 14, 63),
    _entry("monitors-06", 2106, CLUSTERED, 400, 100, 20, 13),
)
SELF_CHECK_NAMES = ("monitors-01", "monitors-05")


def budget_for(instance) -> float:
    return 5.0

# Monitor Placement

Place k air-quality monitors so that people are measured close to where they live.

## The scenario

Tomasz Wierzbicki has a grant for a handful of air-quality monitors across
Lindenfeld. The city council gave him a list of rooftops he may use, not all
of them near where people actually live. A household far from every monitor
is as good as unmeasured, so beyond a certain distance it counts as missed.

## The problem, precisely

- **Demand points:** `instance.size` points `instance.points[i] = (x, y)` on a
  0..1000 grid, each with a population `instance.weights[i]`.
- **Candidate sites:** `instance.sites[s] = (x, y)`. They are not demand
  points in general.
- **Choose:** exactly `instance.k` distinct candidate ids.
- **Distance:** `isqrt(dx^2 + dy^2)`. A distance above the coverage radius
  `instance.radius` (D) counts as the penalty `instance.penalty` (P > D).
- **Plan:** `{"sites": [s, ...]}`.
- **Cost:** sum over points of `population x min(P, distance to the nearest
  chosen site)` with the rule above.

## How you're scored

Your best valid cost is read at 5%, 20%, 50% and 100% of the 5-second budget
(weights 0.10, 0.20, 0.30, 0.40). At each checkpoint the cost is placed on a
curve through three anchors: the starter scores 0.25, the published baseline
0.50, the reference 1.00, linearly in between; beating the reference scores
1.00, no valid plan scores 0.

*Example:* baseline 3,607,578, reference 3,372,943 at a checkpoint. A cost of
3,490,260 is halfway between them and scores 0.75 there.

Points: quality 70, robustness 20 (shifted instances, halved if one family is
strong and another weak), engineering 10 (valid plans, no crashes or
overruns, a plan by the first checkpoint).

## Your submission

```python
from adapter import Solver

class MySolver(Solver):
    def solve(self, instance, submit_candidate):
        receipt = submit_candidate({"sites": [...]})   # any number of times
        return {"sites": [...]}
```

Put it in `adapters/mine.py`, then:

```
python self_check.py --adapter adapters.mine:MySolver
python run.py --adapter adapters.mine:MySolver --out report.json
```

## The trap

Read this before you write anything: **candidate sites are not demand
points.** The best spot for a cluster may not be a candidate at all, and a
point beyond the radius costs the full penalty however close the nearest
monitor is to the radius.

## Baselines

| Anchor | Program | Score |
| --- | --- | --- |
| Starter | k random candidates | 0.25 |
| Baseline | greedy add, then the single best swap | 0.50 |
| Reference | a stronger search, not published | 1.00 |

## Your head start

`adapters/starter.py` is a working solver. `adapter.py` gives you
`dist_table(instance)` (the charged distance from every point to every
candidate, built once) and `assign(instance, sites) -> (cost, nearest)`.

## A hint

Adding sites greedily is a good start, not a finish.

## Instance families

| Family | Public | Private | What changes |
| --- | --- | --- | --- |
| uniform | 4 | 4 | points spread evenly, even populations, small k |
| clustered (shifted) | 2 | 4 | dense clusters, heavy-tailed populations, few candidates inside clusters, large k |

Sizes: 120 to 400 points, 25 to 110 candidates. The private suite uses unseen seeds.

## Files

`adapter.py`, `adapters/starter.py`, `adapters/baseline.py`, `data.py`,
`validator.py`, `self_check.py`, `run.py`, `public_reference.json`,
`benchkit/`, `SECURITY.md`.

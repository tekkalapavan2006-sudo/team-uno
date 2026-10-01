"""Bench config for Monitor Placement."""

import _paths  # noqa: F401

import data
import validator
from benchkit import BenchConfig

BENCH = BenchConfig(
    name="monitors",
    bench_dir=_paths.BENCH_DIR,
    data=data,
    validate=validator.validate,
    probe="private.probes.textbook:TextbookProbe",
    shift_markers=lambda p: "clustered, heavy-tailed, large k" if p["shifted"] else "",
)

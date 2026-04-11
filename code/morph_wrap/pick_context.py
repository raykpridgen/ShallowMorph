#!/usr/bin/env python3
"""
Pick per-dataset context_frames for Sweep B from Sweep A rollout metrics (I6-A).

Expected ``rollout`` CSV columns (header row):
  run_id, rollout_step, mse, dataset, context_frames

Optional: ``ssim``. Rows are aggregated with mean MSE over ``1 <= rollout_step <= k_max``.

Example::

    PYTHONPATH=code python -m morph_wrap.pick_context \\
        --rollout-csv out/sweepA_rollout.csv \\
        --out out/sweep_b_context.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Tuple

from morph_wrap.sweep_config import DATASET_ORDER, DatasetKey


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("inf")


def pick_context(
    rows: List[Dict[str, str]],
    *,
    k_max: int | None,
) -> Dict[DatasetKey, int]:
    # (dataset, context_frames) -> list of mse values (already filtered by step)
    bucket: DefaultDict[Tuple[str, int], List[float]] = defaultdict(list)
    for r in rows:
        try:
            ds = r["dataset"].strip()
            n = int(r["context_frames"])
            step = int(r["rollout_step"])
            mse = float(r["mse"])
        except (KeyError, ValueError):
            continue
        if k_max is not None and (step < 1 or step > k_max):
            continue
        if step < 1:
            continue
        bucket[(ds, n)].append(mse)

    best: Dict[DatasetKey, int] = {}
    for ds in DATASET_ORDER:
        candidates = [n for (d, n) in bucket if d == ds]
        if not candidates:
            raise ValueError(f"No rollout rows for dataset {ds}")
        uniq_n = sorted(set(candidates))
        means = {n: _mean(bucket[(ds, n)]) for n in uniq_n}
        best_n = min(uniq_n, key=lambda n: means[n])
        best[ds] = best_n
    return best


def main() -> int:
    p = argparse.ArgumentParser(description="Pick Sweep B context from rollout CSV")
    p.add_argument("--rollout-csv", type=Path, required=True)
    p.add_argument("--k-max", type=int, default=None, help="Max rollout_step to include")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    with args.rollout_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    try:
        best = pick_context(rows, k_max=args.k_max)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(best, indent=2) + "\n")
    print(json.dumps(best, indent=2))
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

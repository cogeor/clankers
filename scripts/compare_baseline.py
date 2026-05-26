#!/usr/bin/env python3
"""Compare a fresh bench CSV against a committed baseline.

Usage:

    python scripts/compare_baseline.py CURRENT BASELINE [--tolerance 0.15]

Exits 0 when every row's `steps_per_sec_mean` is within `tolerance`
fraction of the baseline; exits 1 on any
`(baseline - current) / baseline > tolerance` regression.

Key columns (matched by header name, so additive schema changes do not
break the comparator):

  - `scenario`
  - `num_envs` (optional; defaults to "0" when absent — matches the
    W5 PR4 11-column schema where the row is keyed by scenario alone)
  - `steps_per_sec_mean`

Host metadata (`notes` column) is ignored: baseline rows recorded on
the maintainer's local machine and CI rows recorded on a Linux GHA
runner can differ in absolute numbers, but the comparator's job is
only to flag regression relative to the *committed* baseline. CI's
own baseline is regenerated on the first green merge to absorb the
Linux-vs-Windows drift; the W7 PR4 commit seeds the file.

Stdlib-only (csv + argparse + sys) — no third-party deps.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def _load(path: Path) -> dict[tuple[str, str], float]:
    """Return `{(scenario, num_envs): steps_per_sec_mean}` keyed by header name."""
    rows: dict[tuple[str, str], float] = {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["scenario"], row.get("num_envs", "0"))
            try:
                value = float(row["steps_per_sec_mean"])
            except (KeyError, ValueError):
                continue
            rows[key] = value
    return rows


def compare(current: Path, baseline: Path, tolerance: float) -> int:
    """Compare current vs baseline.

    Returns 0 if every baseline row's `steps_per_sec_mean` is within
    `tolerance` fraction of the current value; 1 otherwise.
    """
    cur = _load(current)
    base = _load(baseline)
    if not base:
        print(f"compare_baseline: baseline {baseline} has no rows", file=sys.stderr)
        return 1

    worst = 0.0
    failures: list[str] = []
    for key, b in base.items():
        c = cur.get(key)
        if c is None:
            failures.append(f"missing row: scenario={key[0]} num_envs={key[1]}")
            continue
        if b <= 0.0:
            # No regression possible against a zero/negative baseline.
            continue
        regression = (b - c) / b
        if regression > worst:
            worst = regression
        if regression > tolerance:
            failures.append(
                f"REGRESSION scenario={key[0]} num_envs={key[1]}: "
                f"baseline={b:.1f}, current={c:.1f} ({regression:.1%} slower)"
            )

    for line in failures:
        print(line, file=sys.stderr)

    if worst > tolerance or failures:
        print(
            f"compare_baseline: worst regression {worst:.1%} exceeds tolerance {tolerance:.1%}",
            file=sys.stderr,
        )
        return 1
    print(
        f"compare_baseline: OK (worst regression {worst:.1%}, tolerance {tolerance:.1%})"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("current", type=Path, help="Fresh bench CSV to validate.")
    ap.add_argument(
        "baseline", type=Path, help="Committed baseline CSV under notes/baselines/."
    )
    ap.add_argument(
        "--tolerance",
        type=float,
        default=0.15,
        help="Fractional regression allowed before failure (default 0.15 = 15%%).",
    )
    ns = ap.parse_args()
    return compare(ns.current, ns.baseline, ns.tolerance)


if __name__ == "__main__":
    sys.exit(main())

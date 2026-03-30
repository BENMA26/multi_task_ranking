#!/usr/bin/env python3
"""
Aggregate multi-seed subset search results.

Input: per-run CSV (one row per setting+seed), e.g. subset_search_summary.csv
Output: grouped CSV (one row per base setting without `_s<seed>` suffix)
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


SEED_SUFFIX_RE = re.compile(r"_s(\d+)$")


def to_float(v: str) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except Exception:
        return None


def mean_std(vals: list[float]) -> tuple[float | None, float | None]:
    if not vals:
        return None, None
    mu = sum(vals) / len(vals)
    if len(vals) == 1:
        return mu, 0.0
    var = sum((x - mu) ** 2 for x in vals) / (len(vals) - 1)
    return mu, math.sqrt(var)


def stage_from_name(name: str) -> str:
    if name.startswith("subset_normal_"):
        return "normal"
    if name.startswith("subset_entropy_"):
        return "entropy"
    if name.startswith("subset_asym_"):
        return "asym"
    if name.startswith("subset_jd_"):
        return "jd"
    return "other"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate subset-search metrics across seeds")
    p.add_argument(
        "--input_csv",
        type=Path,
        default=Path("experiments/final_ali_ccp/subset_search_summary.csv"),
    )
    p.add_argument(
        "--out_csv",
        type=Path,
        default=Path("experiments/final_ali_ccp/subset_search_seed_agg.csv"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_csv.exists():
        raise FileNotFoundError(f"missing input csv: {args.input_csv}")

    groups: dict[str, dict] = {}
    with args.input_csv.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            exp = row.get("experiment", "")
            if not exp.startswith("subset_"):
                continue
            m = SEED_SUFFIX_RE.search(exp)
            base = exp
            seed = ""
            if m:
                seed = m.group(1)
                base = exp[: m.start()]

            g = groups.setdefault(
                base,
                {
                    "setting": base,
                    "stage": stage_from_name(base),
                    "runs": 0,
                    "ok_runs": 0,
                    "seeds": [],
                    "val_vals": [],
                    "test_vals": [],
                },
            )
            g["runs"] += 1
            if seed:
                g["seeds"].append(seed)

            if row.get("status") == "OK":
                g["ok_runs"] += 1

            v = to_float(row.get("val_combined_auc", ""))
            t = to_float(row.get("test_combined_auc", ""))
            if v is not None:
                g["val_vals"].append(v)
            if t is not None:
                g["test_vals"].append(t)

    out_rows = []
    for base, g in groups.items():
        val_mean, val_std = mean_std(g["val_vals"])
        test_mean, test_std = mean_std(g["test_vals"])
        out_rows.append(
            {
                "setting": base,
                "stage": g["stage"],
                "runs": g["runs"],
                "ok_runs": g["ok_runs"],
                "seeds": ",".join(sorted(set(g["seeds"]))),
                "val_combined_auc_mean": "" if val_mean is None else f"{val_mean:.6f}",
                "val_combined_auc_std": "" if val_std is None else f"{val_std:.6f}",
                "test_combined_auc_mean": "" if test_mean is None else f"{test_mean:.6f}",
                "test_combined_auc_std": "" if test_std is None else f"{test_std:.6f}",
            }
        )

    out_rows.sort(
        key=lambda r: (
            float(r["val_combined_auc_mean"]) if r["val_combined_auc_mean"] else float("-inf")
        ),
        reverse=True,
    )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "setting",
        "stage",
        "runs",
        "ok_runs",
        "seeds",
        "val_combined_auc_mean",
        "val_combined_auc_std",
        "test_combined_auc_mean",
        "test_combined_auc_std",
    ]
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    print(f"[OK] wrote {len(out_rows)} grouped settings to {args.out_csv}")
    if out_rows:
        top = out_rows[0]
        print(
            "[TOP] "
            f"{top['setting']} val_mean={top['val_combined_auc_mean']} "
            f"test_mean={top['test_combined_auc_mean']}"
        )


if __name__ == "__main__":
    main()

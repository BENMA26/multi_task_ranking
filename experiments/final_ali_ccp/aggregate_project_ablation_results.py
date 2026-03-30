#!/usr/bin/env python3
"""
Aggregate project-specific ablation results across seeds and output:
1) grouped metrics per ablation setting
2) best setting per model
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


def infer_model_variant(setting: str) -> tuple[str, str]:
    if setting.startswith("projab_share_bottom_"):
        return "share_bottom", setting[len("projab_share_bottom_") :]
    if setting.startswith("projab_mmoe_"):
        return "mmoe", setting[len("projab_mmoe_") :]
    if setting.startswith("projab_ple_"):
        return "ple", setting[len("projab_ple_") :]
    return "other", "other"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate project ablation results")
    p.add_argument(
        "--input_csv",
        type=Path,
        default=Path("experiments/final_ali_ccp/outputs/project_compare/ablation/ablation_summary.csv"),
    )
    p.add_argument(
        "--out_csv",
        type=Path,
        default=Path("experiments/final_ali_ccp/outputs/project_compare/ablation/ablation_grouped.csv"),
    )
    p.add_argument(
        "--out_best_csv",
        type=Path,
        default=Path("experiments/final_ali_ccp/outputs/project_compare/ablation/ablation_best_by_model.csv"),
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
            if not exp.startswith("projab_"):
                continue
            m = SEED_SUFFIX_RE.search(exp)
            base = exp
            seed = ""
            if m:
                seed = m.group(1)
                base = exp[: m.start()]

            model, variant = infer_model_variant(base)
            g = groups.setdefault(
                base,
                {
                    "setting": base,
                    "model": model,
                    "variant": variant,
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
                "model": g["model"],
                "variant": g["variant"],
                "runs": g["runs"],
                "ok_runs": g["ok_runs"],
                "seeds": ",".join(sorted(set(g["seeds"]))),
                "val_combined_auc_mean": "" if val_mean is None else f"{val_mean:.6f}",
                "val_combined_auc_std": "" if val_std is None else f"{val_std:.6f}",
                "test_combined_auc_mean": "" if test_mean is None else f"{test_mean:.6f}",
                "test_combined_auc_std": "" if test_std is None else f"{test_std:.6f}",
            }
        )

    def mean_for_sort(row: dict) -> float:
        v = row.get("val_combined_auc_mean", "")
        return float(v) if v else float("-inf")

    out_rows.sort(key=lambda r: (r["model"], -mean_for_sort(r)))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "setting",
        "model",
        "variant",
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

    best_rows: list[dict] = []
    best_map: dict[str, dict] = {}
    for row in out_rows:
        model = row["model"]
        if model not in best_map:
            best_map[model] = row
            best_rows.append(row)

    with args.out_best_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(best_rows)

    print(f"[OK] grouped rows: {len(out_rows)} -> {args.out_csv}")
    print(f"[OK] best rows   : {len(best_rows)} -> {args.out_best_csv}")
    for row in best_rows:
        print(
            "[BEST] "
            f"model={row['model']} setting={row['setting']} "
            f"val_mean={row['val_combined_auc_mean']}"
        )


if __name__ == "__main__":
    main()

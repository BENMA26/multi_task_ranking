#!/usr/bin/env python3
"""
从 experiments/final_ali_ccp/logs 下的 Slurm 日志中提取关键指标并汇总为 CSV。
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Iterable


METRIC_KEYS = (
    "val_ctr_auc",
    "val_cvr_auc",
    "val_combined_auc",
    "test_ctr_auc",
    "test_cvr_auc",
    "test_combined_auc",
)

TABLE_METRIC_RE = re.compile(r"[|│]\s*([a-z_]+)\s*[|│]\s*([-+]?\d*\.?\d+)\s*[|│]")
INLINE_METRIC_RE = re.compile(r"\b([a-z_]+)\s*[=:]\s*([-+]?\d*\.?\d+)")
EXP_RE = re.compile(r"^Experiment:\s*(.+?)\s*$")
STATUS_RE = re.compile(r"^\[(OK|ERROR)\]\s+(.+?)\s+(completed|failed)\.?$")


def parse_log(log_path: Path) -> Dict[str, Dict[str, str]]:
    exp_records: Dict[str, Dict[str, str]] = {}
    current_exp: str | None = None

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            m_exp = EXP_RE.match(line)
            if m_exp:
                current_exp = m_exp.group(1).strip()
                exp_records.setdefault(
                    current_exp,
                    {
                        "experiment": current_exp,
                        "status": "UNKNOWN",
                        "log_file": str(log_path),
                    },
                )
                continue

            m_status = STATUS_RE.match(line)
            if m_status:
                status = m_status.group(1)
                exp_name = m_status.group(2).strip()
                exp_records.setdefault(
                    exp_name,
                    {
                        "experiment": exp_name,
                        "status": "UNKNOWN",
                        "log_file": str(log_path),
                    },
                )
                exp_records[exp_name]["status"] = status
                continue

            if current_exp is None:
                continue

            rec = exp_records[current_exp]
            for m in TABLE_METRIC_RE.finditer(line):
                key, value = m.group(1), m.group(2)
                if key in METRIC_KEYS:
                    rec[key] = value

            for m in INLINE_METRIC_RE.finditer(line):
                key, value = m.group(1), m.group(2)
                if key in METRIC_KEYS:
                    rec[key] = value

    return exp_records


def collect_records(log_files: Iterable[Path]) -> Dict[str, Dict[str, str]]:
    all_records: Dict[str, Dict[str, str]] = {}
    for log_path in log_files:
        parsed = parse_log(log_path)
        for exp_name, record in parsed.items():
            all_records[exp_name] = record
    return all_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize final Ali-CCP experiment logs")
    parser.add_argument(
        "--log_dir",
        type=Path,
        default=Path("experiments/final_ali_ccp/logs"),
        help="日志目录（默认 experiments/final_ali_ccp/logs）",
    )
    parser.add_argument(
        "--out_csv",
        type=Path,
        default=Path("experiments/final_ali_ccp/ali_ccp_summary.csv"),
        help="输出 CSV 路径",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.log_dir.exists():
        raise FileNotFoundError(f"log dir not found: {args.log_dir}")

    log_files = sorted(
        [p for p in args.log_dir.glob("*.out")] + [p for p in args.log_dir.glob("*.err")]
    )
    if not log_files:
        raise RuntimeError(f"no log files found under {args.log_dir}")

    records = collect_records(log_files)
    rows = sorted(records.values(), key=lambda x: x["experiment"])

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["experiment", "status", *METRIC_KEYS, "log_file"]
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"[OK] Wrote {len(rows)} experiment rows to {args.out_csv}")


if __name__ == "__main__":
    main()

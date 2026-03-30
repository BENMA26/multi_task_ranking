#!/bin/bash
# One-click project pipeline:
# 1) hyperparameter search
# 2) ablation experiments based on HP best settings

set -euo pipefail

PROJECT_ROOT=/work/home/maben/project/rec_sys/projects/multi_task_ranking
cd "${PROJECT_ROOT}"

echo "[PIPELINE] Step 1/2: Hyperparameter Search"
bash experiments/final_ali_ccp/run_project_hp_search.sh

echo "[PIPELINE] Step 2/2: Ablation Experiments"
bash experiments/final_ali_ccp/run_project_ablation.sh

echo "[PIPELINE] Done."
echo "[PIPELINE] HP grouped:"
echo "  experiments/final_ali_ccp/outputs/project_compare/hp/hp_search_grouped.csv"
echo "[PIPELINE] Ablation grouped:"
echo "  experiments/final_ali_ccp/outputs/project_compare/ablation/ablation_grouped.csv"

#!/bin/bash
#SBATCH --job-name=mtl_adaftr_best
#SBATCH --output=logs/mtl_adaftr_best_%j.out
#SBATCH --error=logs/mtl_adaftr_best_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:4
#SBATCH --mem=96G
#SBATCH --time=144:00:00
#SBATCH --partition=normal
#SBATCH --exclude=g01n01

set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR"

TRAIN_PY=/work/home/maben/project/rec_sys/projects/multi_task_ranking/train_rank.py
OUTPUT_ROOT=/work/home/maben/project/rec_sys/projects/multi_task_ranking/experiments/adaftr_best_param_experiment/outputs

EXP_NAME="adaftr_esmm_best_quicktune"

echo "=========================================="
echo "Experiment: ${EXP_NAME}"
echo "Model: adaftr | ESMM: true"
echo "Best params: alpha=0.01, tau_min=0.08, tau_max=0.50"
echo "=========================================="

python -u ${TRAIN_PY} \
    --model adaftr \
    --batch_size 1024 \
    --num_workers 32 \
    --max_epochs 20 \
    --learning_rate 7e-4 \
    --dropout 0.25 \
    --embedding_dim 32 \
    --expert_hidden_dims 256 128 \
    --num_experts 6 \
    --adaftr_tower_hidden_dims 256 128 64 \
    --alpha_contrastive 0.01 \
    --tau_min 0.08 \
    --tau_max 0.50 \
    --relatedness_hidden_dim 64 \
    --lambda_rel 0.1 \
    --accelerator gpu \
    --devices 4 \
    --esmm \
    --sigmoid 1 \
    --strategy ddp_find_unused_parameters_false \
    --early_stop_patience 3 \
    --exp_dir ${OUTPUT_ROOT}/${EXP_NAME}

echo "[OK] ${EXP_NAME} completed."

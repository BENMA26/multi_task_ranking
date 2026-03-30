#!/bin/bash
#SBATCH --job-name=mtl_hard_sample
#SBATCH --output=logs/mtl_hard_%j.out
#SBATCH --error=logs/mtl_hard_%j.err
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
OUTPUT_ROOT=/work/home/maben/project/rec_sys/projects/multi_task_ranking/experiments/hard_sample_discovery/outputs

COMMON_ARGS="
    --model mmoe
    --batch_size 1024
    --num_workers 32
    --max_epochs 20
    --learning_rate 7e-4
    --dropout 0.25
    --embedding_dim 32
    --expert_hidden_dims 256 128
    --num_experts 6
    --tower_hidden_dims 64
    --accelerator gpu
    --devices 4
    --esmm
    --sigmoid 1
    --strategy ddp_find_unused_parameters_false
    --early_stop_patience 3
"

run_exp() {
    local name=$1
    local extra=$2
    echo "=========================================="
    echo "Experiment: ${name}"
    echo "=========================================="
    python -u ${TRAIN_PY} ${COMMON_ARGS} ${extra} \
        --exp_dir ${OUTPUT_ROOT}/${name}

    if [ $? -ne 0 ]; then
        echo "[ERROR] ${name} failed! Continuing..."
    else
        echo "[OK] ${name} completed."
    fi
    echo ""
}

# 1) MMOE baseline（无 hard sample）
run_exp "mmoe_esmm_baseline" ""

# 2) Hard sample: 低比例、温和权重
run_exp "mmoe_esmm_hard_ratio10_w15" "
    --use_hard_sample
    --hard_sample_ratio 0.10
    --hard_sample_weight 1.5
    --hard_sample_warmup_epochs 1
"

# 3) Hard sample: 中等比例、标准权重
run_exp "mmoe_esmm_hard_ratio20_w20" "
    --use_hard_sample
    --hard_sample_ratio 0.20
    --hard_sample_weight 2.0
    --hard_sample_warmup_epochs 1
"

# 4) Hard sample: 高比例、更强权重
run_exp "mmoe_esmm_hard_ratio30_w25" "
    --use_hard_sample
    --hard_sample_ratio 0.30
    --hard_sample_weight 2.5
    --hard_sample_warmup_epochs 1
"

echo "All hard sample discovery experiments finished."

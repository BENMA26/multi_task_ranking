#!/bin/bash
#SBATCH --job-name=mtl_dcnv2
#SBATCH --output=logs/mtl_dcnv2_%j.out
#SBATCH --error=logs/mtl_dcnv2_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:4
#SBATCH --mem=96G
#SBATCH --time=144:00:00
#SBATCH --partition=normal
#SBATCH --exclude=g01n01

mkdir -p logs

TRAIN_PY=/work/home/maben/project/rec_sys/projects/multi_task_ranking/train_rank.py
OUTPUT_ROOT=/work/home/maben/project/rec_sys/projects/multi_task_ranking/experiments/dcn_v2_feature_cross/outputs

# 公共超参（与 entropy_regularization 实验保持一致，便于横向对比）
COMMON_ARGS="
    --batch_size 1024
    --num_workers 32
    --max_epochs 20
    --learning_rate 1e-3
    --dropout 0.2
    --embedding_dim 32
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

# ── 实验 1: ShareBottom (baseline) ───────────────────────────────────────
run_exp "share_bottom_baseline" "
    --model share_bottom
    --shared_hidden_dims 256 128
"

# ── 实验 2: ShareBottom + DCN-v2 ─────────────────────────────────────────
run_exp "share_bottom_dcnv2" "
    --model share_bottom
    --shared_hidden_dims 256 128
    --use_dcn --dcn_num_layers 2
"

# ── 实验 3: MMOE (baseline) ──────────────────────────────────────────────
run_exp "mmoe_esmm_baseline" "
    --model mmoe
    --expert_hidden_dims 256 128
    --num_experts 6
"

# ── 实验 4: MMOE + DCN-v2 ────────────────────────────────────────────────
run_exp "mmoe_esmm_dcnv2" "
    --model mmoe
    --expert_hidden_dims 256 128
    --num_experts 6
    --use_dcn --dcn_num_layers 2
"

echo "All 4 DCN-v2 experiments finished."

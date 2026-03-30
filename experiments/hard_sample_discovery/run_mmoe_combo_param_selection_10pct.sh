#!/bin/bash
#SBATCH --job-name=mtl_mmoe_combo_sel10
#SBATCH --output=/work/home/maben/project/rec_sys/projects/multi_task_ranking/experiments/hard_sample_discovery/logs/mtl_mmoe_combo_sel10_%j.out
#SBATCH --error=/work/home/maben/project/rec_sys/projects/multi_task_ranking/experiments/hard_sample_discovery/logs/mtl_mmoe_combo_sel10_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:4
#SBATCH --mem=96G
#SBATCH --time=144:00:00
#SBATCH --partition=normal
#SBATCH --exclude=g01n01

set -u

PROJECT_ROOT=/work/home/maben/project/rec_sys/projects/multi_task_ranking
LOG_DIR=${PROJECT_ROOT}/experiments/hard_sample_discovery/logs

cd "${PROJECT_ROOT}"
mkdir -p "${LOG_DIR}"

TRAIN_PY=${PROJECT_ROOT}/train_rank.py
OUTPUT_ROOT=${PROJECT_ROOT}/experiments/hard_sample_discovery/outputs_param_select_10pct
mkdir -p "${OUTPUT_ROOT}"

# 10% 子集用于参数选择（分层采样，保证标签结构稳定）
COMMON_ARGS="
    --model mmoe
    --batch_size 1024
    --num_workers 32
    --max_epochs 8
    --accelerator gpu
    --devices 4
    --strategy ddp_find_unused_parameters_false
    --early_stop_patience 2
    --esmm
    --sigmoid 1
    --train_subset_frac 0.1
    --train_subset_stratify
"

run_exp() {
    local name=$1
    local extra=$2
    echo "=========================================="
    echo "Experiment: ${name}"
    echo "=========================================="
    if python -u "${TRAIN_PY}" ${COMMON_ARGS} ${extra} --exp_dir "${OUTPUT_ROOT}/${name}"; then
        echo "[OK] ${name} completed."
    else
        echo "[ERROR] ${name} failed! Continue next."
    fi
    echo ""
}

# ------------------------------------------------------------
# Principled Selection Strategy (staged OAT sweeps)
# ------------------------------------------------------------
# Stage A: optimization (learning rate)
for LR in 3e-4 7e-4 1e-3; do
    tag=${LR/./p}
    run_exp "sel10_stageA_opt_lr${tag}" "
        --learning_rate ${LR}
        --dropout 0.25
        --num_experts 4
        --expert_hidden_dims 256 128
        --tower_hidden_dims 64
        --use_hard_sample
        --hard_sample_ratio 0.10
        --hard_sample_weight 1.5
        --hard_sample_warmup_epochs 1
        --use_entropy_reg
        --lambda_entropy 0.01
        --gradient_clip_val 1.0
        --gradient_clip_algorithm norm
    "
done

# Stage B: regularization (dropout)
for DROPOUT in 0.15 0.25 0.35; do
    tag=${DROPOUT/./p}
    run_exp "sel10_stageB_reg_dropout${tag}" "
        --learning_rate 7e-4
        --dropout ${DROPOUT}
        --num_experts 4
        --expert_hidden_dims 256 128
        --tower_hidden_dims 64
        --use_hard_sample
        --hard_sample_ratio 0.10
        --hard_sample_weight 1.5
        --hard_sample_warmup_epochs 1
        --use_entropy_reg
        --lambda_entropy 0.01
        --gradient_clip_val 1.0
        --gradient_clip_algorithm norm
    "
done

# Stage C: capacity (num_experts)
for EXPERTS in 4 6 8; do
    run_exp "sel10_stageC_cap_e${EXPERTS}" "
        --learning_rate 7e-4
        --dropout 0.25
        --num_experts ${EXPERTS}
        --expert_hidden_dims 256 128
        --tower_hidden_dims 64
        --use_hard_sample
        --hard_sample_ratio 0.10
        --hard_sample_weight 1.5
        --hard_sample_warmup_epochs 1
        --use_entropy_reg
        --lambda_entropy 0.01
        --gradient_clip_val 1.0
        --gradient_clip_algorithm norm
    "
done

# Stage D: hard-sample discovery
for RATIO in 0.05 0.10 0.20; do
    tag=${RATIO/./p}
    run_exp "sel10_stageD_hard_ratio${tag}" "
        --learning_rate 7e-4
        --dropout 0.25
        --num_experts 4
        --expert_hidden_dims 256 128
        --tower_hidden_dims 64
        --use_hard_sample
        --hard_sample_ratio ${RATIO}
        --hard_sample_weight 1.5
        --hard_sample_warmup_epochs 1
        --use_entropy_reg
        --lambda_entropy 0.01
        --gradient_clip_val 1.0
        --gradient_clip_algorithm norm
    "
done

for WEIGHT in 1.2 1.5 2.0; do
    tag=${WEIGHT/./p}
    run_exp "sel10_stageD_hard_weight${tag}" "
        --learning_rate 7e-4
        --dropout 0.25
        --num_experts 4
        --expert_hidden_dims 256 128
        --tower_hidden_dims 64
        --use_hard_sample
        --hard_sample_ratio 0.10
        --hard_sample_weight ${WEIGHT}
        --hard_sample_warmup_epochs 1
        --use_entropy_reg
        --lambda_entropy 0.01
        --gradient_clip_val 1.0
        --gradient_clip_algorithm norm
    "
done

# Stage E: entropy regularization strength
for ENT in 0.005 0.01 0.02; do
    tag=${ENT/./p}
    run_exp "sel10_stageE_ent${tag}" "
        --learning_rate 7e-4
        --dropout 0.25
        --num_experts 4
        --expert_hidden_dims 256 128
        --tower_hidden_dims 64
        --use_hard_sample
        --hard_sample_ratio 0.10
        --hard_sample_weight 1.5
        --hard_sample_warmup_epochs 1
        --use_entropy_reg
        --lambda_entropy ${ENT}
        --gradient_clip_val 1.0
        --gradient_clip_algorithm norm
    "
done

# Stage F: gradient clipping strength
for CLIP in 0.5 1.0 2.0; do
    tag=${CLIP/./p}
    run_exp "sel10_stageF_clip${tag}" "
        --learning_rate 7e-4
        --dropout 0.25
        --num_experts 4
        --expert_hidden_dims 256 128
        --tower_hidden_dims 64
        --use_hard_sample
        --hard_sample_ratio 0.10
        --hard_sample_weight 1.5
        --hard_sample_warmup_epochs 1
        --use_entropy_reg
        --lambda_entropy 0.01
        --gradient_clip_val ${CLIP}
        --gradient_clip_algorithm norm
    "
done

echo "All 10%-subset principled parameter-selection runs finished."

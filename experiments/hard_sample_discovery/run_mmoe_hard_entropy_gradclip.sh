#!/bin/bash
#SBATCH --job-name=mtl_mmoe_hard_ent_clip
#SBATCH --output=/work/home/maben/project/rec_sys/projects/multi_task_ranking/experiments/hard_sample_discovery/logs/mtl_mmoe_hard_ent_clip_%j.out
#SBATCH --error=/work/home/maben/project/rec_sys/projects/multi_task_ranking/experiments/hard_sample_discovery/logs/mtl_mmoe_hard_ent_clip_%j.err
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
OUTPUT_ROOT=${PROJECT_ROOT}/experiments/hard_sample_discovery/outputs_combo
mkdir -p "${OUTPUT_ROOT}"

# 可通过环境变量覆盖默认值
HARD_RATIO=${HARD_RATIO:-0.10}
HARD_WEIGHT=${HARD_WEIGHT:-1.5}
HARD_WARMUP_EPOCHS=${HARD_WARMUP_EPOCHS:-1}
LAMBDA_ENTROPY=${LAMBDA_ENTROPY:-0.01}
GRAD_CLIP_VAL=${GRAD_CLIP_VAL:-1.0}
GRAD_CLIP_ALGO=${GRAD_CLIP_ALGO:-norm}
NUM_EXPERTS=${NUM_EXPERTS:-4}

ratio_tag=${HARD_RATIO/./p}
weight_tag=${HARD_WEIGHT/./p}
ent_tag=${LAMBDA_ENTROPY/./p}
clip_tag=${GRAD_CLIP_VAL/./p}
experts_tag=${NUM_EXPERTS}

EXP_NAME="mmoe_esmm_hard_r${ratio_tag}_w${weight_tag}_ent${ent_tag}_clip${clip_tag}_e${experts_tag}"

echo "=========================================="
echo "Experiment: ${EXP_NAME}"
echo "Model: mmoe + hard sample + entropy + gradient clipping"
echo "hard_ratio=${HARD_RATIO}, hard_weight=${HARD_WEIGHT}, hard_warmup=${HARD_WARMUP_EPOCHS}"
echo "lambda_entropy=${LAMBDA_ENTROPY}, grad_clip_val=${GRAD_CLIP_VAL}, grad_clip_algo=${GRAD_CLIP_ALGO}"
echo "num_experts=${NUM_EXPERTS}"
echo "=========================================="

python -u "${TRAIN_PY}" \
    --model mmoe \
    --batch_size 1024 \
    --num_workers 32 \
    --max_epochs 20 \
    --learning_rate 7e-4 \
    --dropout 0.25 \
    --embedding_dim 32 \
    --expert_hidden_dims 256 128 \
    --num_experts "${NUM_EXPERTS}" \
    --tower_hidden_dims 64 \
    --accelerator gpu \
    --devices 4 \
    --esmm \
    --sigmoid 1 \
    --strategy ddp_find_unused_parameters_false \
    --early_stop_patience 3 \
    --use_hard_sample \
    --hard_sample_ratio "${HARD_RATIO}" \
    --hard_sample_weight "${HARD_WEIGHT}" \
    --hard_sample_warmup_epochs "${HARD_WARMUP_EPOCHS}" \
    --use_entropy_reg \
    --lambda_entropy "${LAMBDA_ENTROPY}" \
    --gradient_clip_val "${GRAD_CLIP_VAL}" \
    --gradient_clip_algorithm "${GRAD_CLIP_ALGO}" \
    --exp_dir "${OUTPUT_ROOT}/${EXP_NAME}"

if [ $? -ne 0 ]; then
    echo "[ERROR] ${EXP_NAME} failed."
    exit 1
fi

echo "[OK] ${EXP_NAME} completed."

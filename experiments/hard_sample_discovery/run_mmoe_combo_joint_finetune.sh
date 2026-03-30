#!/bin/bash
#SBATCH --job-name=mtl_mmoe_joint_ft
#SBATCH --output=/work/home/maben/project/rec_sys/projects/multi_task_ranking/experiments/hard_sample_discovery/logs/mtl_mmoe_joint_ft_%j.out
#SBATCH --error=/work/home/maben/project/rec_sys/projects/multi_task_ranking/experiments/hard_sample_discovery/logs/mtl_mmoe_joint_ft_%j.err
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
TRAIN_PY=${PROJECT_ROOT}/train_rank.py
OUTPUT_ROOT=${PROJECT_ROOT}/experiments/hard_sample_discovery/outputs_joint_finetune

cd "${PROJECT_ROOT}"
mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_ROOT}"

# Joint-finetune defaults from 10% principled selection
LEARNING_RATE=${LEARNING_RATE:-3e-4}
DROPOUT=${DROPOUT:-0.25}
NUM_EXPERTS=${NUM_EXPERTS:-6}
HARD_RATIO=${HARD_RATIO:-0.20}
HARD_WEIGHT=${HARD_WEIGHT:-1.5}
HARD_WARMUP_EPOCHS=${HARD_WARMUP_EPOCHS:-1}
LAMBDA_ENTROPY=${LAMBDA_ENTROPY:-0.01}
GRAD_CLIP_VAL=${GRAD_CLIP_VAL:-2.0}
GRAD_CLIP_ALGO=${GRAD_CLIP_ALGO:-norm}
MAX_EPOCHS=${MAX_EPOCHS:-20}
PATIENCE=${PATIENCE:-3}

lr_tag=${LEARNING_RATE/./p}
drop_tag=${DROPOUT/./p}
ratio_tag=${HARD_RATIO/./p}
weight_tag=${HARD_WEIGHT/./p}
ent_tag=${LAMBDA_ENTROPY/./p}
clip_tag=${GRAD_CLIP_VAL/./p}

EXP_NAME="mmoe_joint_ft_lr${lr_tag}_do${drop_tag}_e${NUM_EXPERTS}_hr${ratio_tag}_hw${weight_tag}_ent${ent_tag}_clip${clip_tag}"

echo "=========================================="
echo "Experiment: ${EXP_NAME}"
echo "Model: mmoe + hard sample + entropy + gradient clipping (joint finetune)"
echo "lr=${LEARNING_RATE}, dropout=${DROPOUT}, num_experts=${NUM_EXPERTS}"
echo "hard_ratio=${HARD_RATIO}, hard_weight=${HARD_WEIGHT}, hard_warmup=${HARD_WARMUP_EPOCHS}"
echo "lambda_entropy=${LAMBDA_ENTROPY}, grad_clip=${GRAD_CLIP_VAL}(${GRAD_CLIP_ALGO})"
echo "max_epochs=${MAX_EPOCHS}, patience=${PATIENCE}"
echo "=========================================="

python -u "${TRAIN_PY}" \
    --model mmoe \
    --batch_size 1024 \
    --num_workers 32 \
    --max_epochs "${MAX_EPOCHS}" \
    --learning_rate "${LEARNING_RATE}" \
    --dropout "${DROPOUT}" \
    --embedding_dim 32 \
    --expert_hidden_dims 256 128 \
    --num_experts "${NUM_EXPERTS}" \
    --tower_hidden_dims 64 \
    --accelerator gpu \
    --devices 4 \
    --esmm \
    --sigmoid 1 \
    --strategy ddp_find_unused_parameters_false \
    --early_stop_patience "${PATIENCE}" \
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

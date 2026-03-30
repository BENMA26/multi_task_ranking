#!/bin/bash
#SBATCH --job-name=mtl_asym_mmoe_on10_rn_ent
#SBATCH --output=/work/home/maben/project/rec_sys/projects/multi_task_ranking/experiments/asymmetric_projection/logs/mtl_asym_mmoe_on10_rn_ent_%j.out
#SBATCH --error=/work/home/maben/project/rec_sys/projects/multi_task_ranking/experiments/asymmetric_projection/logs/mtl_asym_mmoe_on10_rn_ent_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:4
#SBATCH --mem=96G
#SBATCH --time=144:00:00
#SBATCH --partition=normal
#SBATCH --exclude=g01n01

set -u

PROJECT_ROOT=/work/home/maben/project/rec_sys/projects/multi_task_ranking
LOG_DIR=${PROJECT_ROOT}/experiments/asymmetric_projection/logs

cd "${PROJECT_ROOT}"
mkdir -p "${LOG_DIR}"

TRAIN_PY=${PROJECT_ROOT}/train_rank.py
OUTPUT_ROOT=${PROJECT_ROOT}/experiments/asymmetric_projection/outputs_mmoe_on10_restore_norm_entropy
mkdir -p "${OUTPUT_ROOT}"

COMMON_ARGS="
    --model mmoe
    --batch_size 1024
    --num_workers 32
    --max_epochs 20
    --learning_rate 7e-4
    --dropout 0.25
    --embedding_dim 32
    --num_experts 4
    --expert_hidden_dims 256 128
    --tower_hidden_dims 64
    --accelerator gpu
    --devices 4
    --esmm
    --sigmoid 1
    --use_entropy_reg
    --lambda_entropy 0.01
    --strategy ddp_find_unused_parameters_false
    --early_stop_patience 3
    --monitor_grad_conflict
    --grad_conflict_interval 100
"

EXP_NAME="mmoe_esmm_asym_on10_rn_m5p0_l0p3_t0p0_ent0p01_e4"

echo "=========================================="
echo "Experiment: ${EXP_NAME}"
echo "=========================================="

python -u "${TRAIN_PY}" ${COMMON_ARGS} \
    --use_asym_proj \
    --asym_proj_only_10 \
    --asym_proj_restore_norm \
    --asym_proj_restore_max_scale 5.0 \
    --asym_proj_lambda 0.3 \
    --asym_proj_tau 0.0 \
    --exp_dir "${OUTPUT_ROOT}/${EXP_NAME}"

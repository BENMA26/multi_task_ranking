#!/bin/bash
#SBATCH --job-name=mtl_asym_proj_sb
#SBATCH --output=/work/home/maben/project/rec_sys/projects/multi_task_ranking/experiments/asymmetric_projection/logs/mtl_asym_sb_%j.out
#SBATCH --error=/work/home/maben/project/rec_sys/projects/multi_task_ranking/experiments/asymmetric_projection/logs/mtl_asym_sb_%j.err
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
OUTPUT_ROOT=${PROJECT_ROOT}/experiments/asymmetric_projection/outputs
mkdir -p "${OUTPUT_ROOT}"

COMMON_ARGS="
    --model share_bottom
    --batch_size 1024
    --num_workers 32
    --max_epochs 20
    --learning_rate 7e-4
    --dropout 0.25
    --embedding_dim 32
    --shared_hidden_dims 256 128
    --tower_hidden_dims 64
    --accelerator gpu
    --devices 4
    --esmm
    --sigmoid 1
    --strategy ddp_find_unused_parameters_false
    --early_stop_patience 3
    --monitor_grad_conflict
    --grad_conflict_interval 100
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
        echo "[ERROR] ${name} failed! Continuing..."
    fi
    echo ""
}

# Asymmetric projection: lambda sweep (tau=0)
for LAMBDA in 0.3 0.5 0.8 1.0; do
    LAMBDA_TAG=${LAMBDA/./p}
    run_exp "share_bottom_esmm_asym_l${LAMBDA_TAG}_t0p0" "
        --use_asym_proj
        --asym_proj_lambda ${LAMBDA}
        --asym_proj_tau 0.0
    "
done

# Thresholded variants
run_exp "share_bottom_esmm_asym_l0p5_t0p1" "
    --use_asym_proj
    --asym_proj_lambda 0.5
    --asym_proj_tau 0.1
"

run_exp "share_bottom_esmm_asym_l1p0_t0p1" "
    --use_asym_proj
    --asym_proj_lambda 1.0
    --asym_proj_tau 0.1
"

echo "All ShareBottom asymmetric projection experiments finished."

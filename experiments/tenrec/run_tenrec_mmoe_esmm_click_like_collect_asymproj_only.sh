#!/bin/bash
#SBATCH --job-name=tenrec_clk_like_asym
#SBATCH --output=/work/home/maben/project/rec_sys/projects/multi_task_ranking/experiments/tenrec/logs/tenrec_clk_like_asym_%j.out
#SBATCH --error=/work/home/maben/project/rec_sys/projects/multi_task_ranking/experiments/tenrec/logs/tenrec_clk_like_asym_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:4
#SBATCH --mem=96G
#SBATCH --time=144:00:00
#SBATCH --partition=normal
#SBATCH --exclude=g01n01

set -u

PROJECT_ROOT=/work/home/maben/project/rec_sys/projects/multi_task_ranking
LOG_DIR=${PROJECT_ROOT}/experiments/tenrec/logs
TRAIN_PY=${PROJECT_ROOT}/train_tenrec_rank.py
OUTPUT_ROOT=${PROJECT_ROOT}/experiments/tenrec/outputs_click_like_collect

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}"
cd "${PROJECT_ROOT}"

# 默认使用 ctr_data_1M.csv（包含 click/like/follow/share，click 有正负样本）
TENREC_INPUT_PATH=${TENREC_INPUT_PATH:-${PROJECT_ROOT}/dataset/tenrec/Tenrec/ctr_data_1M.csv}
if [ ! -f "${TENREC_INPUT_PATH}" ]; then
    echo "[ERROR] 数据文件不存在: ${TENREC_INPUT_PATH}"
    exit 1
fi

# 第三任务默认使用 follow，回退 share
TENREC_COLLECT_LABEL=${TENREC_COLLECT_LABEL:-follow}
TENREC_COLLECT_FALLBACK=${TENREC_COLLECT_FALLBACK:-share}

echo "[INFO] TENREC_INPUT_PATH=${TENREC_INPUT_PATH}"
echo "[INFO] label mapping: click=click like=like collect=${TENREC_COLLECT_LABEL} fallback=${TENREC_COLLECT_FALLBACK}"
echo "=========================================="
echo "Experiment: tenrec_mmoe_esmm_click_like_collect_asymproj"
echo "=========================================="

python -u "${TRAIN_PY}" \
    --input_path "${TENREC_INPUT_PATH}" \
    --click_label click \
    --like_label like \
    --collect_label "${TENREC_COLLECT_LABEL}" \
    --collect_label_fallback "${TENREC_COLLECT_FALLBACK}" \
    --filter_click0_with_other_behavior \
    --batch_size 1024 \
    --num_workers 32 \
    --max_epochs "${MAX_EPOCHS:-20}" \
    --learning_rate 3e-4 \
    --dropout 0.25 \
    --embedding_dim 32 \
    --num_experts 6 \
    --expert_hidden_dims 256 128 \
    --tower_hidden_dims 64 \
    --w_click 1.0 \
    --w_like 1.0 \
    --w_collect 1.0 \
    --accelerator gpu \
    --devices 4 \
    --strategy ddp_find_unused_parameters_false \
    --early_stop_patience "${EARLY_STOP_PATIENCE:-3}" \
    --use_asym_proj \
    --asym_proj_lambda 1.0 \
    --asym_proj_tau 0.0 \
    --asym_proj_restore_norm \
    --asym_proj_restore_max_scale 5.0 \
    --exp_dir "${OUTPUT_ROOT}/tenrec_mmoe_esmm_click_like_collect_asymproj"

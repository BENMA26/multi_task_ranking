#!/bin/bash
#SBATCH --job-name=mtl8_nodeA_p1
#SBATCH --output=/work/home/maben/project/rec_sys/projects/multi_task_ranking/experiments/eight_settings_dualnode/slurm/nodeA_%j.out
#SBATCH --error=/work/home/maben/project/rec_sys/projects/multi_task_ranking/experiments/eight_settings_dualnode/slurm/nodeA_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:4
#SBATCH --mem=96G
#SBATCH --time=144:00:00
#SBATCH --partition=normal

set -u -o pipefail

PROJECT_ROOT=/work/home/maben/project/rec_sys/projects/multi_task_ranking
TRAIN_PY=${PROJECT_ROOT}/train_rank.py
EXP_ROOT=${PROJECT_ROOT}/experiments/eight_settings_dualnode
RUN_ID=${RUN_ID:-nodeA_p1_$(date +%Y%m%d_%H%M%S)}

SLURM_LOG_DIR=${EXP_ROOT}/slurm
TASK_LOG_DIR=${EXP_ROOT}/logs/${RUN_ID}
OUTPUT_ROOT=${EXP_ROOT}/outputs/${RUN_ID}
SUMMARY_TSV=${EXP_ROOT}/summaries/${RUN_ID}.tsv

mkdir -p "${SLURM_LOG_DIR}" "${TASK_LOG_DIR}" "${OUTPUT_ROOT}" "${EXP_ROOT}/summaries"
cd "${PROJECT_ROOT}"

ALI_CCP_TRAIN_PATH=${ALI_CCP_TRAIN_PATH:-/work/home/maben/project/rec_sys/projects/Ali_CCP_REC/dataset/datasetsali_ccp_train.parquet}
ALI_CCP_VAL_PATH=${ALI_CCP_VAL_PATH:-/work/home/maben/project/rec_sys/projects/Ali_CCP_REC/dataset/datasetsali_ccp_val.parquet}
ALI_CCP_TEST_PATH=${ALI_CCP_TEST_PATH:-/work/home/maben/project/rec_sys/projects/Ali_CCP_REC/dataset/datasetsali_ccp_test.parquet}

SEEDS=${SEEDS:-42,2027}
BATCH_SIZE=${BATCH_SIZE:-1024}
NUM_WORKERS=${NUM_WORKERS:-32}
MAX_EPOCHS=${MAX_EPOCHS:-20}
EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-2}
EMBEDDING_DIM=${EMBEDDING_DIM:-32}
LEARNING_RATE=${LEARNING_RATE:-7e-4}
DROPOUT=${DROPOUT:-0.25}
LAMBDA_ENTROPY=${LAMBDA_ENTROPY:-0.0005}

TRAIN_SUBSET_FRAC=${TRAIN_SUBSET_FRAC:-1.0}
TRAIN_SUBSET_SEED=${TRAIN_SUBSET_SEED:-42}
TRAIN_SUBSET_STRATIFY=${TRAIN_SUBSET_STRATIFY:-1}

if [ ! -f "${ALI_CCP_TRAIN_PATH}" ] || [ ! -f "${ALI_CCP_VAL_PATH}" ] || [ ! -f "${ALI_CCP_TEST_PATH}" ]; then
    echo "[ERROR] Ali-CCP parquet files not found."
    echo "  train=${ALI_CCP_TRAIN_PATH}"
    echo "  val=${ALI_CCP_VAL_PATH}"
    echo "  test=${ALI_CCP_TEST_PATH}"
    exit 1
fi

IFS=',' read -r -a SEED_LIST <<< "${SEEDS}"
if [ "${#SEED_LIST[@]}" -eq 0 ]; then
    SEED_LIST=(42)
fi

COMMON_ARGS=(
    --train_path "${ALI_CCP_TRAIN_PATH}"
    --val_path "${ALI_CCP_VAL_PATH}"
    --test_path "${ALI_CCP_TEST_PATH}"
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --max_epochs "${MAX_EPOCHS}"
    --early_stop_patience "${EARLY_STOP_PATIENCE}"
    --accelerator gpu
    --devices 4
    --strategy ddp_find_unused_parameters_false
    --esmm
    --sigmoid 1
    --train_subset_frac "${TRAIN_SUBSET_FRAC}"
    --train_subset_seed "${TRAIN_SUBSET_SEED}"
    --embedding_dim "${EMBEDDING_DIM}"
    --tower_hidden_dims 64
)
if [ "${TRAIN_SUBSET_STRATIFY}" = "1" ]; then
    COMMON_ARGS+=(--train_subset_stratify)
fi

echo -e "setting\tseed\tstatus\tlog_file" > "${SUMMARY_TSV}"

ok_count=0
fail_count=0

run_case() {
    local setting="$1"
    shift
    local extra_args=("$@")

    for raw_seed in "${SEED_LIST[@]}"; do
        local seed
        seed=$(echo "${raw_seed}" | xargs)
        local exp_name="${setting}_s${seed}"
        local log_file="${TASK_LOG_DIR}/${exp_name}.log"

        echo "=========================================="
        echo "[NODE-A] setting=${setting} seed=${seed}"
        echo "=========================================="

        if python -u "${TRAIN_PY}" train \
            "${COMMON_ARGS[@]}" \
            "${extra_args[@]}" \
            --seed "${seed}" \
            --exp_dir "${OUTPUT_ROOT}/${exp_name}" \
            > "${log_file}" 2>&1; then
            echo "[OK] ${exp_name}"
            echo -e "${setting}\t${seed}\tOK\t${log_file}" >> "${SUMMARY_TSV}"
            ok_count=$((ok_count + 1))
        else
            echo "[ERROR] ${exp_name}"
            echo -e "${setting}\t${seed}\tFAIL\t${log_file}" >> "${SUMMARY_TSV}"
            fail_count=$((fail_count + 1))
        fi
        echo
    done
}

# 1) share bottom
run_case "01_share_bottom" \
    --model share_bottom \
    --shared_hidden_dims 256 128 \
    --regularization none \
    --grad_surgery normal \
    --grad_scope all \
    --learning_rate "${LEARNING_RATE}" \
    --dropout "${DROPOUT}"

# 2) mmoe
run_case "02_mmoe" \
    --model mmoe \
    --num_experts 6 \
    --expert_hidden_dims 256 128 \
    --regularization none \
    --grad_surgery normal \
    --grad_scope all \
    --learning_rate "${LEARNING_RATE}" \
    --dropout "${DROPOUT}"

# 3) mmoe + entropy reg
run_case "03_mmoe_entropy" \
    --model mmoe \
    --num_experts 6 \
    --expert_hidden_dims 256 128 \
    --regularization entropy \
    --lambda_entropy "${LAMBDA_ENTROPY}" \
    --grad_surgery normal \
    --grad_scope all \
    --learning_rate "${LEARNING_RATE}" \
    --dropout "${DROPOUT}"

echo "[DONE] nodeA serial run finished. ok=${ok_count} fail=${fail_count}"
echo "[DONE] summary=${SUMMARY_TSV}"

if [ "${fail_count}" -gt 0 ]; then
    exit 1
fi
exit 0

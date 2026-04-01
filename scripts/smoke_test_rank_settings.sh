#!/usr/bin/env bash

# Smoke test for 8 ranking settings:
# 1) share_bottom
# 2) mmoe
# 3) mmoe + entropy
# 4) mmoe + entropy + upgrad (all)
# 5) mmoe + entropy + upgrad (on10)
# 6) mmoe + entropy + asymmetric_projection (all)
# 7) mmoe + entropy + asymmetric_projection (on10)
# 8) mmoe + entropy + asymmetric_projection (on10 + restore_norm)

set -u -o pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/home/maben/project/rec_sys/projects/multi_task_ranking}
TRAIN_PY=${TRAIN_PY:-${PROJECT_ROOT}/train_rank.py}

RUN_ID=${RUN_ID:-smoke_rank_settings_$(date +%Y%m%d_%H%M%S)}
OUT_ROOT=${OUT_ROOT:-${PROJECT_ROOT}/smoke_test_outputs/${RUN_ID}}
LOG_DIR=${OUT_ROOT}/logs
SUMMARY_TSV=${OUT_ROOT}/summary.tsv

ALI_CCP_TRAIN_PATH=${ALI_CCP_TRAIN_PATH:-/work/home/maben/project/rec_sys/projects/Ali_CCP_REC/dataset/datasetsali_ccp_train.parquet}
ALI_CCP_VAL_PATH=${ALI_CCP_VAL_PATH:-/work/home/maben/project/rec_sys/projects/Ali_CCP_REC/dataset/datasetsali_ccp_val.parquet}
ALI_CCP_TEST_PATH=${ALI_CCP_TEST_PATH:-/work/home/maben/project/rec_sys/projects/Ali_CCP_REC/dataset/datasetsali_ccp_test.parquet}

# Smoke-test runtime controls
BATCH_SIZE=${BATCH_SIZE:-2048}
NUM_WORKERS=${NUM_WORKERS:-4}
MAX_EPOCHS=${MAX_EPOCHS:-1}
EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-1}
TRAIN_SUBSET_FRAC=${TRAIN_SUBSET_FRAC:-0.001}
TRAIN_SUBSET_SEED=${TRAIN_SUBSET_SEED:-42}
SEED=${SEED:-42}
ACCELERATOR=${ACCELERATOR:-auto}
DEVICES=${DEVICES:-1}
STRATEGY=${STRATEGY:-single_device}

# Method controls
LAMBDA_ENTROPY=${LAMBDA_ENTROPY:-0.01}
ASYM_LAMBDA=${ASYM_LAMBDA:-1.0}
ASYM_TAU=${ASYM_TAU:-0.0}
ASYM_RESTORE_MAX_SCALE=${ASYM_RESTORE_MAX_SCALE:-5.0}

if [ ! -f "${TRAIN_PY}" ]; then
  echo "[ERROR] train script not found: ${TRAIN_PY}"
  exit 1
fi
if [ ! -f "${ALI_CCP_TRAIN_PATH}" ] || [ ! -f "${ALI_CCP_VAL_PATH}" ] || [ ! -f "${ALI_CCP_TEST_PATH}" ]; then
  echo "[ERROR] Ali-CCP dataset files not found."
  echo "  train=${ALI_CCP_TRAIN_PATH}"
  echo "  val=${ALI_CCP_VAL_PATH}"
  echo "  test=${ALI_CCP_TEST_PATH}"
  exit 1
fi

mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"

echo -e "setting\tstatus\tduration_sec\tlog_file" > "${SUMMARY_TSV}"

COMMON_ARGS=(
  --train_path "${ALI_CCP_TRAIN_PATH}"
  --val_path "${ALI_CCP_VAL_PATH}"
  --test_path "${ALI_CCP_TEST_PATH}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --max_epochs "${MAX_EPOCHS}"
  --early_stop_patience "${EARLY_STOP_PATIENCE}"
  --accelerator "${ACCELERATOR}"
  --devices "${DEVICES}"
  --strategy "${STRATEGY}"
  --esmm
  --sigmoid 1
  --regularization none
  --grad_surgery normal
  --grad_scope all
  --train_subset_frac "${TRAIN_SUBSET_FRAC}"
  --train_subset_seed "${TRAIN_SUBSET_SEED}"
  --seed "${SEED}"
)

ok_count=0
fail_count=0

run_case() {
  local name="$1"
  shift
  local log_file="${LOG_DIR}/${name}.log"
  local start_ts end_ts duration status

  echo "=========================================="
  echo "[SMOKE] ${name}"
  echo "=========================================="

  start_ts=$(date +%s)
  if python -u "${TRAIN_PY}" train "${COMMON_ARGS[@]}" "$@" --exp_dir "${OUT_ROOT}/${name}" > "${log_file}" 2>&1; then
    status="OK"
    ok_count=$((ok_count + 1))
  else
    status="FAIL"
    fail_count=$((fail_count + 1))
    echo "[WARN] ${name} failed. tail log:"
    tail -n 40 "${log_file}" || true
  fi
  end_ts=$(date +%s)
  duration=$((end_ts - start_ts))

  echo -e "${name}\t${status}\t${duration}\t${log_file}" >> "${SUMMARY_TSV}"
  echo "[${status}] ${name} (${duration}s)"
  echo
}

# 1) share_bottom
run_case "01_share_bottom" \
  --model share_bottom \
  --shared_hidden_dims 256 128 \
  --learning_rate 7e-4 \
  --dropout 0.25

# 2) mmoe
run_case "02_mmoe" \
  --model mmoe \
  --num_experts 6 \
  --expert_hidden_dims 256 128 \
  --learning_rate 7e-4 \
  --dropout 0.25

# 3) mmoe + entropy
run_case "03_mmoe_entropy" \
  --model mmoe \
  --num_experts 6 \
  --expert_hidden_dims 256 128 \
  --learning_rate 7e-4 \
  --dropout 0.25 \
  --regularization entropy \
  --lambda_entropy "${LAMBDA_ENTROPY}"

# 4) mmoe + entropy + upgrad (all)
run_case "04_mmoe_entropy_upgrad_all" \
  --model mmoe \
  --num_experts 6 \
  --expert_hidden_dims 256 128 \
  --learning_rate 7e-4 \
  --dropout 0.25 \
  --regularization entropy \
  --lambda_entropy "${LAMBDA_ENTROPY}" \
  --grad_surgery upgrad \
  --grad_scope all

# 5) mmoe + entropy + upgrad (on10)
run_case "05_mmoe_entropy_upgrad_on10" \
  --model mmoe \
  --num_experts 6 \
  --expert_hidden_dims 256 128 \
  --learning_rate 7e-4 \
  --dropout 0.25 \
  --regularization entropy \
  --lambda_entropy "${LAMBDA_ENTROPY}" \
  --grad_surgery upgrad \
  --grad_scope on10

# 6) mmoe + entropy + asymmetric_projection (all)
run_case "06_mmoe_entropy_asym_all" \
  --model mmoe \
  --num_experts 6 \
  --expert_hidden_dims 256 128 \
  --learning_rate 7e-4 \
  --dropout 0.25 \
  --regularization entropy \
  --lambda_entropy "${LAMBDA_ENTROPY}" \
  --grad_surgery asymmetric_projection \
  --grad_scope all \
  --asym_proj_lambda "${ASYM_LAMBDA}" \
  --asym_proj_tau "${ASYM_TAU}"

# 7) mmoe + entropy + asymmetric_projection (on10)
run_case "07_mmoe_entropy_asym_on10" \
  --model mmoe \
  --num_experts 6 \
  --expert_hidden_dims 256 128 \
  --learning_rate 7e-4 \
  --dropout 0.25 \
  --regularization entropy \
  --lambda_entropy "${LAMBDA_ENTROPY}" \
  --grad_surgery asymmetric_projection \
  --grad_scope on10 \
  --asym_proj_lambda "${ASYM_LAMBDA}" \
  --asym_proj_tau "${ASYM_TAU}"

# 8) mmoe + entropy + asymmetric_projection (on10 + restore_norm)
run_case "08_mmoe_entropy_asym_on10_restore_norm" \
  --model mmoe \
  --num_experts 6 \
  --expert_hidden_dims 256 128 \
  --learning_rate 7e-4 \
  --dropout 0.25 \
  --regularization entropy \
  --lambda_entropy "${LAMBDA_ENTROPY}" \
  --grad_surgery asymmetric_projection \
  --grad_scope on10 \
  --asym_proj_lambda "${ASYM_LAMBDA}" \
  --asym_proj_tau "${ASYM_TAU}" \
  --asym_proj_restore_norm \
  --asym_proj_restore_max_scale "${ASYM_RESTORE_MAX_SCALE}"

echo "=========================================="
echo "[DONE] smoke test finished"
echo "run_id=${RUN_ID}"
echo "ok=${ok_count} fail=${fail_count}"
echo "summary=${SUMMARY_TSV}"
echo "=========================================="

if [ "${fail_count}" -gt 0 ]; then
  exit 1
fi

exit 0

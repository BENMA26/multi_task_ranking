#!/bin/bash
# Run project ablations using best settings from HP search:
# models: share_bottom / mmoe / ple
# variants: baseline, entropy, jd, asym, entropy+jd, entropy+asym

set -u

PROJECT_ROOT=/work/home/maben/project/rec_sys/projects/multi_task_ranking
TRAIN_PY=${PROJECT_ROOT}/train_rank.py
LOG_DIR=${PROJECT_ROOT}/experiments/final_ali_ccp/logs/project_compare/ablation
OUTPUT_ROOT=${PROJECT_ROOT}/experiments/final_ali_ccp/outputs/project_compare/ablation
BEST_CSV=${BEST_CSV:-${PROJECT_ROOT}/experiments/final_ali_ccp/outputs/project_compare/hp/hp_best_by_stage_model.csv}
SUMMARY_CSV=${OUTPUT_ROOT}/ablation_summary.csv
GROUPED_CSV=${OUTPUT_ROOT}/ablation_grouped.csv
BEST_MODEL_CSV=${OUTPUT_ROOT}/ablation_best_by_model.csv

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}"
cd "${PROJECT_ROOT}"

RUN_TS=${RUN_TS:-$(date +%Y%m%d_%H%M%S)}
SCRIPT_LOG=${LOG_DIR}/project_ablation_${RUN_TS}.out
exec > >(tee -a "${SCRIPT_LOG}") 2>&1

if [ ! -f "${BEST_CSV}" ]; then
    echo "[ERROR] best hp csv not found: ${BEST_CSV}"
    exit 1
fi

ALI_CCP_TRAIN_PATH=${ALI_CCP_TRAIN_PATH:-/work/home/maben/project/rec_sys/projects/Ali_CCP_REC/dataset/datasetsali_ccp_train.parquet}
ALI_CCP_VAL_PATH=${ALI_CCP_VAL_PATH:-/work/home/maben/project/rec_sys/projects/Ali_CCP_REC/dataset/datasetsali_ccp_val.parquet}
ALI_CCP_TEST_PATH=${ALI_CCP_TEST_PATH:-/work/home/maben/project/rec_sys/projects/Ali_CCP_REC/dataset/datasetsali_ccp_test.parquet}

if [ ! -f "${ALI_CCP_TRAIN_PATH}" ] || [ ! -f "${ALI_CCP_VAL_PATH}" ] || [ ! -f "${ALI_CCP_TEST_PATH}" ]; then
    echo "[ERROR] Ali-CCP parquet files not found."
    echo "  train=${ALI_CCP_TRAIN_PATH}"
    echo "  val=${ALI_CCP_VAL_PATH}"
    echo "  test=${ALI_CCP_TEST_PATH}"
    exit 1
fi

SEEDS=${SEEDS:-42,2027,3407}
MAX_EXPERIMENTS=${MAX_EXPERIMENTS:-0}

TRAIN_SUBSET_FRAC=${TRAIN_SUBSET_FRAC:-0.1}
TRAIN_SUBSET_SEED=${TRAIN_SUBSET_SEED:-42}
TRAIN_SUBSET_STRATIFY=${TRAIN_SUBSET_STRATIFY:-1}
BATCH_SIZE=${BATCH_SIZE:-1024}
NUM_WORKERS=${NUM_WORKERS:-16}
MAX_EPOCHS=${MAX_EPOCHS:-6}
EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-2}
ACCELERATOR=${ACCELERATOR:-auto}
DEVICES=${DEVICES:-1}
STRATEGY=${STRATEGY:-auto}

IFS=',' read -r -a SEED_LIST <<< "${SEEDS}"
if [ "${#SEED_LIST[@]}" -eq 0 ]; then
    SEED_LIST=(42)
fi

if [ "${STRATEGY}" = "auto" ]; then
    STRATEGY=ddp_find_unused_parameters_false
fi

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
    --train_subset_frac "${TRAIN_SUBSET_FRAC}"
    --train_subset_seed "${TRAIN_SUBSET_SEED}"
)

if [ "${TRAIN_SUBSET_STRATIFY}" = "1" ]; then
    COMMON_ARGS+=(--train_subset_stratify)
fi

RUN_COUNT=0
can_continue() {
    if [ "${MAX_EXPERIMENTS}" -le 0 ]; then
        return 0
    fi
    [ "${RUN_COUNT}" -lt "${MAX_EXPERIMENTS}" ]
}

run_exp() {
    local base_name=$1
    shift
    local extra_args=("$@")

    for raw_seed in "${SEED_LIST[@]}"; do
        can_continue || return 0
        local seed
        seed=$(echo "${raw_seed}" | xargs)
        local name="${base_name}_s${seed}"

        RUN_COUNT=$((RUN_COUNT + 1))
        echo "=========================================="
        echo "Experiment: ${name}"
        echo "Run [${RUN_COUNT}] | seed=${seed}"
        echo "=========================================="

        if python -u "${TRAIN_PY}" "${COMMON_ARGS[@]}" "${extra_args[@]}" --seed "${seed}" --exp_dir "${OUTPUT_ROOT}/${name}"; then
            echo "[OK] ${name} completed."
        else
            echo "[ERROR] ${name} failed."
        fi
        echo ""
    done
}

best_setting() {
    local stage=$1
    local model=$2
    python - "${BEST_CSV}" "${stage}" "${model}" <<'PY'
import csv
import sys

path, stage, model = sys.argv[1], sys.argv[2], sys.argv[3]
setting = ""
with open(path, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row.get("stage") == stage and row.get("model") == model:
            setting = row.get("setting", "")
            break
print(setting)
PY
}

tag_to_float() {
    local tag=$1
    echo "${tag//p/.}"
}

normal_args() {
    local setting=$1
    if [[ "${setting}" == projhp_normal_share_bottom_* ]]; then
        if [[ "${setting}" =~ _sh([0-9]+)x([0-9]+)$ ]]; then
            echo "--model share_bottom --shared_hidden_dims ${BASH_REMATCH[1]} ${BASH_REMATCH[2]} --learning_rate 7e-4 --dropout 0.25"
            return 0
        fi
    elif [[ "${setting}" == projhp_normal_mmoe_* ]]; then
        if [[ "${setting}" =~ _e([0-9]+)$ ]]; then
            echo "--model mmoe --num_experts ${BASH_REMATCH[1]} --expert_hidden_dims 256 128 --learning_rate 7e-4 --dropout 0.25"
            return 0
        fi
    elif [[ "${setting}" == projhp_normal_ple_* ]]; then
        if [[ "${setting}" =~ _ns([0-9]+)_nsh([0-9]+)_l([0-9]+)$ ]]; then
            echo "--model ple --num_specific_experts ${BASH_REMATCH[1]} --num_shared_experts ${BASH_REMATCH[2]} --num_levels ${BASH_REMATCH[3]} --expert_hidden_dims 256 128 --learning_rate 7e-4 --dropout 0.25"
            return 0
        fi
    fi
    return 1
}

entropy_args() {
    local setting=$1
    if [[ "${setting}" =~ _l([0-9p]+)$ ]]; then
        local lam
        lam=$(tag_to_float "${BASH_REMATCH[1]}")
        echo "--use_entropy_reg --lambda_entropy ${lam}"
        return 0
    fi
    return 1
}

jd_args() {
    local setting=$1
    local method=${setting##*_}
    if [ -n "${method}" ]; then
        echo "--use_torchjd --aggregation_method ${method}"
        return 0
    fi
    return 1
}

asym_args() {
    local setting=$1
    local extra=""
    if [[ "${setting}" =~ _l([0-9p]+)_t([0-9p]+)$ ]]; then
        local lam tau
        lam=$(tag_to_float "${BASH_REMATCH[1]}")
        tau=$(tag_to_float "${BASH_REMATCH[2]}")
        extra="--use_asym_proj --asym_proj_lambda ${lam} --asym_proj_tau ${tau}"
    else
        return 1
    fi
    if [[ "${setting}" == *"_on10_rn_"* ]]; then
        extra="${extra} --asym_proj_only_10 --asym_proj_restore_norm --asym_proj_restore_max_scale 5.0"
    fi
    echo "${extra}"
    return 0
}

echo "[INFO] project ablation config"
echo "  best_csv=${BEST_CSV}"
echo "  seeds=${SEEDS}"
echo "  subset_frac=${TRAIN_SUBSET_FRAC}"
echo "  epochs=${MAX_EPOCHS}"
echo "  accelerator=${ACCELERATOR} devices=${DEVICES} strategy=${STRATEGY}"
echo "  max_experiments=${MAX_EXPERIMENTS}"
echo "  output_root=${OUTPUT_ROOT}"
echo "  script_log=${SCRIPT_LOG}"

for MODEL in share_bottom mmoe ple; do
    can_continue || break

    NORMAL_SETTING=$(best_setting normal "${MODEL}")
    if [ -z "${NORMAL_SETTING}" ]; then
        echo "[WARN] missing normal best setting for model=${MODEL}, skip."
        continue
    fi

    BASE_ARGS_STR=$(normal_args "${NORMAL_SETTING}") || {
        echo "[WARN] failed to parse normal setting for model=${MODEL}: ${NORMAL_SETTING}"
        continue
    }

    JD_SETTING=$(best_setting jd "${MODEL}")
    ENT_SETTING=$(best_setting entropy "${MODEL}")
    ASYM_SETTING=$(best_setting asym "${MODEL}")

    read -r -a BASE_ARGS <<< "${BASE_ARGS_STR}"

    run_exp "projab_${MODEL}_baseline" "${BASE_ARGS[@]}"

    if [ -n "${JD_SETTING}" ]; then
        JD_ARGS_STR=$(jd_args "${JD_SETTING}") || JD_ARGS_STR=""
        if [ -n "${JD_ARGS_STR}" ]; then
            read -r -a JD_ARGS <<< "${JD_ARGS_STR}"
            run_exp "projab_${MODEL}_jd" "${BASE_ARGS[@]}" "${JD_ARGS[@]}"
        fi
    fi

    ENT_ARGS_STR=""
    if [ -n "${ENT_SETTING}" ]; then
        ENT_ARGS_STR=$(entropy_args "${ENT_SETTING}") || ENT_ARGS_STR=""
        if [ -n "${ENT_ARGS_STR}" ]; then
            read -r -a ENT_ARGS <<< "${ENT_ARGS_STR}"
            run_exp "projab_${MODEL}_entropy" "${BASE_ARGS[@]}" "${ENT_ARGS[@]}"
        fi
    fi

    ASYM_ARGS_STR=""
    if [ -n "${ASYM_SETTING}" ]; then
        ASYM_ARGS_STR=$(asym_args "${ASYM_SETTING}") || ASYM_ARGS_STR=""
        if [ -n "${ASYM_ARGS_STR}" ]; then
            read -r -a ASYM_ARGS <<< "${ASYM_ARGS_STR}"
            run_exp "projab_${MODEL}_asym" "${BASE_ARGS[@]}" "${ASYM_ARGS[@]}"
        fi
    fi

    if [ -n "${ENT_ARGS_STR}" ] && [ -n "${JD_ARGS_STR:-}" ]; then
        read -r -a ENT_ARGS <<< "${ENT_ARGS_STR}"
        read -r -a JD_ARGS <<< "${JD_ARGS_STR}"
        run_exp "projab_${MODEL}_entropy_jd" "${BASE_ARGS[@]}" "${ENT_ARGS[@]}" "${JD_ARGS[@]}"
    fi

    if [ -n "${ENT_ARGS_STR}" ] && [ -n "${ASYM_ARGS_STR}" ] && [ "${MODEL}" != "ple" ]; then
        read -r -a ENT_ARGS <<< "${ENT_ARGS_STR}"
        read -r -a ASYM_ARGS <<< "${ASYM_ARGS_STR}"
        run_exp "projab_${MODEL}_entropy_asym" "${BASE_ARGS[@]}" "${ENT_ARGS[@]}" "${ASYM_ARGS[@]}"
    fi
done

echo "[INFO] ablation finished. total_runs=${RUN_COUNT}"
echo "[INFO] summarizing ablation logs"

python experiments/final_ali_ccp/summarize_ali_ccp_results.py \
    --log_dir "${LOG_DIR}" \
    --out_csv "${SUMMARY_CSV}"

python experiments/final_ali_ccp/aggregate_project_ablation_results.py \
    --input_csv "${SUMMARY_CSV}" \
    --out_csv "${GROUPED_CSV}" \
    --out_best_csv "${BEST_MODEL_CSV}"

echo "[INFO] ablation summary ready:"
echo "  summary_csv=${SUMMARY_CSV}"
echo "  grouped_csv=${GROUPED_CSV}"
echo "  best_model_csv=${BEST_MODEL_CSV}"

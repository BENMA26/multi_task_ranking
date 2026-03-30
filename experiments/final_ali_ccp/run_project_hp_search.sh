#!/bin/bash
# Project-specific hyperparameter search for:
#   share_bottom / mmoe / ple + ESMM
#   entropy regularization / TorchJD / asymmetric projection

set -u

PROJECT_ROOT=/work/home/maben/project/rec_sys/projects/multi_task_ranking
TRAIN_PY=${PROJECT_ROOT}/train_rank.py
LOG_DIR=${PROJECT_ROOT}/experiments/final_ali_ccp/logs/project_compare/hp
OUTPUT_ROOT=${PROJECT_ROOT}/experiments/final_ali_ccp/outputs/project_compare/hp
SUMMARY_CSV=${OUTPUT_ROOT}/hp_search_summary.csv
GROUPED_CSV=${OUTPUT_ROOT}/hp_search_grouped.csv
BEST_CSV=${OUTPUT_ROOT}/hp_best_by_stage_model.csv

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}"
cd "${PROJECT_ROOT}"

RUN_TS=${RUN_TS:-$(date +%Y%m%d_%H%M%S)}
SCRIPT_LOG=${LOG_DIR}/project_hp_search_${RUN_TS}.out
exec > >(tee -a "${SCRIPT_LOG}") 2>&1

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

STAGES=${STAGES:-normal,entropy,jd,asym}
SEARCH_PROFILE=${SEARCH_PROFILE:-quick}
SEEDS=${SEEDS:-42,2027,3407}
MAX_EXPERIMENTS=${MAX_EXPERIMENTS:-0}

TRAIN_SUBSET_FRAC=${TRAIN_SUBSET_FRAC:-0.1}
TRAIN_SUBSET_SEED=${TRAIN_SUBSET_SEED:-42}
TRAIN_SUBSET_STRATIFY=${TRAIN_SUBSET_STRATIFY:-1}
BATCH_SIZE=${BATCH_SIZE:-1024}
NUM_WORKERS=${NUM_WORKERS:-16}
MAX_EPOCHS=${MAX_EPOCHS:-4}
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

contains_stage() {
    local target=$1
    [[ ",${STAGES}," == *",${target},"* ]]
}

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

echo "[INFO] project hp search config"
echo "  stages=${STAGES}"
echo "  profile=${SEARCH_PROFILE}"
echo "  seeds=${SEEDS}"
echo "  subset_frac=${TRAIN_SUBSET_FRAC}"
echo "  epochs=${MAX_EPOCHS}"
echo "  accelerator=${ACCELERATOR} devices=${DEVICES} strategy=${STRATEGY}"
echo "  max_experiments=${MAX_EXPERIMENTS}"
echo "  log_dir=${LOG_DIR}"
echo "  output_root=${OUTPUT_ROOT}"
echo "  script_log=${SCRIPT_LOG}"

# ---------------------------
# Stage 1: Normal (architecture)
# ---------------------------
if contains_stage normal; then
    echo "[INFO] stage=normal"
    if [ "${SEARCH_PROFILE}" = "mini" ]; then
        SB_HIDDENS=("256 128")
        MMOE_EXPERTS=(6)
        PLE_NS=(1)
    elif [ "${SEARCH_PROFILE}" = "full" ]; then
        SB_HIDDENS=("256 128" "384 192" "512 256")
        MMOE_EXPERTS=(4 6 8)
        PLE_NS=(1 2 3)
    else
        SB_HIDDENS=("256 128" "384 192")
        MMOE_EXPERTS=(4 6)
        PLE_NS=(1 2)
    fi

    for SH in "${SB_HIDDENS[@]}"; do
        can_continue || break
        tag=$(echo "${SH}" | tr ' ' 'x')
        run_exp "projhp_normal_share_bottom_sh${tag}" \
            --model share_bottom \
            --shared_hidden_dims ${SH} \
            --learning_rate 7e-4 \
            --dropout 0.25
    done

    for E in "${MMOE_EXPERTS[@]}"; do
        can_continue || break
        run_exp "projhp_normal_mmoe_e${E}" \
            --model mmoe \
            --num_experts ${E} \
            --expert_hidden_dims 256 128 \
            --learning_rate 7e-4 \
            --dropout 0.25
    done

    for NS in "${PLE_NS[@]}"; do
        can_continue || break
        run_exp "projhp_normal_ple_ns${NS}_nsh2_l2" \
            --model ple \
            --num_specific_experts ${NS} \
            --num_shared_experts 2 \
            --num_levels 2 \
            --expert_hidden_dims 256 128 \
            --learning_rate 7e-4 \
            --dropout 0.25
    done
fi

# ---------------------------
# Stage 2: Entropy regularization
# ---------------------------
if contains_stage entropy && can_continue; then
    echo "[INFO] stage=entropy"
    if [ "${SEARCH_PROFILE}" = "mini" ]; then
        ENTROPY_LAMS=(0.01)
    elif [ "${SEARCH_PROFILE}" = "full" ]; then
        ENTROPY_LAMS=(0.002 0.005 0.01 0.02)
    else
        ENTROPY_LAMS=(0.005 0.01 0.02)
    fi

    for MODEL in mmoe ple; do
        for LAM in "${ENTROPY_LAMS[@]}"; do
            can_continue || break 2
            ltag=${LAM/./p}
            if [ "${MODEL}" = "mmoe" ]; then
                run_exp "projhp_entropy_mmoe_l${ltag}" \
                    --model mmoe \
                    --num_experts 6 \
                    --expert_hidden_dims 256 128 \
                    --learning_rate 7e-4 \
                    --dropout 0.25 \
                    --use_entropy_reg \
                    --lambda_entropy ${LAM}
            else
                run_exp "projhp_entropy_ple_l${ltag}" \
                    --model ple \
                    --num_specific_experts 1 \
                    --num_shared_experts 2 \
                    --num_levels 2 \
                    --expert_hidden_dims 256 128 \
                    --learning_rate 7e-4 \
                    --dropout 0.25 \
                    --use_entropy_reg \
                    --lambda_entropy ${LAM}
            fi
        done
    done
fi

# ---------------------------
# Stage 3: TorchJD
# ---------------------------
if contains_stage jd && can_continue; then
    echo "[INFO] stage=jd"
    if [ "${SEARCH_PROFILE}" = "mini" ]; then
        JD_METHODS=(upgrad)
    elif [ "${SEARCH_PROFILE}" = "full" ]; then
        JD_METHODS=(upgrad mgda pcgrad graddrop)
    else
        JD_METHODS=(upgrad mgda pcgrad)
    fi

    for METHOD in "${JD_METHODS[@]}"; do
        can_continue || break
        run_exp "projhp_jd_share_bottom_${METHOD}" \
            --model share_bottom \
            --shared_hidden_dims 256 128 \
            --learning_rate 7e-4 \
            --dropout 0.25 \
            --use_torchjd \
            --aggregation_method ${METHOD}
    done

    for METHOD in "${JD_METHODS[@]}"; do
        can_continue || break
        run_exp "projhp_jd_mmoe_${METHOD}" \
            --model mmoe \
            --num_experts 6 \
            --expert_hidden_dims 256 128 \
            --learning_rate 7e-4 \
            --dropout 0.25 \
            --use_torchjd \
            --aggregation_method ${METHOD}
    done

    for METHOD in "${JD_METHODS[@]}"; do
        can_continue || break
        run_exp "projhp_jd_ple_${METHOD}" \
            --model ple \
            --num_specific_experts 1 \
            --num_shared_experts 2 \
            --num_levels 2 \
            --expert_hidden_dims 256 128 \
            --learning_rate 7e-4 \
            --dropout 0.25 \
            --use_torchjd \
            --aggregation_method ${METHOD}
    done
fi

# ---------------------------
# Stage 4: Asymmetric Projection
# ---------------------------
if contains_stage asym && can_continue; then
    echo "[INFO] stage=asym"
    if [ "${SEARCH_PROFILE}" = "mini" ]; then
        ASYM_VARIANTS=(base)
    else
        ASYM_VARIANTS=(base on10_rn)
    fi

    for V in "${ASYM_VARIANTS[@]}"; do
        can_continue || break
        if [ "${V}" = "base" ]; then
            run_exp "projhp_asym_share_bottom_base_l1_t0" \
                --model share_bottom \
                --shared_hidden_dims 256 128 \
                --learning_rate 7e-4 \
                --dropout 0.25 \
                --use_asym_proj \
                --asym_proj_lambda 1.0 \
                --asym_proj_tau 0.0
        else
            run_exp "projhp_asym_share_bottom_on10_rn_l1_t0" \
                --model share_bottom \
                --shared_hidden_dims 256 128 \
                --learning_rate 7e-4 \
                --dropout 0.25 \
                --use_asym_proj \
                --asym_proj_lambda 1.0 \
                --asym_proj_tau 0.0 \
                --asym_proj_only_10 \
                --asym_proj_restore_norm \
                --asym_proj_restore_max_scale 5.0
        fi
    done

    for V in "${ASYM_VARIANTS[@]}"; do
        can_continue || break
        if [ "${V}" = "base" ]; then
            run_exp "projhp_asym_mmoe_base_l1_t0" \
                --model mmoe \
                --num_experts 6 \
                --expert_hidden_dims 256 128 \
                --learning_rate 7e-4 \
                --dropout 0.25 \
                --use_asym_proj \
                --asym_proj_lambda 1.0 \
                --asym_proj_tau 0.0
        else
            run_exp "projhp_asym_mmoe_on10_rn_l1_t0" \
                --model mmoe \
                --num_experts 6 \
                --expert_hidden_dims 256 128 \
                --learning_rate 7e-4 \
                --dropout 0.25 \
                --use_asym_proj \
                --asym_proj_lambda 1.0 \
                --asym_proj_tau 0.0 \
                --asym_proj_only_10 \
                --asym_proj_restore_norm \
                --asym_proj_restore_max_scale 5.0
        fi
    done
fi

echo "[INFO] hp search finished. total_runs=${RUN_COUNT}"
echo "[INFO] now summarizing hp search logs"

python experiments/final_ali_ccp/summarize_ali_ccp_results.py \
    --log_dir "${LOG_DIR}" \
    --out_csv "${SUMMARY_CSV}"

python experiments/final_ali_ccp/aggregate_project_hp_results.py \
    --input_csv "${SUMMARY_CSV}" \
    --out_csv "${GROUPED_CSV}" \
    --out_best_csv "${BEST_CSV}"

echo "[INFO] hp summary ready:"
echo "  summary_csv=${SUMMARY_CSV}"
echo "  grouped_csv=${GROUPED_CSV}"
echo "  best_csv=${BEST_CSV}"

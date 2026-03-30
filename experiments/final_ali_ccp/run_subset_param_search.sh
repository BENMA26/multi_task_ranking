#!/bin/bash
# Subset-based parameter search for Ali-CCP:
#   normal / entropy / asym / jd
# Includes architecture search across share_bottom / moe / mmoe / ple.

set -u

PROJECT_ROOT=/work/home/maben/project/rec_sys/projects/multi_task_ranking
TRAIN_PY=${PROJECT_ROOT}/train_rank.py
LOG_DIR=${PROJECT_ROOT}/experiments/final_ali_ccp/logs
OUTPUT_ROOT=${PROJECT_ROOT}/experiments/final_ali_ccp/outputs/subset_param_search

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}"
cd "${PROJECT_ROOT}"

ALI_CCP_TRAIN_PATH=${ALI_CCP_TRAIN_PATH:-/work/home/maben/project/rec_sys/projects/Ali_CCP_REC/dataset/datasetsali_ccp_train.parquet}
ALI_CCP_VAL_PATH=${ALI_CCP_VAL_PATH:-/work/home/maben/project/rec_sys/projects/Ali_CCP_REC/dataset/datasetsali_ccp_val.parquet}
ALI_CCP_TEST_PATH=${ALI_CCP_TEST_PATH:-/work/home/maben/project/rec_sys/projects/Ali_CCP_REC/dataset/datasetsali_ccp_test.parquet}

if [ ! -f "${ALI_CCP_TRAIN_PATH}" ] || [ ! -f "${ALI_CCP_VAL_PATH}" ] || [ ! -f "${ALI_CCP_TEST_PATH}" ]; then
    echo "[ERROR] Ali-CCP files not found:"
    echo "  train=${ALI_CCP_TRAIN_PATH}"
    echo "  val=${ALI_CCP_VAL_PATH}"
    echo "  test=${ALI_CCP_TEST_PATH}"
    exit 1
fi

STAGES=${STAGES:-normal,entropy,asym,jd}
TRAIN_SUBSET_FRAC=${TRAIN_SUBSET_FRAC:-0.1}
TRAIN_SUBSET_SEED=${TRAIN_SUBSET_SEED:-42}
TRAIN_SUBSET_STRATIFY=${TRAIN_SUBSET_STRATIFY:-1}
MAX_EPOCHS=${MAX_EPOCHS:-4}
EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-2}
BATCH_SIZE=${BATCH_SIZE:-1024}
NUM_WORKERS=${NUM_WORKERS:-16}
ACCELERATOR=${ACCELERATOR:-auto}
DEVICES=${DEVICES:-1}
STRATEGY=${STRATEGY:-auto}
SEARCH_PROFILE=${SEARCH_PROFILE:-quick}
MAX_EXPERIMENTS=${MAX_EXPERIMENTS:-0}
SEEDS=${SEEDS:-42,2027,3407}

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

echo "[INFO] subset search config"
echo "  stages=${STAGES}"
echo "  profile=${SEARCH_PROFILE}"
echo "  subset_frac=${TRAIN_SUBSET_FRAC}"
echo "  epochs=${MAX_EPOCHS}"
echo "  seeds=${SEEDS}"
echo "  accelerator=${ACCELERATOR} devices=${DEVICES} strategy=${STRATEGY}"
echo "  max_experiments=${MAX_EXPERIMENTS}"

# ---------------------------
# Stage 1: Normal (architecture + base hparams)
# ---------------------------
if contains_stage normal; then
    echo "[INFO] stage=normal"

    if [ "${SEARCH_PROFILE}" = "mini" ]; then
        SB_HIDDENS=("256 128")
        EXPERT_OPTIONS=(6)
        PLE_NS_OPTIONS=(1)
    else
        SB_HIDDENS=("256 128" "384 192")
        EXPERT_OPTIONS=(4 6)
        PLE_NS_OPTIONS=(1 2)
    fi

    for SHARED in "${SB_HIDDENS[@]}"; do
        can_continue || break
        tag=$(echo "${SHARED}" | tr ' ' 'x')
        run_exp "subset_normal_share_bottom_sh${tag}" \
            --model share_bottom \
            --shared_hidden_dims ${SHARED} \
            --learning_rate 1e-3 \
            --dropout 0.2
    done

    for EXPERTS in "${EXPERT_OPTIONS[@]}"; do
        can_continue || break
        run_exp "subset_normal_moe_e${EXPERTS}" \
            --model moe \
            --num_experts ${EXPERTS} \
            --expert_hidden_dims 256 128 \
            --learning_rate 1e-3 \
            --dropout 0.2
    done

    for EXPERTS in "${EXPERT_OPTIONS[@]}"; do
        can_continue || break
        run_exp "subset_normal_mmoe_e${EXPERTS}" \
            --model mmoe \
            --num_experts ${EXPERTS} \
            --expert_hidden_dims 256 128 \
            --learning_rate 1e-3 \
            --dropout 0.2
    done

    for NS in "${PLE_NS_OPTIONS[@]}"; do
        can_continue || break
        run_exp "subset_normal_ple_ns${NS}_nsh2_l2" \
            --model ple \
            --num_specific_experts ${NS} \
            --num_shared_experts 2 \
            --num_levels 2 \
            --expert_hidden_dims 256 128 \
            --learning_rate 1e-3 \
            --dropout 0.2
    done
fi

# ---------------------------
# Stage 2: Entropy regularization
# ---------------------------
if contains_stage entropy && can_continue; then
    echo "[INFO] stage=entropy"
    if [ "${SEARCH_PROFILE}" = "full" ]; then
        ENT_LAMBDAS=(0.002 0.005 0.01 0.02)
    elif [ "${SEARCH_PROFILE}" = "mini" ]; then
        ENT_LAMBDAS=(0.01)
    else
        ENT_LAMBDAS=(0.005 0.01)
    fi

    for MODEL in moe mmoe ple; do
        for LAM in "${ENT_LAMBDAS[@]}"; do
            can_continue || break 2
            ltag=${LAM/./p}
            run_exp "subset_entropy_${MODEL}_l${ltag}" \
                --model ${MODEL} \
                --num_experts 6 \
                --expert_hidden_dims 256 128 \
                --num_specific_experts 1 \
                --num_shared_experts 2 \
                --num_levels 2 \
                --learning_rate 7e-4 \
                --dropout 0.25 \
                --use_entropy_reg \
                --lambda_entropy ${LAM}
        done
    done
fi

# ---------------------------
# Stage 3: Asymmetric projection
# ---------------------------
if contains_stage asym && can_continue; then
    echo "[INFO] stage=asym"
    if [ "${SEARCH_PROFILE}" = "mini" ]; then
        ASYM_VARIANTS=(base)
    else
        ASYM_VARIANTS=(base on10_rn)
    fi

    for MODEL in share_bottom moe mmoe; do
        can_continue || break
        if [ "${MODEL}" = "share_bottom" ]; then
            BASE_ARGS=(--model share_bottom --shared_hidden_dims 256 128)
        else
            BASE_ARGS=(--model ${MODEL} --num_experts 6 --expert_hidden_dims 256 128)
        fi

        for ASYM_KIND in "${ASYM_VARIANTS[@]}"; do
            can_continue || break 2
            if [ "${ASYM_KIND}" = "base" ]; then
                run_exp "subset_asym_${MODEL}_l1_t0" \
                    "${BASE_ARGS[@]}" \
                    --learning_rate 7e-4 \
                    --dropout 0.25 \
                    --use_asym_proj \
                    --asym_proj_lambda 1.0 \
                    --asym_proj_tau 0.0
            else
                run_exp "subset_asym_${MODEL}_l1_t0_on10_rn" \
                    "${BASE_ARGS[@]}" \
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
    done
fi

# ---------------------------
# Stage 4: TorchJD
# ---------------------------
if contains_stage jd && can_continue; then
    echo "[INFO] stage=jd"
    if [ "${SEARCH_PROFILE}" = "full" ]; then
        JD_METHODS=(upgrad mgda pcgrad graddrop)
    elif [ "${SEARCH_PROFILE}" = "mini" ]; then
        JD_METHODS=(upgrad)
    else
        JD_METHODS=(upgrad pcgrad)
    fi

    for MODEL in share_bottom moe mmoe ple; do
        if [ "${MODEL}" = "share_bottom" ]; then
            MODEL_ARGS=(--model share_bottom --shared_hidden_dims 256 128)
        elif [ "${MODEL}" = "ple" ]; then
            MODEL_ARGS=(--model ple --expert_hidden_dims 256 128 --num_specific_experts 1 --num_shared_experts 2 --num_levels 2)
        else
            MODEL_ARGS=(--model ${MODEL} --num_experts 6 --expert_hidden_dims 256 128)
        fi

        for METHOD in "${JD_METHODS[@]}"; do
            can_continue || break 2
            run_exp "subset_jd_${MODEL}_${METHOD}" \
                "${MODEL_ARGS[@]}" \
                --learning_rate 7e-4 \
                --dropout 0.25 \
                --use_torchjd \
                --aggregation_method "${METHOD}"
        done
    done
fi

echo "[INFO] subset parameter search finished. total_runs=${RUN_COUNT}"
echo "[INFO] outputs=${OUTPUT_ROOT}"

#!/bin/bash
#SBATCH --job-name=ali_ccp_final
#SBATCH --output=/work/home/maben/project/rec_sys/projects/multi_task_ranking/experiments/final_ali_ccp/logs/ali_ccp_final_%j.out
#SBATCH --error=/work/home/maben/project/rec_sys/projects/multi_task_ranking/experiments/final_ali_ccp/logs/ali_ccp_final_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:4
#SBATCH --mem=96G
#SBATCH --time=144:00:00
#SBATCH --partition=normal
#SBATCH --exclude=g01n01

set -u

PROJECT_ROOT=/work/home/maben/project/rec_sys/projects/multi_task_ranking
TRAIN_PY=${PROJECT_ROOT}/train_rank.py
LOG_DIR=${PROJECT_ROOT}/experiments/final_ali_ccp/logs
OUTPUT_ROOT=${PROJECT_ROOT}/experiments/final_ali_ccp/outputs

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}"
cd "${PROJECT_ROOT}"

ALI_CCP_TRAIN_PATH=${ALI_CCP_TRAIN_PATH:-/work/home/maben/project/rec_sys/projects/Ali_CCP_REC/dataset/datasetsali_ccp_train.parquet}
ALI_CCP_VAL_PATH=${ALI_CCP_VAL_PATH:-/work/home/maben/project/rec_sys/projects/Ali_CCP_REC/dataset/datasetsali_ccp_val.parquet}
ALI_CCP_TEST_PATH=${ALI_CCP_TEST_PATH:-/work/home/maben/project/rec_sys/projects/Ali_CCP_REC/dataset/datasetsali_ccp_test.parquet}

if [ ! -f "${ALI_CCP_TRAIN_PATH}" ]; then
    echo "[ERROR] train file not found: ${ALI_CCP_TRAIN_PATH}"
    exit 1
fi
if [ ! -f "${ALI_CCP_VAL_PATH}" ]; then
    echo "[ERROR] val file not found: ${ALI_CCP_VAL_PATH}"
    exit 1
fi
if [ ! -f "${ALI_CCP_TEST_PATH}" ]; then
    echo "[ERROR] test file not found: ${ALI_CCP_TEST_PATH}"
    exit 1
fi

echo "[INFO] Dataset:"
echo "  train=${ALI_CCP_TRAIN_PATH}"
echo "  val=${ALI_CCP_VAL_PATH}"
echo "  test=${ALI_CCP_TEST_PATH}"

STAGES=${STAGES:-baseline,entropy,torchjd,asymproj}
TRAIN_SUBSET_FRAC=${TRAIN_SUBSET_FRAC:-1.0}
TRAIN_SUBSET_SEED=${TRAIN_SUBSET_SEED:-42}
TRAIN_SUBSET_STRATIFY=${TRAIN_SUBSET_STRATIFY:-0}

COMMON_ARGS=(
    --train_path "${ALI_CCP_TRAIN_PATH}"
    --val_path "${ALI_CCP_VAL_PATH}"
    --test_path "${ALI_CCP_TEST_PATH}"
    --batch_size "${BATCH_SIZE:-1024}"
    --num_workers "${NUM_WORKERS:-32}"
    --max_epochs "${MAX_EPOCHS:-20}"
    --learning_rate "${LEARNING_RATE:-1e-3}"
    --dropout "${DROPOUT:-0.2}"
    --embedding_dim "${EMBEDDING_DIM:-32}"
    --tower_hidden_dims 64
    --accelerator "${ACCELERATOR:-gpu}"
    --devices "${DEVICES:-4}"
    --strategy "${STRATEGY:-ddp_find_unused_parameters_false}"
    --early_stop_patience "${EARLY_STOP_PATIENCE:-3}"
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

MODEL_ARGS=()
set_model_args() {
    local model=$1
    case "${model}" in
        share_bottom)
            MODEL_ARGS=(--model share_bottom --shared_hidden_dims 256 128)
            ;;
        moe)
            MODEL_ARGS=(--model moe --num_experts 6 --expert_hidden_dims 256 128)
            ;;
        mmoe)
            MODEL_ARGS=(--model mmoe --num_experts 6 --expert_hidden_dims 256 128)
            ;;
        ple)
            MODEL_ARGS=(--model ple --expert_hidden_dims 256 128 --num_specific_experts 1 --num_shared_experts 2 --num_levels 2)
            ;;
        *)
            echo "[ERROR] unknown model: ${model}"
            exit 1
            ;;
    esac
}

run_exp() {
    local name=$1
    shift
    local extra_args=("$@")

    echo "=========================================="
    echo "Experiment: ${name}"
    echo "=========================================="

    if python -u "${TRAIN_PY}" "${COMMON_ARGS[@]}" "${extra_args[@]}" --exp_dir "${OUTPUT_ROOT}/${name}"; then
        echo "[OK] ${name} completed."
    else
        echo "[ERROR] ${name} failed."
    fi
    echo ""
}

if contains_stage baseline; then
    echo "[INFO] Running stage: baseline"
    for model in share_bottom moe mmoe ple; do
        set_model_args "${model}"
        run_exp "ali_ccp_${model}_esmm_baseline" "${MODEL_ARGS[@]}"
    done
fi

if contains_stage entropy; then
    echo "[INFO] Running stage: entropy (gate polarization mitigation)"
    for model in moe mmoe ple; do
        set_model_args "${model}"
        for lam in 0.005 0.01 0.02; do
            lam_tag=${lam/./p}
            run_exp "ali_ccp_${model}_esmm_entropy_l${lam_tag}" \
                "${MODEL_ARGS[@]}" \
                --use_entropy_reg \
                --lambda_entropy "${lam}"
        done
    done
fi

if contains_stage torchjd; then
    echo "[INFO] Running stage: torchjd"
    for model in share_bottom moe mmoe ple; do
        set_model_args "${model}"
        for method in upgrad mgda pcgrad graddrop; do
            run_exp "ali_ccp_${model}_esmm_torchjd_${method}" \
                "${MODEL_ARGS[@]}" \
                --use_torchjd \
                --aggregation_method "${method}"
        done
    done
fi

if contains_stage asymproj; then
    echo "[INFO] Running stage: asymproj (share_bottom/moe/mmoe)"
    run_exp "ali_ccp_share_bottom_esmm_asymproj_l1_t0" \
        --model share_bottom \
        --shared_hidden_dims 256 128 \
        --use_asym_proj \
        --asym_proj_lambda 1.0 \
        --asym_proj_tau 0.0

    run_exp "ali_ccp_share_bottom_esmm_asymproj_l1_t0_on10_rn" \
        --model share_bottom \
        --shared_hidden_dims 256 128 \
        --use_asym_proj \
        --asym_proj_lambda 1.0 \
        --asym_proj_tau 0.0 \
        --asym_proj_only_10 \
        --asym_proj_restore_norm \
        --asym_proj_restore_max_scale 5.0

    run_exp "ali_ccp_moe_esmm_asymproj_l1_t0" \
        --model moe \
        --num_experts 6 \
        --expert_hidden_dims 256 128 \
        --use_asym_proj \
        --asym_proj_lambda 1.0 \
        --asym_proj_tau 0.0

    run_exp "ali_ccp_moe_esmm_asymproj_l1_t0_on10_rn" \
        --model moe \
        --num_experts 6 \
        --expert_hidden_dims 256 128 \
        --use_asym_proj \
        --asym_proj_lambda 1.0 \
        --asym_proj_tau 0.0 \
        --asym_proj_only_10 \
        --asym_proj_restore_norm \
        --asym_proj_restore_max_scale 5.0

    run_exp "ali_ccp_mmoe_esmm_asymproj_l1_t0" \
        --model mmoe \
        --num_experts 6 \
        --expert_hidden_dims 256 128 \
        --use_asym_proj \
        --asym_proj_lambda 1.0 \
        --asym_proj_tau 0.0

    run_exp "ali_ccp_mmoe_esmm_asymproj_l1_t0_on10_rn" \
        --model mmoe \
        --num_experts 6 \
        --expert_hidden_dims 256 128 \
        --use_asym_proj \
        --asym_proj_lambda 1.0 \
        --asym_proj_tau 0.0 \
        --asym_proj_only_10 \
        --asym_proj_restore_norm \
        --asym_proj_restore_max_scale 5.0
fi

echo "[INFO] All selected stages finished."

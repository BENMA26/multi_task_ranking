#!/bin/bash
#SBATCH --job-name=mtl_rank
#SBATCH --output=logs/mtl_esmm_rank_%j_log_weight.out
#SBATCH --error=logs/mtl_esmm_rank_%j_log_weight.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:4
#SBATCH --mem=96G
#SBATCH --time=144:00:00
#SBATCH --partition=normal
#SBATCH --exclude=g01n01
mkdir -p logs

MODELS=("moe")
METHODS=("upgrad")

TOTAL=${#MODELS[@]}
TOTAL=$((${#MODELS[@]} * ${#METHODS[@]}))
CURRENT=0

for MODEL in "${MODELS[@]}"; do
    for METHOD in "${METHODS[@]}"; do
        CURRENT=$((CURRENT + 1))

        if [ "$METHOD" == "none" ]; then
            EXP_NAME="${MODEL}_esmm_baseline_log_weight"
            USE_TORCHJD=""
        else
            EXP_NAME="${MODEL}_${METHOD}"
            USE_TORCHJD="--use_torchjd --aggregation_method ${METHOD}"
        fi

        echo "=========================================="
        echo "Experiment [${CURRENT}/${TOTAL}]: ${EXP_NAME}"
        echo "Model: ${MODEL} | Method: ${METHOD}"
        echo "=========================================="

        python -u /work/home/maben/project/rec_sys/projects/multi_task_ranking/train_rank.py \
            --model ${MODEL} \
            ${USE_TORCHJD} \
            --batch_size 1024 \
            --num_workers 32 \
            --max_epochs 20 \
            --learning_rate 1e-3 \
            --dropout 0.2 \
            --embedding_dim 32 \
            --accelerator gpu \
            --devices 4 \
            --esmm \
            --strategy ddp_find_unused_parameters_false \
            --exp_dir /work/home/maben/project/rec_sys/projects/multi_task_ranking/experiments/basic_experiment/outputs/${EXP_NAME}

        if [ $? -ne 0 ]; then
            echo "[ERROR] Experiment ${EXP_NAME} failed! Continuing to next..."
        else
            echo "[OK] Experiment ${EXP_NAME} completed."
        fi

        echo ""
    done
done

echo "All ${TOTAL} experiments finished."
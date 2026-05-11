#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="../data/cifar10_download"
CHECKPOINT_DIR="./checkpoints"
RESULTS_DIR="./results"
STDOUT_DIR="./stdout_logs"

BASE_RUN="cifar10_resnet20"
BASE_SEED=6

mkdir -p "$CHECKPOINT_DIR" "$RESULTS_DIR" "$STDOUT_DIR"

for N_SCALE in $(seq 4 0.5 5); do
  N_FMT=$(printf "%.1f" "$N_SCALE")
  N_TAG=${N_FMT/./p}

  RESULTS_FILE="${RESULTS_DIR}/nscale_${N_TAG}.csv"

  # Uncomment this if you want to start fresh each time for this N_scale
  # rm -f "$RESULTS_FILE"

  for REPEAT in $(seq 0 4); do
    SEED=$((BASE_SEED + REPEAT))
    RUN_NAME="${BASE_RUN}_N${N_TAG}_seed${SEED}"
    OUT_LOG="${STDOUT_DIR}/${RUN_NAME}.log"

    echo "=== N_scale=${N_FMT} | run $((REPEAT+1))/10 | seed=${SEED} ==="

    python main_cifar10.py \
      --data_root "$DATA_ROOT" \
      --batch_size 256 \
      --epochs 300 \
      --size 20 \
      --base_lr 3e-3 \
      --weight_decay 0.02 \
      --label_smoothing 0.05 \
      --num_workers 4 \
      --run_name "$RUN_NAME" \
      --checkpoint_dir "$CHECKPOINT_DIR" \
      --no-compile \
      --amp \
      --use_ggd \
      --N_scale "$N_FMT" \
      --seed "$SEED" \
      --results_file "$RESULTS_FILE" \
      2>&1 | tee "$OUT_LOG"
  done
done
 
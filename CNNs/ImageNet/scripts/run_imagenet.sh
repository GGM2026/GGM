#!/usr/bin/env bash
set -euo pipefail

ulimit -l unlimited || echo "Warning: could not raise memlock (need higher hard limit / sudo / limits.conf)"
DATA_ROOT="../imagenet_download"

NPROC_PER_NODE="2"

MODEL="resnet"
SIZE="18"
N_FACTOR="1.0"
PRELU_FLAG="--prelu"

IMG_SIZE="224"
VAL_FRACTION=""
SPLIT_SEED="1337"

EPOCHS="150"
BATCH_SIZE="128"
NUM_WORKERS="40"
ACCUM_STEPS="1"
BASE_LR="3e-3"
WEIGHT_DECAY="0"
LABEL_SMOOTHING="0.1"
SEED="1337"

OPTIMIZER="adamw"
MOMENTUM="0.9"
NESTEROV_FLAG="${NESTEROV_FLAG:-}"

NUM_RUNS="1"
SEED_STEP="1000"
RESUME_FLAG=""
TEST_FLAG=""
AMP_FLAG="--amp"
COMPILE_FLAG=""

FULL_PRECISION_FLAG=""

RUN_NAME="${MODEL}${SIZE}_ggm_nu${N_FACTOR}_Gregular"
CKPT_DIR="./checkpoints/imagenet"

double_residual_flag=""

torchrun \
  --master_port=29502 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  train.py \
  --dataset imagenet \
  --run_name "${RUN_NAME}" \
  --ckpt_dir "${CKPT_DIR}" \
  ${RESUME_FLAG} \
  --seed "${SEED}" \
  --data_root "${DATA_ROOT}" \
  --img_size "${IMG_SIZE}" \
  --model "${MODEL}" \
  --size "${SIZE}" \
  --N_factor "${N_FACTOR}" \
  ${PRELU_FLAG} \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --accumulation_steps "${ACCUM_STEPS}" \
  --base_lr "${BASE_LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --label_smoothing "${LABEL_SMOOTHING}" \
  ${FULL_PRECISION_FLAG} \
  --num_runs "${NUM_RUNS}" \
  --seed_step "${SEED_STEP}" \
  --optimizer "${OPTIMIZER}" \
  --momentum "${MOMENTUM}" \
  ${NESTEROV_FLAG} \
  ${AMP_FLAG} \
  ${TEST_FLAG} \
  ${double_residual_flag} \
  ${COMPILE_FLAG} \
  | tee "train_${RUN_NAME}.log"
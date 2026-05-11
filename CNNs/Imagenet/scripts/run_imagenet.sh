#!/usr/bin/env bash
set -euo pipefail

# Allow large pinned/locked host memory allocations (CUDA / pinned DataLoader, etc.)
ulimit -l unlimited || echo "Warning: could not raise memlock (need higher hard limit / sudo / limits.conf)"
# -------------------------------
# ImageNet DDP run w/ torchrun
# -------------------------------

DATA_ROOT="../imagenet_download"   # should contain train/ and val/ (or whatever train.py expects)

# DDP config (single node)
NPROC_PER_NODE="1" # Number of GPUS you want to allocate for this experiment. 

# Experiment naming / checkpoints
# The checkpoints will be saved in "CKPT_DIR/RUN_NAME"
RUN_NAME="resnet18_ggd_bs256_ep200_N3"
CKPT_DIR="./checkpoints/imagenet"

# Model config
MODEL="resnet" # choices: resnet, vgg
SIZE="18" # choices: 20, 18, 34
N_SCALE="3.0" # Redundant here. 
PRELU_FLAG="--prelu" # Set to "--prelu" if you want prelu as the activation. Otherwise it stays as the default activation from the network.

# Data / split (optional: only used if your train.py supports these for imagenet)
IMG_SIZE="224"
VAL_FRACTION=""   # e.g. "0.1" if you ever do train/val split from train set
SPLIT_SEED="1337" # only relevant if using val_fraction

# Training
EPOCHS="200"
BATCH_SIZE="256" # per-GPU, The baseline used 256 overall with 32 per gpu for 8 gpus. We have 4 so it is 64
NUM_WORKERS="40" # Number of CPU cores
ACCUM_STEPS="1" # How many steps to accumulate gradients. 
BASE_LR="0.1"
WEIGHT_DECAY="1e-4" # Changed from 1e-4
LABEL_SMOOTHING="0.1"
SEED="1337"

# Optimizer
OPTIMIZER="sgd" # choices: adamw, sgd
MOMENTUM="0.9"
NESTEROV_FLAG="${NESTEROV_FLAG:-}"

# GGD / precision knobs
CHUNK_N="10000" # Ignored for GGM. Works for GGD. 
GGM_FLAG="" # If you want to activate the GGM layers. If full precision is activated this would be ignored.

# Multi-run / resume / test
NUM_RUNS="1"
SEED_STEP="1000"
RESUME_FLAG="--resume" # set to "--resume" to enable resume 
TEST_FLAG="" # set to "--test" to enable. When you have checkpoint and only want to check accuracy.
AMP_FLAG="--amp" # set to "--amp" to enable

FULL_PRECISION_FLAG=""  # set to "--full_precision" if you want to disable Conv2dGGD/ConvGGM swapping. 

# Double Residual for Conv Blocks -> Set to --double_residual
double_residual_flag="" 

torchrun \
  --master_port=29503 \
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
  --N_scale "${N_SCALE}" \
  ${PRELU_FLAG} \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --accumulation_steps "${ACCUM_STEPS}" \
  --base_lr "${BASE_LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --label_smoothing "${LABEL_SMOOTHING}" \
  ${FULL_PRECISION_FLAG} \
  --chunk_N "${CHUNK_N}" \
  --num_runs "${NUM_RUNS}" \
  --seed_step "${SEED_STEP}" \
  --optimizer "${OPTIMIZER}" \
  --momentum "${MOMENTUM}" \
  ${NESTEROV_FLAG} \
  ${AMP_FLAG} \
  ${TEST_FLAG} \
  ${GGM_FLAG} \
  ${double_residual_flag} \
  --no-compile

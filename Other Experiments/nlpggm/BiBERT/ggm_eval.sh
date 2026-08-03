#!/usr/bin/env bash
set -euo pipefail

TASK_DIR="../data/"
TASK_NAME="SST-2"
TASK_NAME_LOWER="sst-2"
TEACHER_MODEL="../dynabert/SST-2"

BEST_QUANT_MODEL="./outputs/${TASK_NAME_LOWER}"

RATIO=2.0

OUTPUT_DIR="./results/BiBERT_GGM_eval"
LOG_DIR="${OUTPUT_DIR}/logs/${TASK_NAME}/${RATIO}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/$(date '+%Y-%m-%d-%H-%M-%S')-${TASK_NAME}-do_eval-ggm-${RATIO}.log"

CUDA_VISIBLE_DEVICES=0 python quant_task_glue.py \
  --data_dir "${TASK_DIR}" \
  --teacher_model "${TEACHER_MODEL}" \
  --student_model "${BEST_QUANT_MODEL}" \
  --task_name "${TASK_NAME}" \
  --output_dir "${OUTPUT_DIR}" \
  --seed 42 \
  --weight_bits 1 \
  --embedding_bits 1 \
  --input_bits 1 \
  --batch_size 32 \
  --do_eval \
  --use_ggm \
  --ggm_ratio ${RATIO} \
  2>&1 | tee "${LOG_FILE}"
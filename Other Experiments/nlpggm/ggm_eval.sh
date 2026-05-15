#!/usr/bin/env bash
set -euo pipefail

TASK_DIR="./data"
TASK_NAME="SST-2"

TEACHER_MODEL="./dynabert/SST-2"
BEST_QUANT_MODEL="./results/BiBERT_GGM_eval/sst-2"

OUTPUT_DIR="./results/BiBERT_GGM_eval"
LOG_DIR="./results/BiBERT_GGM_eval/logs"


LOG_FILE="${LOG_DIR}/$(date '+%Y-%m-%d-%H-%M-%S')-${TASK_NAME}-do_eval.log"

CUDA_VISIBLE_DEVICES=0 python quant_task_glue.py \
  --data_dir "${TASK_DIR}" \
  --teacher_model "${TEACHER_MODEL}" \
  --student_model "${BEST_QUANT_MODEL}" \
  --task_name "${TASK_NAME}" \
  --seed 42 \
  --weight_bits 1 \
  --embedding_bits 1 \
  --input_bits 1 \
  --batch_size 32 \
  --do_eval \
  --use_ggm \
  --ggm_n_factor 6.0 \
  --ggm_eps 1e-5 
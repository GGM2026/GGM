#!/usr/bin/env bash
# One-command setup + run for the 1-bit GGM ViT on ImageNet-1k.
#
#   bash run.sh          # install deps, download ImageNet val, run inference
#   bash run.sh train    # install deps, download ImageNet train+val, train
#
# ImageNet-1k is gated on Hugging Face: run `hf auth login` once and
# accept the terms at https://huggingface.co/datasets/ILSVRC/imagenet-1k first.
set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-eval}"

echo "==> Installing dependencies"
pip install -r requirements.txt

if [ "$MODE" = "train" ]; then
    echo "==> Downloading + organizing ImageNet (train + val) into ./imagenet"
    python prepare_imagenet.py --output ./imagenet --splits train,validation
    echo "==> Starting training"
    python main.py
else
    echo "==> Downloading + organizing ImageNet (val) into ./imagenet"
    python prepare_imagenet.py --output ./imagenet --splits validation
    echo "==> Running inference (reproduce Top-1 56.19% / Top-5 79.61%)"
    python evaluate.py --data_dir ./imagenet/val
fi

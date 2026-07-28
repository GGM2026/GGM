#!/usr/bin/env bash
# Prepares PTB-XL (PhysioNet 1.0.3, ~1.7 GB) for the run scripts in this folder.
#
#   bash setup_data.sh
#
# Writes what they read via --root_path ./PTB-XL :
#   ./PTB-XL/Feature/feature_<pid>.npy   windowed ECG per patient
#   ./PTB-XL/Label/label.npy             columns [class, patient_id]
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
PTBXL_URL="https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"

for c in curl unzip; do
    command -v "$c" >/dev/null 2>&1 || { echo "ERROR: '$c' is required." >&2; exit 1; }
done

echo "==> PTB-XL (PhysioNet 1.0.3)"

if [ -f "./PTB-XL/Label/label.npy" ]; then
    echo "    already prepared at ./PTB-XL, nothing to do"
    exit 0
fi

if [ ! -d "./ptb-xl/records500" ]; then
    echo "    downloading (~1.7 GB, public, no login)"
    curl -L --fail --retry 3 -o ptbxl.zip "$PTBXL_URL"
    echo "    extracting"
    unzip -q ptbxl.zip -d ptbxl_tmp
    # the archive nests everything under one long directory name
    inner="$(find ptbxl_tmp -maxdepth 1 -mindepth 1 -type d | head -1)"
    rm -rf ./ptb-xl && mv "$inner" ./ptb-xl
    rm -rf ptbxl_tmp ptbxl.zip
    echo "    raw data in ./ptb-xl"
else
    echo "    raw data already at ./ptb-xl, skipping download"
fi

# reads ./ptb-xl/records500 + ./ptb-xl/ptbxl_database.csv, writes ./PTB-XL
echo "    preprocessing -> ./PTB-XL/{Feature,Label}"
"$PYTHON" data_preprocessing/ptbxl_preprocess.py

echo
echo "Layout check:"
for p in "./PTB-XL/Feature" "./PTB-XL/Label/label.npy"; do
    [ -e "$p" ] && echo "  present  $p" || echo "  MISSING  $p"
done

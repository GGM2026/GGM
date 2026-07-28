#!/usr/bin/env bash
# Prepares ADFTD (OpenNeuro ds004504) for the run scripts in this folder.
#
#   bash setup_data.sh
#
# Writes what they read via --root_path ../ADFTD/ :
#   ../ADFTD/Feature/*.npy      one array per subject
#   ../ADFTD/Label/label.npy    columns [class, subject_id], 0=HC 1=FTD 2=AD
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
ADFTD_BASE="https://s3.amazonaws.com/openneuro.org/ds004504"
RAW_DIR="../ADFTD_raw"
OUT_DIR="../ADFTD"

command -v curl >/dev/null 2>&1 || { echo "ERROR: curl is required." >&2; exit 1; }

echo "==> ADFTD (OpenNeuro ds004504)"

if [ -f "$OUT_DIR/Label/label.npy" ]; then
    echo "    already prepared at $OUT_DIR, nothing to do"
    exit 0
fi

if [ ! -d "$RAW_DIR" ]; then
    echo "    downloading raw EEG (public, no login)"
    mkdir -p "$RAW_DIR"
    curl -L --fail --retry 3 -o "$RAW_DIR/participants.tsv" "$ADFTD_BASE/participants.tsv"
    # EEGLAB single-file format: the .set carries the data, there is no .fdt
    while IFS=$'\t' read -r sub _rest; do
        [ "$sub" = "participant_id" ] && continue
        [ -n "$sub" ] || continue
        mkdir -p "$RAW_DIR/$sub/eeg"
        curl -sL --fail --retry 2 \
            -o "$RAW_DIR/$sub/eeg/${sub}_task-eyesclosed_eeg.set" \
            "$ADFTD_BASE/$sub/eeg/${sub}_task-eyesclosed_eeg.set" \
            || echo "    warning: $sub not retrieved"
    done < "$RAW_DIR/participants.tsv"
    echo "    raw EEG in $RAW_DIR"
else
    echo "    raw data already at $RAW_DIR, skipping download"
fi

if [ ! -f "data_preprocessing/adftd_preprocess.py" ]; then
    cat >&2 <<MSG

    Raw EEG is ready in $RAW_DIR.

    To build the arrays the loader reads, place adftd_preprocess.py in
    data_preprocessing/ and re-run this script. It should write:

        $OUT_DIR/Feature/*.npy      one array per subject
        $OUT_DIR/Label/label.npy    columns [class, subject_id]
                                    class 0=HC, 1=FTD, 2=AD
                                    (participants.tsv Group: C=0, F=1, A=2)

MSG
    exit 1
fi

echo "    preprocessing -> $OUT_DIR/{Feature,Label}"
"$PYTHON" data_preprocessing/adftd_preprocess.py

echo
echo "Layout check:"
for p in "$OUT_DIR/Feature" "$OUT_DIR/Label/label.npy"; do
    [ -e "$p" ] && echo "  present  $p" || echo "  MISSING  $p"
done

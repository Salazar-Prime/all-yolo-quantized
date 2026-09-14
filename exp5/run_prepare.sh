#!/usr/bin/env bash
set -euo pipefail
cd /home/varun/work/all-yolo-quantized
run="${1:?run ID required}"
gpu="${2:?physical GPU index required}"
if [[ -f "exp5/runs/$run/gpu-$gpu.stop" ]]; then exit 0; fi
export CUDA_VISIBLE_DEVICES="$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)"
export YOLO_AUTOINSTALL=false OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export YOLO_CONFIG_DIR="$PWD/exp5/.cache/settings" MPLCONFIGDIR="$PWD/exp5/.cache/matplotlib"
mkdir -p "exp5/runs/$run"
exec 8>"exp5/runs/$run/gpu-$gpu.lock"
flock -n 8 || exit 0
while IFS=, read -r model rest; do
    if [[ -f "exp5/runs/$run/gpu-$gpu.stop" ]]; then exit 0; fi
    if [[ "$model" == model ]]; then continue; fi
    folder="exp5/runs/$run/$model"
    mkdir -p "$folder"
    exec 9>"$folder/prepare.lock"
    if ! flock -n 9; then continue; fi
    if [[ -f "$folder/prepare-fp32.exit" && -f "$folder/prepare-ptq.exit" && -f "$folder/prepare-qat.exit" ]]; then continue; fi
    while (( $(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits) > 500 )); do
        sleep 5
    done
    printf '%s GPU %s preparing %s\n' "$(date -Is)" "$gpu" "$model"
    for phase in fp32 ptq qat; do
        if [[ -f "$folder/prepare-$phase.exit" ]]; then continue; fi
        set +e
        exp5/.venv/bin/python -u exp5/prepare.py "$model" "$run" "$phase" >"$folder/prepare-$phase.log" 2>&1
        rc=$?
        set -e
        printf '%s\n' "$rc" >"$folder/prepare-$phase.exit"
    done
done <exp5/models.csv

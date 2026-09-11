#!/usr/bin/env bash
set -euo pipefail

experiment_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_dir="$(dirname "$experiment_dir")"
run_name="${1:?Usage: bash exp3/run_rainbow.sh RUN_NAME}"
output_dir="${experiment_dir}/runs/${run_name}"
python_bin="${experiment_dir}/.venv/bin/python"

[[ "$(hostname -s)" == digital-ag ]] || { echo 'Exp3 must run on Rainbow.' >&2; exit 2; }
gpu_uuid="$(nvidia-smi -i 1 --query-gpu=uuid --format=csv,noheader)"
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -Fxq "$gpu_uuid"; then
    echo 'Rainbow GPU 1 is already in use.' >&2
    exit 2
fi

mkdir -p "$output_dir" "$experiment_dir/.cache/settings" "$experiment_dir/.cache/matplotlib"
cd "$repository_dir"
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=16
export YOLO_AUTOINSTALL=false
export YOLO_CONFIG_DIR="$experiment_dir/.cache/settings"
export MPLCONFIGDIR="$experiment_dir/.cache/matplotlib"
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
printf '%s\n' "$$" > "$output_dir/pid.txt"
date --iso-8601=seconds > "$output_dir/started-at.txt"
trap 'rc=$?; printf "%s\n" "$rc" > "$output_dir/exit-code.txt"; date --iso-8601=seconds > "$output_dir/finished-at.txt"' EXIT
nvidia-smi > "$output_dir/gpu-start.txt"
git rev-parse HEAD > "$output_dir/source-revision.txt"
"$python_bin" -u exp3/run.py --manifest exp3/dataset.json --output "$output_dir" --smoke > "$output_dir/smoke.log" 2>&1
"$python_bin" -u exp3/run.py --manifest exp3/dataset.json --output "$output_dir" > "$output_dir/evaluation.log" 2>&1
nvidia-smi > "$output_dir/gpu-finish.txt"

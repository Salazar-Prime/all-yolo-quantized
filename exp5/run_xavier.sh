#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
model_dir="${1:?model directory required}"
service="${2-}"
if (( $# >= 2 )); then shift 2; else shift; fi
if (( $# == 0 )); then set -- ptq qat onnx fp32 fp16; fi
telemetry="$(dirname -- "$model_dir")/telemetry.jsonl"
export YOLO_AUTOINSTALL=false YOLO_CONFIG_DIR="$PWD/exp5/.cache/settings" MPLCONFIGDIR="$PWD/exp5/.cache/matplotlib"
mkdir -p "$YOLO_CONFIG_DIR" "$MPLCONFIGDIR"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
if [[ -n "$service" ]] && systemctl --user is-active --quiet "$service"; then
    systemctl --user stop "$service"
    trap 'systemctl --user start "$service"' EXIT
fi
exp5/.venv/bin/python -c 'import torch; assert torch.ones(1, device="cuda").sum().item() == 1'
for variant in "$@"; do
    start=$(wc -l <"$telemetry")
    (( start > 0 ))
    rc=0
    python_bin=exp5/.venv/bin/python
    if [[ "$variant" == onnx ]]; then python_bin=exp5/.venv-onnx/bin/python; fi
    mkdir -p "$model_dir/$variant"
    if [[ "$variant" != onnx ]]; then
        set +e
        exp5/.venv/bin/python -u exp5/benchmark.py "$model_dir" "$variant" build >"$model_dir/$variant/build.log" 2>&1
        rc=$?
        set -e
        printf '%s\n' "$rc" >"$model_dir/$variant/build.exit"
    fi
    if (( rc == 0 )) && [[ "$variant" == *_fp16 ]]; then
        set +e
        "$python_bin" -u exp5/benchmark.py "$model_dir" "$variant" evaluate --images 32 --passes 1 >"$model_dir/$variant/smoke.log" 2>&1
        rc=$?
        set -e
        printf '%s\n' "$rc" >"$model_dir/$variant/smoke.exit"
        mkdir -p "$model_dir/$variant/smoke"
        for result in "$model_dir/$variant/pass1" "$model_dir/$variant/pass1.json"; do
            if [[ -e "$result" ]]; then mv "$result" "$model_dir/$variant/smoke/"; fi
        done
    fi
    if (( rc == 0 )); then
        "$python_bin" exp5/benchmark.py "$model_dir" "$variant" idle
        set +e
        "$python_bin" -u exp5/benchmark.py "$model_dir" "$variant" evaluate >"$model_dir/$variant/evaluate.log" 2>&1
        rc=$?
        set -e
        printf '%s\n' "$rc" >"$model_dir/$variant/evaluate.exit"
        if [[ "$variant" == onnx || "$variant" == *_fp16 ]] && (( rc == 0 )); then
            set +e
            "$python_bin" -u exp5/benchmark.py "$model_dir" "$variant" profile >"$model_dir/$variant/profile.log" 2>&1
            rc=$?
            set -e
            printf '%s\n' "$rc" >"$model_dir/$variant/profile.exit"
        fi
    fi
    sed -n "${start},$(wc -l <"$telemetry")p" "$telemetry" >"$model_dir/$variant/telemetry.jsonl"
    cp "$model_dir/device.json" "$model_dir/$variant/device.json"
done

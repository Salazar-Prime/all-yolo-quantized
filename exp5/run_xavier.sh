#!/usr/bin/env bash
set -euo pipefail
cd /home/usr/work/all-yolo-quantized
model_dir="${1:?model directory required}"
export YOLO_AUTOINSTALL=false YOLO_CONFIG_DIR="$PWD/exp5/.cache/settings" MPLCONFIGDIR="$PWD/exp5/.cache/matplotlib"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
systemctl --user stop yolo26s-rtsp.service
trap 'systemctl --user start yolo26s-rtsp.service' EXIT
for variant in onnx fp32 fp16 ptq qat; do
    python_bin=exp5/.venv/bin/python
    if [[ "$variant" == onnx ]]; then python_bin=exp5/.venv-onnx/bin/python; fi
    mkdir -p "$model_dir/$variant"
    if [[ "$variant" != onnx ]]; then
        set +e
        exp5/.venv/bin/python -u exp5/benchmark.py "$model_dir" "$variant" build >"$model_dir/$variant/build.log" 2>&1
        rc=$?
        set -e
        printf '%s\n' "$rc" >"$model_dir/$variant/build.exit"
        if (( rc != 0 )); then continue; fi
    fi
    "$python_bin" exp5/benchmark.py "$model_dir" "$variant" idle
    set +e
    "$python_bin" -u exp5/benchmark.py "$model_dir" "$variant" evaluate >"$model_dir/$variant/evaluate.log" 2>&1
    rc=$?
    set -e
    printf '%s\n' "$rc" >"$model_dir/$variant/evaluate.exit"
    if [[ "$variant" == onnx ]] && (( rc == 0 )); then
        set +e
        "$python_bin" -u exp5/benchmark.py "$model_dir" "$variant" profile >"$model_dir/$variant/profile.log" 2>&1
        rc=$?
        set -e
        printf '%s\n' "$rc" >"$model_dir/$variant/profile.exit"
    fi
done

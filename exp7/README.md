# Experiment 7: Effect of TensorRT workspace size on model performance

**Status: `exp7-20260922` is continuing with FP16, PTQ + FP16, and QAT + FP16 only, all at 2 GiB.**

## Intent

Test whether increasing TensorRT's workspace limit from **1,024 MiB (1 GiB)** to **2,048 MiB (2 GiB)** improves inference
performance on Ubuntu Xavier NX devices. Measure inference speed, memory use, build time, and accuracy.

The workspace limit affects which layer implementations TensorRT can select. A larger limit may permit faster
implementations, but it is a ceiling rather than a reservation and does not guarantee a speedup. Weights, activations,
and other allocations also consume Xavier's shared CPU/GPU RAM.

See NVIDIA's [workspace-size guidance](https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/troubleshooting-faq.html).

## Comparison and baseline reuse

Run all **28 models** in [Exp5's manifest](../exp5/models.csv), with three 2 GiB variants per model:
**84 engine configurations and 252 full-test passes**. Reuse archived 1 GiB results as requested; do not rebuild or reevaluate
any baseline. No retraining, recalibration, or ONNX changes are needed.

| Configuration   | Variant    | Precision flags | Archived 1 GiB baseline |
| --------------- | ---------- | --------------- | ----------------------- |
| FP16            | `fp16`     | `--fp16`        | Exp5                    |
| INT8 PTQ + FP16 | `ptq_fp16` | `--int8 --fp16` | Exp6                    |
| INT8 QAT + FP16 | `qat_fp16` | `--int8 --fp16` | Exp6                    |

The new build option is `--memPoolSize=workspace:2048`; archived builds used `--memPoolSize=workspace:1024`.
FP16 baselines come from Exp5; INT8 + FP16 baselines come from Exp6. All 84 selected baseline rows have three completed passes and recorded 1 GiB builds with TF32 disabled. Baseline results and
provenance are saved in the run directory. Original sources are `exp5/runs/exp5-20260914` and `exp6/runs/exp6-20260918`.

Reuse the identical ONNX inputs and precision flags, input shape, quantization scales, and TensorRT/CUDA versions.
Verify input SHA-256 hashes on the target before building. Keep TF32 disabled. Each board starts with a fresh campaign
timing cache and reuses it across builds; archive cache snapshots before and after builds when present. Original
baseline starting caches are not reproducible, so cache history remains a comparison limitation.

Exp6 changed the floating-point precision allowed alongside INT8. Exp7 changes workspace within each precision
configuration. Because baselines are reused, the resulting ratios are **historical comparisons**, not paired
measurements isolating workspace alone. Device, OS, clocks, thermals, competing workloads, and timing-cache history
can differ. Report these differences rather than attributing the entire FPS change to workspace.

## Precision policy correction

Only `fp16`, `ptq_fp16`, and `qat_fp16` are scheduled. Every new INT8 engine enables `--fp16` for supported
floating-point operations. INT8-only `ptq` and `qat` cases were removed at the user's request after the campaign began.
Previously collected INT8-only artifacts are retained for provenance but excluded from the active results and monitor.
Interrupted INT8-only work is stopped; completed FP16 and INT8 + FP16 stages are preserved when restarting queues.
The original protocol and baseline index are retained under the run's `precision-change/` directory.

## Devices and measurement

[protocol.json](protocol.json) owns the assignments. The campaign assignments cover six devices: `ubuntu`,
`ubuntu-2`, `ubuntu-3`, `ubuntu-4`, `ubuntu-5`, and `ubuntu-6`. The user requested proceeding without offline `ubuntu-1`
and `ubuntu-7`; their models were redistributed across those boards. During the precision correction, `ubuntu-3`
was recovered after a reboot and `ubuntu-6` was offline; its controller remains stopped pending recovery. Exclude
`soysan` from execution.
Build and evaluate one model/variant at a time per board. Anvil runs controllers and archives artifacts; inference and
engine construction run only on Xavier.

- Reuse the [Exp5 evaluation protocol](../exp5/README.md#data-and-measurement-protocol): batch size 1, input
  `1x3x640x640`, square letterboxing, the same 4,579 test images, 100 warmup inferences, and three measured full-test passes.
- Preserve the one-to-many detection head, external NMS, confidence 0.001, NMS IoU 0.5, and maximum 300 detections.
- Verify the deployed package versions and SHA-256 hashes of the source and all 9,158 dataset files before launch.
  Retain `MODE_20W_6CORE`, six online CPU cores, and automatic clocks. Record actual device/software identities.
- Run a 32-image smoke evaluation after each successful build. Measure synchronized inference latency/FPS separately
  from preprocessing + inference + postprocessing FPS. Exclude build, initialization, and warmup from inference timing.
- Report all passes, mAP@0.5, and mAP@0.5:0.95. Do not tune against test accuracy. Changed implementations can affect
  numerical outputs even when precision flags and weights remain unchanged.
- Capture RAM, swap, utilization, clocks, and temperatures with timestamped `tegrastats` samples. Summarize build and
  inference windows separately. Current SSH access lacks passwordless sudo, so telemetry is unprivileged and board
  power/energy readings are unavailable; missing power coverage must not be interpreted as zero power.
- Profile layer timings separately after measured passes. Archive build commands, ONNX and engine hashes, detailed
  layer inspection, timing caches, raw telemetry, failures, and verified per-model archive receipts under `exp7/`.

There is no alternating 1 GiB/2 GiB execution order because the 1 GiB measurements are historical. New variants run in
protocol order on each device, with fresh warmup before each measurement pass.

## Execution and results

Reuse the existing [controller](../exp5/run.py), [benchmark](../exp5/benchmark.py), and
[summarizer](../exp5/summarize.py). The controller sets `TRT_WORKSPACE_MIB` from the protocol; existing campaigns default
to 1,024 MiB. The Xavier wrapper now defaults to `ptq_fp16` and `qat_fp16` for INT8 work. The 2 GiB campaign runs
smoke evaluation and separate profiling for every selected variant.

After device, dataset, and telemetry validation, start one persistent controller per configured device:

```bash
python3 -u exp5/run.py exp7-20260922 --protocol exp7/protocol.json --device ubuntu
```

The existing [Ubuntu SSH aliases](../exp6/ssh_config) use each board's Anvil reverse tunnel. Collected results live under
`exp7/runs/exp7-20260922/`, including `results.csv`, `status.json`, `baseline-results.csv`, `baseline-provenance.json`,
setup records, controller logs, per-pass measurements, and engine inspection. Each model's remote artifacts are removed
only after collection and SHA-256 verification. Completed archives are skipped on controller restart.

For a live snapshot:

```bash
bash .panepilot/actions/xavier-progress.sh --protocol exp7/protocol.json
```

## Interpreting results

Calculate **workspace speedup = new 2 GiB inference FPS / archived 1 GiB inference FPS** for each model and configuration.
Report all pass measurements, mAP differences in percentage points, build time, and memory cost alongside the old and
new device identities. Some older Exp5 cases lack per-variant device metadata; preserve that provenance limitation.

A repeatable gain beyond observed pass variation is useful evidence, but historical comparisons cannot establish that
workspace alone caused the difference. Report unchanged performance, slower results, and memory failures equally;
a larger workspace is not assumed to be better, and gains may differ between FP16 and INT8.

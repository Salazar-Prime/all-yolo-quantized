# Experiment 6: INT8 with FP16 for remaining layers

**Status: launch in progress for `exp6-20260918`; the first engine build is running on `ubuntu-1`.**

## Intent

Determine whether enabling FP16 for the floating-point portions of Exp5's INT8 PTQ and QAT engines improves inference
speed. Compare each new engine against archived Exp5 INT8 and FP16 results, while measuring accuracy.
The central question is: **does INT8 with FP16 enabled for remaining layers recover the speed lost to FP32 work, and can
it outperform FP16 alone?**

Reuse the existing calibrated PTQ and trained QAT ONNX files. This experiment changes the TensorRT build configuration;
it does not require new calibration, retraining, or changes to quantization scales or Q/DQ placement.

## Why FP16 was faster in Exp5

Lower arithmetic precision does not guarantee lower whole-model latency. Exp5's
[engine builder](../exp5/benchmark.py) enables `--fp16` for FP16 engines but only `--int8` for PTQ/QAT engines.
The inspected YOLO26n and YOLO11n INT8 engines contain INT8 and FP32 output tensor formats, with no FP16 outputs.
Thus, their unquantized portions can retain more expensive FP32 work. Exp5's
[protocol](../exp5/protocol.json) also deliberately leaves the detection head and attention positional convolution
outside quantization.

The archived YOLO26n results illustrate the difference:

| Measurement                    |  FP16 | INT8 PTQ | INT8 QAT |
| ------------------------------ | ----: | -------: | -------: |
| Inference FPS                  | 95.88 |    61.75 |    61.06 |
| Engine layers                  |   228 |      334 |      335 |
| Layers identified as reformats |    55 |      124 |      124 |

Sources: [Exp5 results](../exp5/runs/exp5-20260914/results.csv) and YOLO26n engine inspection files for
[FP16](../exp5/runs/exp5-20260914/yolo26n/fp16/layers.json),
[PTQ](../exp5/runs/exp5-20260914/yolo26n/ptq/layers.json), and
[QAT](../exp5/runs/exp5-20260914/yolo26n/qat/layers.json). Reformat counts identify entries whose `Name` or `LayerType`
contains `reformat`, case-insensitively; they are not counts of quantization operations alone.

Three mechanisms could explain the slowdown:

- **FP32 work remains:** faster INT8 convolutions may not compensate for floating-point portions running in FP32.
- **Conversions and fusion differ:** quantization/dequantization and tensor layout changes can add work or prevent
  operations from being fused. The higher layer and reformat counts support investigating this possibility.
- **Batch size 1 exposes overhead:** kernel launches and conversions can take a substantial share of latency when
  each inference processes only one image.

These are hypotheses about elapsed time. Tensor output formats and layer counts do not measure execution time or prove
which mechanism dominates. NVIDIA's [Q/DQ placement guidance](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/work-quantized-types.html#q-dq-layer-placement-recommendations)
describes both reduced performance from unfused Q/DQ operations and potential latency improvements from enabling FP16.
Exp6 must verify the behavior on the deployed TensorRT 8.5.2 runtime.

## Proposed comparison

Run the same 28 models in [Exp5's manifest](../exp5/models.csv), with **only two new variants per model**:
56 model/variant combinations and 168 full-test passes. [protocol.json](protocol.json) owns the assignments across the
eight Ubuntu Xavier devices. No INT8-only or FP16-only baseline is rebuilt or rerun.

| New variant                  | ONNX source from Exp5 | TensorRT precision flags |
| ---------------------------- | --------------------- | ------------------------ |
| INT8 PTQ + FP16 (`ptq_fp16`) | Identical `ptq.onnx`  | `--int8 --fp16`          |
| INT8 QAT + FP16 (`qat_fp16`) | Identical `qat.onnx`  | `--int8 --fp16`          |

For each PTQ/QAT pair, the intended build change is enabling `--fp16`. Explicit Q/DQ nodes continue to define
quantization boundaries. Enabling FP16 permits TensorRT to use it where supported; it does **not** force every remaining
operation to FP16 or guarantee a speedup. Inspect the resulting engines and report any remaining FP32 work.

## Measurement plan

Reuse the [Exp5 measurement protocol](../exp5/README.md#data-and-measurement-protocol) and existing benchmark owner.
Build and evaluate only on `ubuntu` through `ubuntu-7`; **exclude `soysan`**. Reuse the prepared ONNX files from
`exp5/runs/exp5-20260914`, verifying their recorded SHA-256 hashes on the destination before building. No new calibration
or training is performed. Anvil runs the controllers and archives the results.

- Keep ONNX hashes, input shape `1x3x640x640`, TensorRT/CUDA versions, workspace limit of 1,024 MiB, and disabled TF32
  unchanged. As in Exp5, each board shares a timing cache across its builds; archive the commands and cache snapshots.
  These are fresh device-local builds, so cache history and selected implementations may differ from Exp5.
- Keep square letterboxing, the one-to-many head, external NMS, confidence 0.001, NMS IoU 0.5, and maximum 300 detections.
  Use the same 4,579 test images, 100 warmup inferences, and three measured full-test passes per variant.
- Reuse the archived Exp5 baselines as requested. Set every board to `MODE_20W_6CORE` (mode 8), verify CPUs `0-5`
  are online, retain automatic clocks, and record clocks, temperatures, and competing workloads. Run one model/variant at
  a time per board. Perform a 32-image smoke evaluation before the three complete measured passes.
- Report synchronized inference latency/FPS separately from preprocessing + inference + postprocessing FPS and full-pass
  wall time. Exclude initialization, engine building, and warmup from inference timing.
- Report mAP@0.5 and mAP@0.5:0.95, including changes from the matching original INT8 engine and FP16 reference. Keep the
  test split for evaluation; do not tune quantization or select checkpoints against its accuracy.
- Save engine hashes, detailed layer inspection, and per-layer timing from a separate profiling run. Compare tensor
  formats, reformats, and time spent in affected operations; keep profiling overhead outside primary timing passes.
- Collect Exp5-style power and memory telemetry. Preserve Exp5 artifacts and store new inputs, engines, logs, and results
  under `exp6/` in the repository and its deployment mirrors, collecting results back here.

## How to interpret the outcome

For each model and quantization method, calculate:

- **Speedup from enabling FP16** = new INT8 + FP16 inference FPS / original INT8 inference FPS.
- **Speed relative to FP16** = new INT8 + FP16 inference FPS / FP16 reference inference FPS.
- **Accuracy change** = new mAP minus reference mAP, reported in percentage points for each reference.

Ratios against Exp5 are historical comparisons, not controlled measurements isolating the precision flag. Evidence of
FP32 operations moving to FP16 and separate layer profiles can help explain the outcome. Report all pass measurements
and accuracy changes, including regressions; pass-to-pass repeatability does not remove cross-run confounders.

If INT8 + FP16 improves on original INT8 but still trails FP16, the build change helps without making INT8 the fastest
option. If it does not improve, conversions, kernel choices, or other bottlenecks may dominate. Exp6 is intended to
measure which outcome occurs; no speedup or accuracy recovery is assumed in advance.

## Device and software differences from Exp5

All eight boards are Xavier NX devices configured to 20 W with six CPU cores online. Keep their installed OS releases:
`ubuntu-1` uses L4T 35.5.0; the other seven use L4T 35.4.1. Exp5 also used `soysan`, which ran L4T 35.6.0.
The target benchmark stack matches Exp5: TensorRT 8.5.2, CUDA 11.4, NVIDIA PyTorch 2.1.0a0+41361538.nv23.06,
torchvision 0.16.0+fbb4cc5, and NumPy 1.23.5. Record actual versions and paths on every device before launch.

Models are redistributed across eight boards, so some comparisons change physical device and L4T release. Thermal
conditions, automatic clocks, available memory, power behavior, timing-cache history, and kernel selection can also
differ from the archived runs. Report the old and new device/software identities alongside every comparison and do not
attribute the entire FPS difference to FP16 fallback. No OS upgrade or Exp7 workspace experiment is part of this run.

## Execution and results

The existing [controller](../exp5/run.py), [benchmark](../exp5/benchmark.py), and
[summarizer](../exp5/summarize.py) are reused. Start one controller per configured device after runtime, dataset, and
privileged telemetry checks; for example:

```bash
python3 -u exp5/run.py exp6-20260918 --protocol exp6/protocol.json --device ubuntu
```

The Ubuntu SSH aliases are defined in [ssh_config](ssh_config). Every board has its own reverse tunnel to Anvil, so no
benchmark traffic passes through `soysan` or another benchmarking board.

Controllers are started only after runtime validation, SHA-256 verification of all 9,158 dataset files, and a live
privileged telemetry check. Each board finishes its setup transfers before benchmarking. Runtime files were copied from `ubuntu`; the remaining
dataset transfers continue directly from Anvil. Launch records and controller logs are retained under the run directory.

Collected results live under `exp6/runs/exp6-20260918/`, including `results.csv`, `status.json`, per-model engine inspection,
smoke checks, per-pass metrics, and raw telemetry. Each model is archived and SHA-256 verified before its temporary device
copy is removed. Device failures preserve remote artifacts for diagnosis. Exp5 baseline files are retained unchanged. The run also stores `exp5-baseline-results.csv` and
`exp5-baseline-provenance.json`; older cases without a per-variant device record explicitly identify their campaign-level
configuration as the available provenance.

## Monitor progress

Select **Exp6 Xavier Progress** in PanePilot for a live table that refreshes every 10 seconds. It queries only the eight
Ubuntu devices in this experiment's protocol and combines their live state with verified Anvil archives. The table shows
the two new variants, smoke checks, build/evaluation/profiling progress, failures, and archive counts. Setup delays and
unavailable devices are reported explicitly. The action only reads state.

For a single refresh in a terminal:

```bash
bash .panepilot/actions/xavier-progress.sh --protocol exp6/protocol.json
```

# Experiment 5: Xavier deployment benchmarks

Paired PTQ/QAT accuracy, FPS, and resource figures, an interactive explorer, and regeneration instructions are in
[`figure_generation/`](figure_generation/README.md). Generated outputs and their input snapshots are stored together.

All **140 model/runtime combinations** completed three full-test passes and were archived by September 18 at 05:58 EDT.
The complete-campaign figures and eight-slide PDF briefing are in `figure_generation/output/exp5-complete-20260918/`.
Energy aggregates are available for 138/140 combinations; YOLO11m PTQ and YOLO12x FP16 have flagged telemetry gaps.

Exp5 distributes its 28 models across three Xavier NX devices. Each device builds and benchmarks one model at a time,
then returns its engines and measurements to Anvil. The **Xavier Model Progress** action reports all three queues.
All 28 models completed Rainbow preparation on September 14 at 15:14 EDT. Xavier benchmarking resumed that evening
from YOLO26s after field testing, with its RTSP service stopped and a fresh telemetry session following the device reboot.
The interrupted snapshot is preserved in `resume-20260914-evening/` within the production run. Verified field copies of
YOLO26n's four engines remain in `~/work/all-yolo-quantized/exp5/field/yolo26n/` on Xavier. A run's `xavier-paused.json`
marker blocks the controller until an explicit resume.
ONNX CUDA, TensorRT FP16, and corrected INT8 PTQ passed 32-image smoke checks; all 275 ONNX profile nodes executed on CUDA.
Rainbow preparation started on GPU 0 and GPU 1 with the YOLO26 family. GPU 0 was released on September 14 at 11:07 EDT;
the remaining preparation completed on GPU 1. Each model's QAT starts from its completed validation calibration. Xavier deployment
remains sequential within each device.

| Device     | Assigned models                        | L4T    | Power mode    |
| ---------- | -------------------------------------- | ------ | ------------- |
| `soysan`   | YOLO26, YOLO26-P2, YOLO11, YOLO12 (17) | 35.6.0 | 20 W, 6 cores |
| `ubuntu`   | YOLOv9 t/s/m/c/e (5)                   | 35.4.1 | 20 W, 6 cores |
| `ubuntu-1` | YOLOv10 n/s/m/b/l/x (6)                | 35.5.0 | 20 W, 6 cores |

`protocol.json` owns these disjoint assignments. Results include a device column and each newly executed model archives
`device.json`, including its actual hardware, runtime versions, boot ID, and power configuration. The operating-system
releases differ, so hardware and software identity must accompany comparisons across families. The two new devices use
the same NVIDIA PyTorch 23.06 release, supported on [JetPack 5.1.x](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform-release-notes/pytorch-jetson-rel.html),
TensorRT 8.5.2, and ONNX Runtime GPU 1.16.0. Engines and timing caches are built and used on their own device.

The active production run is `exp5-20260914`. An earlier INT8 build failed on the attention positional convolution;
exp4 already excludes that layer, and this run applies the same exclusion before QAT and export. Earlier preparation
outputs and the failed build are preserved under `exp5-20260913` and `smoke-20260913`. That smoke run also contains a
one-epoch CPU QAT setup check made while Rainbow GPUs were occupied; it is not a production result. Current preparation
uses CUDA. GPU jobs are ordinary shared-machine jobs; an exclusive-process request was rejected for insufficient
permissions.

## Inputs and scope

Use the **28 Exp2 checkpoints** evaluated by `exp3/runs/exp3-20260911/full/exp2`. That directory contains evaluation
records; the actual weights are in `exp3/inputs/checkpoints/exp2`. `models.csv` records their repository-relative paths,
SHA-256 hashes, file sizes, original training jobs, and corresponding evaluation records. All 28 hashes were verified
against those records. The YOLO26 family finished first. Remaining models now run in ascending checkpoint size within each device, completing
INT8 PTQ and QAT across the queue before returning to ONNX, FP32, and FP16. Verified variants are reused.

Models: YOLOv9 `t/s/m/c/e`, YOLOv10 `n/s/m/b/l/x`, YOLO11 `n/s/m/l/x`, YOLO12 `n/s/m/l/x`, YOLO26 `n/s/m/l/x`, and
YOLO26-P2 `n/s`: 28 architectures and **140 model/runtime combinations**.

| Variant           | Runtime on Xavier | Input                                           |
| ----------------- | ----------------- | ----------------------------------------------- |
| ONNX FP32         | ONNX Runtime CUDA | Full-precision ONNX                             |
| TensorRT FP32     | TensorRT GPU      | Engine built on Xavier from full-precision ONNX |
| TensorRT FP16     | TensorRT GPU      | Engine built on Xavier from full-precision ONNX |
| TensorRT INT8 PTQ | TensorRT GPU      | Validation-calibrated explicit Q/DQ ONNX        |
| TensorRT INT8 QAT | TensorRT GPU      | Validation-fine-tuned explicit Q/DQ ONNX        |

The ONNX row means execution with ONNX Runtime CUDA, with provider placement recorded. CPU fallback must be reported.
INT8 engine layer precision must be inspected and mixed precision disclosed; an INT8 filename is insufficient evidence.

## Relationship to exp4

The reference experiment is
`/anvil/scratch/x-vaggarwal/yolov7Quantization/projects/QAT720_withLLM/exp4`.
Its completed QAT campaign contains 27 models, excluding `yolov9e`. Exp5 includes that model and requires a measured
deployment outcome. Its absence from exp4 does not establish that it is too large for Xavier.

Exp4's production recipe used 512 validation images for PTQ and 10 QAT epochs with AdamW, `lr0=0.0001`, `lrf=0.1`,
cosine learning-rate decay, one warmup epoch, and AMP disabled. Its QAT data YAML mapped training to the original training
split. Exp5 uses **all 1,832 original validation images** for calibration and fine-tuning, as requested; historical
PTQ/QAT checkpoints therefore cannot substitute for fresh exp5 quantization.

The fine-tuning budget is the same 10 epochs, fixed before inspecting test results. Export the
final-epoch quantized checkpoint. Training-set diagnostics may use the reused validation split but are not held-out
accuracy. The 4,579-image test split is reserved for final mAP and throughput measurements.

`prepare.py` reuses this repository's `prepare_qat`, QAT checkpoint serialization, native detection trainer, and ONNX
exporter. Its trainer hook supplies the fully calibrated model, avoiding the native trainer's shorter calibration
default. Export and QAT preparation run on Rainbow in a mirror of this repository. Engine construction and all reported
deployment benchmarks run on Xavier. Anvil is the control and artifact archive host. The two `run_prepare.sh` workers
claim separate models with file locks and run each model's calibration before its QAT; device execution follows the
INT8-first, ascending-checkpoint-size order within each device.

A run's `gpu-<index>.stop` file retires that Rainbow worker before another model starts and prevents its restart. The
September 14 transition used the existing model locks to let both already-running models finish, release GPU 0 first,
and continue the queue on GPU 1. The transition script, event record, and log are preserved in the production run folder.

## Data and measurement protocol

`protocol.json` records the settings and hashes of the existing seed-42 manifests. Validation and test image
names are unique and disjoint. The test set includes all 3,180 background images and 3,596 labeled objects.

- One class, `weed`; fixed input `1x3x640x640`; batch size 1; square letterboxing.
- Preserve exp3's one-to-many head and external NMS policy for every variant, including YOLOv10 and YOLO26. Exp4's
  deployment exporter used a different head policy; do not reuse that export behavior unchanged.
- mAP confidence floor 0.001, NMS IoU 0.5, maximum 300 detections; report mAP@0.5 and mAP@0.5:0.95 on a 0–1 scale.
- Warm up with 100 test-image inferences, excluded from results. Run three complete measured passes over the same 4,579
  test images; save results per pass. Compute accuracy per pass, without merging duplicate predictions across repeats.
- Report synchronized inference latency/FPS separately from preprocessing + inference + postprocessing FPS. Record
  complete-pass wall time and image count as well, so loading overhead is visible. Initialization, compilation, engine
  building, and warmup are outside inference timing. Synthetic `trtexec` input timing cannot replace test-image timing.
- Record exact device model/RAM, JetPack, CUDA, TensorRT, ONNX Runtime, source revisions, model hashes, engine hashes,
  input shape, power mode, clocks, temperatures, and runtime provider/layer precision.

Record unfused reference model GFLOPs per image using a documented profiler, counting a multiply-accumulate as two operations and listing
uncovered operators. Derive **effective GFLOP/s = reference GFLOPs/image × measured inference FPS**. This is an equivalent
model-work rate, not a hardware instruction counter. For INT8, label the corresponding rate effective GOP/s; integer
operations are not floating-point operations. Actual instruction throughput requires separate supported hardware
profiling, which must run outside the primary timing passes.

## Power and resource telemetry

Use timestamped raw `tegrastats` samples at 100 ms intervals with explicit build, warmup, inference, and idle boundaries.
Record a 30-second idle baseline. Capture board input power (actual rail name and units), any available CPU/GPU rails,
system RAM, swap, CPU utilization per core, GPU utilization, memory-controller utilization, clocks, and temperatures.
Record the process RSS high-water mark, including initialization and warmup, separately. Jetson memory is shared by CPU
and GPU; do not label total system RAM as dedicated VRAM.

Summarize mean/peak power in watts, integrate timestamped board power over each measured window for joules, and divide
by processed images for joules/image. Keep engine-build energy separate from inference energy. Report sample coverage
and unavailable sensors explicitly. Never add the total input rail to its component rails.

NVIDIA documents the available memory, utilization, and power fields in the
[tegrastats reference](https://docs.nvidia.com/jetson/archives/r35.4.1/DeveloperGuide/text/AT/JetsonLinuxDevelopmentTools/TegrastatsUtility.html).
This Xavier NX has 6,833 MiB usable shared RAM, JetPack 5.1.4 / L4T R35.6.0, CUDA 11.4, and TensorRT 8.5.2.2. Its existing
power mode is `MODE_20W_6CORE`, with automatic clocks. TensorRT 8.5 uses `--buildOnly`; the newer `--skipInference` flag
is unsupported here. Engine builds use a 1,024 MiB workspace limit and disable TF32. A timing cache is shared across
production builds to reduce repeated tactic searches, with a snapshot archived per successful build. Build time and
energy therefore include the effect of cache reuse; inference timing excludes the build process.

## Parallel devices and verified archive

Stage the required dataset once. Transfer one model at a time, build and benchmark its selected variants serially on Xavier,
then collect its engines, ONNX/checkpoint outputs, logs, metrics, and raw telemetry into `exp5/runs/<run-id>/<model>/`
on Anvil scratch. Each variant keeps its own telemetry and device identity, so later formats can run in a different
boot session. A variant archive receipt records each verified subset; the model archive receipt requires all five variants.
Verify every collected artifact with SHA-256 before deleting that model's temporary files from
Xavier. If collection fails, keep the remote artifacts and stop advancing the model queue. No generated engine remains
on Xavier after its verified archive completes, and no engine is built on Anvil or Rainbow for the Xavier measurements.

The originally active DeepStream workload is the user service `yolo26s-rtsp.service`. Each deployment specifies its
inference service, if present; the runner stops an active service and restores it on exit. Neither new board had a running
inference service. Record and preserve the existing power/clock configuration.

Use one row per model/variant, including failures. Distinguish build OOM, inference OOM, disk exhaustion, unsupported
operators, and incompatible runtimes. Preserve failing commands and logs. Only demonstrated memory failures justify a
"too large under this configuration" result; checkpoint file size alone is not a deployment-capacity measurement.

## SSH setup

The Anvil SSH config contains `soysan-over-purdue-ip`, user `usr`, requested address `192.168.100.164`, and the dedicated
key `~/.ssh/id_ed25519_soysan_anvil`. The public key is installed and key-only login has been verified. No password is
stored in this repository. `connectivity.json` records the failed direct probes and working relay.

Both `192.168.100.164:22` and `100.83.255.62:22` timed out from Anvil and Rainbow. Anvil routes the first address into
its internal `192.168.0.0/16` network. The user confirmed that their computer can SSH to `100.83.255.62` through Tailscale.

Run this on that computer and keep it connected:

```bash
ssh -NT \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -R 127.0.0.1:2225:100.83.255.62:22 \
  x-vaggarwal@login06.anvil.rcac.purdue.edu
```

This exposes the relayed SSH endpoint only on Anvil login06's loopback interface. The alias now uses `127.0.0.1:2225`
with `HostKeyAlias soysan-over-purdue-ip`. The tunnel must remain connected for transfers and control.

The added devices were bootstrapped through this Xavier using Tailscale addresses `100.82.57.52` (`ubuntu`) and
`100.64.106.106` (`ubuntu-1`). Their Anvil aliases are `xavier-ubuntu` and `xavier-ubuntu-1`. Each now maintains its own
outbound reverse tunnel to Anvil, listening only on `127.0.0.1:2226` or `:2227`. Restricted keys permit their designated
forward and disable shell access. This avoids model and dataset transfers through the original board during its power
measurements. The aliases ending in `-via-soysan` retain the original jump route for recovery.

## Runtime and execution

Rainbow uses `exp5/.venv`, inheriting its existing CUDA PyTorch environment, with ModelOpt 0.44.0 and
`ultralytics-thop` 2.1.6. Set `CUDA_VISIBLE_DEVICES` to the selected GPU's UUID before calling `prepare.py`; the script
verifies the actual CUDA device. Source checkpoint hashes are checked before each preparation stage.

Xavier uses `exp5/.venv` with NVIDIA PyTorch 2.1.0a0, torchvision 0.16.0 built from source for SM 7.2, TensorRT 8.5.2,
and NumPy 1.23.5. `exp5/.venv-onnx` shares those packages but overrides NumPy to 1.24.4 for NVIDIA's ONNX Runtime GPU
1.16.0 wheel. The wheel SHA-256 is `44c82c33c41a0670702c67af1f1425002d57f66d5efb971b35e86067d0e971d9`.

`benchmark.py` uses the native detection validator, adding real-image warmup and measurement boundaries through hooks.
Its separate ONNX profiling pass records node placement outside measured passes. TensorRT layer inspection is saved
alongside each engine. `telemetry.py` runs with root privileges because the board power rails are otherwise unavailable;
creating `telemetry.stop` next to its output stops only its own collector. The collector holds a lock on its output for
its lifetime; the controller requires that lock to be held before launching its queue. After an unexpected restart,
preserve the previous raw telemetry and interrupted model, record the new boot identity, and start a fresh collector.

After dataset transfer and smoke validation, start a fresh privileged collector on Xavier for the production run:

```bash
cd /home/usr/work/all-yolo-quantized
sudo -b python3 exp5/telemetry.py exp5/runs/exp5-20260914/telemetry.jsonl
```

Then run the control process on Anvil:

```bash
python3 -u exp5/run.py exp5-20260914
python3 -u exp5/run.py exp5-20260914 --device ubuntu
python3 -u exp5/run.py exp5-20260914 --device ubuntu-1
```

Run each command in a separate persistent process. Each controller takes a lock for its device and filters the manifest
to its assigned families. An archive lock serializes collection and summary updates; remote benchmarks remain parallel.
Every device requires its own privileged telemetry collector and run-level `device.json` before launch. A controller
runs PTQ/QAT for every remaining model before its floating-point queue, then writes `complete-<device>.json`; `complete.json` requires all three queues to finish.

The controller keeps computation remote, collects artifacts after each model, verifies hashes before deleting device
copies, and updates `results.csv` and `status.json` using `summarize.py`. It retries interrupted SSH connections and
transfers. Each worker first runs a CUDA kernel; a device initialization failure stops its controller before model variants
are attempted. A worker that disappears without an exit record also stops its controller for recovery. Inspect failed
stage logs before deliberately retrying them with a new run ID.

In this folder's PanePilot actions, select **Xavier Model Progress**, or run `bash .panepilot/actions/xavier-progress.sh`.
The shell script calls the progress reporter, which prints all 28 models using concurrent live device queries and the Anvil archive
for the production run in `protocol.json`. The table shows each runtime's build/test progress, failures, and archival
status. An unavailable Xavier connection is labeled, with existing Anvil copies shown where available. The action only
reads experiment state.

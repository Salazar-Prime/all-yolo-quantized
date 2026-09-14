# Experiment 5: Xavier deployment benchmarks

Status: YOLO26n's full five-format benchmark is running on Xavier using all 4,579 checksum-verified test images.
ONNX CUDA, TensorRT FP16, and corrected INT8 PTQ passed 32-image smoke checks; all 275 ONNX profile nodes executed on CUDA.
Two Rainbow workers prepare independent models on GPU 0 and GPU 1, starting with the YOLO26 family. Each model's QAT
starts from its completed validation calibration. Xavier deployment remains sequential.

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
against those records. The YOLO26 family runs first, starting with YOLO26n; remaining models follow checkpoint size order.

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
manifest order.

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

## Sequential deployment and archive

Stage the required dataset once. Transfer one model at a time, build and benchmark its variants serially on Xavier,
then collect its engines, ONNX/checkpoint outputs, logs, metrics, and raw telemetry into `exp5/runs/<run-id>/<model>/`
on Anvil scratch. Verify every collected artifact with SHA-256 before deleting that model's temporary files from
Xavier. If collection fails, keep the remote artifacts and stop advancing the model queue. No generated engine remains
on Xavier after its verified archive completes, and no engine is built on Anvil or Rainbow for the Xavier measurements.

The originally active DeepStream workload is the user service `yolo26s-rtsp.service`. It was stopped for setup and device
measurements. The device runner restores it on exit. Record and preserve the existing power/clock configuration.

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
creating `telemetry.stop` next to its output stops only its own collector.

After dataset transfer and smoke validation, start a fresh privileged collector on Xavier for the production run:

```bash
cd /home/usr/work/all-yolo-quantized
sudo -b python3 exp5/telemetry.py exp5/runs/exp5-20260914/telemetry.jsonl
```

Then run the control process on Anvil:

```bash
python3 -u exp5/run.py exp5-20260914
```

The controller keeps computation remote, collects artifacts after each model, verifies hashes before deleting device
copies, and updates `results.csv` and `status.json` using `summarize.py`. It retries interrupted SSH connections and
transfers. Completed stages and locks support resuming collection after a connection interruption. Inspect failed
stage logs before deliberately retrying them with a new run ID.

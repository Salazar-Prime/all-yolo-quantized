# Experiment 7: Effect of TensorRT workspace size on model performance

**Status: planned experiment; documentation only. No Exp7 builds or benchmarks have started.**

## Intent

Test whether increasing TensorRT's workspace limit improves model inference performance on the Ubuntu Xavier NX
devices. Start by comparing **1,024 MiB (1 GiB)**, the limit used in Exp5 and planned for Exp6, against
**2,048 MiB (2 GiB)**. Measure inference speed, memory use, and accuracy for each configuration.

## Why workspace size might matter

TensorRT uses temporary workspace for its layer implementations. The workspace limit affects which implementations
the engine builder can consider. Increasing it may allow a faster implementation that requires more temporary memory.
If the best available implementation already fits within 1 GiB, a larger limit may provide no speedup.

The limit is a ceiling, not a reservation of the entire amount. Runtime workspace depends on the selected
implementations, and weights, activations, and other runtime allocations require additional memory. Xavier shares RAM
between CPU and GPU, so both engine-build and inference memory use matter.

See NVIDIA's [workspace-size guidance](https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/troubleshooting-faq.html).

## Proposed comparison

Start with YOLO26n and YOLO11n. For each model, compare both workspace limits for the same five configurations described
in [Exp6](../exp6/README.md): FP16, INT8 PTQ, INT8 PTQ with FP16 enabled, INT8 QAT, and INT8 QAT with FP16 enabled.

| Workspace limit | TensorRT build option          | Purpose                                                |
| --------------- | ------------------------------ | ------------------------------------------------------ |
| 1 GiB           | `--memPoolSize=workspace:1024` | Reference configuration                                |
| 2 GiB           | `--memPoolSize=workspace:2048` | Test whether additional workspace improves performance |

Within each comparison, change only the workspace limit. Reuse the identical ONNX file, precision flags, quantization
scales, input shape, and TensorRT/CUDA versions. Keep TF32 disabled and use equivalent starting timing-cache snapshots.
No retraining or recalibration is needed. The larger workspace requires a new engine build; changing a runtime setting
on an existing engine does not repeat TensorRT's implementation selection.

Exp6 tests the effect of enabling FP16 for the floating-point portions of INT8 engines. Exp7 separately tests workspace
size, allowing the effect of each change to be assessed.

## Measurement plan

- Use only the eight Ubuntu Xavier devices (`ubuntu` through `ubuntu-7`) on Tailscale. Exclude `soysan`.
  Build and measure both members of each comparison on the same device, with one workload at a time per device.
- Reuse the [Exp5 evaluation protocol](../exp5/README.md#data-and-measurement-protocol): batch size 1, input
  `1x3x640x640`, the same 4,579 test images, 100 warmup inferences, three measured passes, and unchanged preprocessing,
  detection-head policy, and external NMS settings.
- Measure synchronized inference latency/FPS and pipeline FPS separately. Alternate comparison order across passes and
  record power mode, clocks, temperatures, and competing workloads.
- Record peak system RAM, swap use, and build time, with engine building and inference measured separately. Archive
  build commands, input and engine hashes, timing caches, and layer inspection outputs.
- Compare mAP@0.5 and mAP@0.5:0.95. Workspace size does not intentionally change model precision or weights, but changed
  numerical implementations can affect outputs, so accuracy must still be checked.
- Profile layer timings separately from primary timing passes to investigate any speedup. Store Exp7 artifacts under
  `exp7/` in the repository and deployment mirrors, collecting results back to this repository.

## Interpreting results

Calculate **workspace speedup = inference FPS at 2 GiB / inference FPS at 1 GiB** for each model and precision
configuration. Report all pass measurements, accuracy differences, and the accompanying memory cost.

A repeatable gain beyond observed pass variation would support increasing the workspace for that configuration.
Unchanged performance would indicate no demonstrated benefit from the larger limit. Memory failures or slower results
must also be reported; a larger workspace is not assumed to be better, and gains may differ between FP16 and INT8.

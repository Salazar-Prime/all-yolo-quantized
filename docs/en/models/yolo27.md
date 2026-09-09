---
comments: true
description: YOLO27 from Ultralytics pairs a streamlined dual-scale CNN design for compact models with query-based NMS-free detection for large models, and is the first Ultralytics family to surpass 60 mAP on COCO.
keywords: YOLO27, Ultralytics YOLO, object detection, NMS-free, end-to-end detection, small object detection, computer vision, AI, real-time inference
---

# Ultralytics YOLO27

## Overview

[Ultralytics](https://www.ultralytics.com) YOLO27 is a family of real-time vision models built around two
complementary designs: a streamlined CNN architecture for the compact N and S models, and a query-based, NMS-free
architecture for the larger M, L, and X models. Both designs are end-to-end and deploy through the same interface.

Across its five detection scales, YOLO27 reaches **41.6-60.4 mAP on COCO** at **1.8-11.4 ms latency on an NVIDIA
T4** — and up to **61.2 mAP** with YOLO27x at a larger 800-pixel input. YOLO27x is the **first Ultralytics model to
surpass 60 mAP on COCO**, while the compact YOLO27n/s improve on YOLO26n/s accuracy at essentially the same speed.

### YOLO27 vs YOLO26

YOLO26 is compared using its end-to-end (one-to-one head) numbers, matching YOLO27's NMS-free evaluation.

| Scale | YOLO26 mAP<sup>val<br>50-95 (e2e)</sup> | YOLO27 mAP<sup>val<br>50-95</sup> | Δ mAP    | YOLO26 T4 (ms) | YOLO27 T4 (ms) |
| ----- | --------------------------------------- | --------------------------------- | -------- | -------------- | -------------- |
| n     | 40.1                                    | 41.6                              | +1.5     | 1.7            | 1.8            |
| s     | 47.8                                    | 49.2                              | +1.4     | 2.5            | 2.7            |
| m     | 52.5                                    | 55.7                              | +3.2     | 4.7            | 4.7            |
| l     | 54.4                                    | 57.7                              | +3.3     | 6.2            | 6.5            |
| x     | 56.9                                    | 60.4                              | **+3.5** | 11.8           | 11.4           |

!!! example "Quickstart"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo27n.pt")  # load a pretrained YOLO27n model
        results = model("path/to/bus.jpg")  # run inference
        ```

    === "CLI"

        ```bash
        yolo predict model=yolo27n.pt source=path/to/bus.jpg
        ```

## Key Features

- **Dual-scale detection**
  Standard detectors predict objects on three feature maps — fine, medium, and coarse. YOLO27 N and S drop the
  medium one and predict only on a fine map (for small objects) and a coarse map (for large objects), with a fixed
  scaling on the fused features keeping the two scales balanced. This removes a large chunk of detection-head
  computation, making the models faster while training as reliably as the full three-scale design.

- **Stronger small-object detection**
  The early, high-resolution feature stage is widened so it can capture more fine-grained detail. Combined with the
  surviving fine prediction map, this improves localization and regression for small objects — the hardest category
  for compact models.

- **Foreground alignment supervision**
  During training, an extra lightweight branch learns to tell "object" from "background" at every location. It is
  designed to close the gap between the denser one-to-many supervision used during training and the one-to-one head
  that produces the final predictions — cutting that accuracy gap from 0.9/0.8 mAP on YOLO26n/s to just 0.4 mAP on
  YOLO27n/s, so the deployed one-to-one head keeps nearly all of the training-time accuracy. The branch is used only
  during training and is removed for inference and export, so it costs nothing at deployment.

- **Query-based detection without NMS**
  The larger models replace dense prediction with a transformer decoder that refines a fixed set of object queries
  and directly outputs the final detections — no non-maximum suppression post-processing needed. YOLO27m and YOLO27l
  pair this decoder with the proven YOLO26-style convolutional backbone, while YOLO27x keeps the same FPN/PAN neck
  and swaps in an UltraViT backbone that uses self-attention in its deepest stage to capture global context.

- **One simple interface**
  Both architectures are used through the same `YOLO` class. The right training, validation, prediction, and export
  pipeline is selected automatically from the model, so code written for one YOLO27 scale works unchanged for the
  others.

## Which YOLO27 Should I Use?

- **YOLO27n / YOLO27s** — edge devices, drones, and real-time video: the fastest models in the family, with improved
  small-object detection from the dual-scale design.
- **YOLO27m / YOLO27l** — the accuracy-speed sweet spot on GPUs: YOLO27m alone improves on YOLO26m by 3.2 mAP at
  the same T4 latency, making it the default choice for production GPU deployment.
- **YOLO27x** — accuracy-critical applications: the first Ultralytics model above 60 mAP on COCO, reaching 61.2 mAP
  at a larger input size while staying real-time on GPU.

---

## Supported Tasks and Modes

YOLO27 supports the following tasks across its five model scales, all with training, validation, inference, and
export support:

| Model        | Filenames                                                                                      | Task                                          | Training | Validation | Inference | Export |
| ------------ | ---------------------------------------------------------------------------------------------- | --------------------------------------------- | -------- | ---------- | --------- | ------ |
| YOLO27       | `yolo27n.pt` `yolo27s.pt` `yolo27m.pt` `yolo27l.pt` `yolo27x.pt`                               | [Detection](../tasks/detect.md)               | ✅       | ✅         | ✅        | ✅     |
| YOLO27-seg   | `yolo27n-seg.pt` `yolo27s-seg.pt` `yolo27m-seg.pt` `yolo27l-seg.pt` `yolo27x-seg.pt`           | [Instance Segmentation](../tasks/segment.md)  | ✅       | ✅         | ✅        | ✅     |
| YOLO27-sem   | `yolo27n-sem.pt` `yolo27s-sem.pt` `yolo27m-sem.pt` `yolo27l-sem.pt` `yolo27x-sem.pt`           | [Semantic Segmentation](../tasks/semantic.md) | ✅       | ✅         | ✅        | ✅     |
| YOLO27-depth | `yolo27n-depth.pt` `yolo27s-depth.pt` `yolo27m-depth.pt` `yolo27l-depth.pt` `yolo27x-depth.pt` | [Depth Estimation](../tasks/depth.md)         | ✅       | ✅         | ✅        | ✅     |
| YOLO27-cls   | `yolo27n-cls.pt` `yolo27s-cls.pt` `yolo27m-cls.pt` `yolo27l-cls.pt` `yolo27x-cls.pt`           | [Classification](../tasks/classify.md)        | ✅       | ✅         | ✅        | ✅     |
| YOLO27-pose  | `yolo27n-pose.pt` `yolo27s-pose.pt` `yolo27m-pose.pt` `yolo27l-pose.pt` `yolo27x-pose.pt`      | [Pose/Keypoints](../tasks/pose.md)            | ✅       | ✅         | ✅        | ✅     |
| YOLO27-obb   | `yolo27n-obb.pt` `yolo27s-obb.pt` `yolo27m-obb.pt` `yolo27l-obb.pt` `yolo27x-obb.pt`           | [Oriented Detection](../tasks/obb.md)         | ✅       | ✅         | ✅        | ✅     |

!!! note "Two architecture paths"

    YOLO27 detection uses two designs under one interface: the N and S scales use the streamlined CNN architecture,
    while the M, L, and X scales use the query-based NMS-free architecture. All other tasks use the CNN architecture.

---

## Performance Metrics

Detection accuracy is reported on the COCO validation set. Inference speed is measured on an NVIDIA RTX PRO 6000
(TensorRT 11, FP16) for GPU and an AMD EPYC 9655 (ONNX Runtime, FP32) for CPU. Accuracy numbers can be reproduced
with `yolo val model=yolo27n.pt data=coco.yaml`.

=== "Detection (COCO)"

    YOLO27x is additionally reported at an 800-pixel input.

    | Model   | Size<br><sup>(pixels)</sup> | mAP<sup>val<br>50-95</sup> | CPU ONNX<br><sup>(ms)</sup> | TensorRT<br><sup>(ms)</sup> | Params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
    | ------- | --------------------------- | -------------------------- | --------------------------- | -------------------------- | ------------------------ | ----------------------- |
    | YOLO27n | 640                         | 41.6                       | 16.1                        | 0.62                       | 3.1                      | 8.2                     |
    | YOLO27s | 640                         | 49.2                       | 33.2                        | 0.79                       | 12.4                     | 31.4                    |
    | YOLO27m | 640                         | 55.7                       | 67.2                        | 1.39                       | 22.8                     | 65.5                    |
    | YOLO27l | 640                         | 57.7                       | 93.8                        | 2.00                       | 30.4                     | 86.1                    |
    | YOLO27x | 640                         | 60.4                       | 149.6                       | 2.32                       | 72.3                     | 166.2                   |
    | YOLO27x | 800                         | **61.2**                   | 213.2                       | 2.90                       | 72.3                     | 254.8                   |

=== "Segmentation (COCO)"

    Measured at a 640-pixel input.

    | Model       | Size<br><sup>(pixels)</sup> | CPU ONNX<br><sup>(ms)</sup> | TensorRT<br><sup>(ms)</sup> | Params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
    | ----------- | --------------------------- | --------------------------- | -------------------------- | ------------------------ | ----------------------- |
    | YOLO27n-seg | 640                         | 22.0                        | 0.73                       | 3.4                      | 12.7                    |
    | YOLO27s-seg | 640                         | 45.8                        | 0.96                       | 13.0                     | 47.9                    |
    | YOLO27m-seg | 640                         | 109.3                       | 1.46                       | 29.5                     | 156.5                   |
    | YOLO27l-seg | 640                         | 125.3                       | 1.91                       | 34.2                     | 178.4                   |
    | YOLO27x-seg | 640                         | 248.3                       | 2.86                       | 76.7                     | 399.0                   |

=== "Semantic Segmentation (Cityscapes)"

    Measured at a 1024 × 2048-pixel input.

    | Model       | Size<br><sup>(pixels)</sup> | mIoU<sup>val</sup> | CPU ONNX<br><sup>(ms)</sup> | TensorRT<br><sup>(ms)</sup> | Params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
    | ----------- | --------------------------- | ------------------ | --------------------------- | -------------------------- | ------------------------ | ----------------------- |
    | YOLO27n-sem | 1024 × 2048                 | 78.8               | 95.6                        | 0.89                       | 2.0                      | 34.6                    |
    | YOLO27s-sem | 1024 × 2048                 | 81.2               | 169.6                       | 1.43                       | 7.8                      | 132.7                   |
    | YOLO27m-sem | 1024 × 2048                 | 82.4               | 336.9                       | 2.71                       | 16.1                     | 387.7                   |
    | YOLO27l-sem | 1024 × 2048                 | 83.6               | 418.0                       | 3.46                       | 20.0                     | 488.7                   |
    | YOLO27x-sem | 1024 × 2048                 | 83.8               | 787.1                       | 6.43                       | 44.8                     | 1090.5                  |

=== "Depth Estimation (NYU Depth V2)"

    Measured at a 768-pixel input with latency on an NVIDIA RTX PRO 6000 (TensorRT) and AMD EPYC 9655 CPU (ONNX).

    | Model         | Params | GFLOPs | CPU ONNX | TensorRT | NYU* δ1 | KITTI-580* δ1 | bench mean |
    | ------------- | ------ | ------ | -------- | -------- | ------- | ------------- | ---------- |
    | YOLO27n-depth | 5.43 M | 49.6   | 47.2 ms  | 0.79 ms  | 0.8314  | 0.8256        | 0.7238     |
    | YOLO27s-depth | 13.06 M | 77.8  | 72.1 ms  | 0.97 ms  | 0.8682  | 0.7835        | 0.7454     |
    | YOLO27m-depth | 23.42 M | 144.4 | 120.2 ms | 1.34 ms  | 0.8652  | 0.7746        | 0.7476     |
    | YOLO27l-depth | 28.09 M | 176.0 | 147.0 ms | 1.83 ms  | 0.8742  | 0.7862        | 0.7616     |
    | YOLO27x-depth | 59.38 M | 343.3 | 253.7 ms | 2.65 ms  | 0.8711  | 0.8041        | 0.7527     |

=== "Classification (ImageNet)"

    Measured at a 224-pixel input.

    | Model       | Size<br><sup>(pixels)</sup> | CPU ONNX<br><sup>(ms)</sup> | TensorRT<br><sup>(ms)</sup> | Params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
    | ----------- | --------------------------- | --------------------------- | -------------------------- | ------------------------ | ----------------------- |
    | YOLO27n-cls | 224                         | 1.9                         | 0.28                       | 2.9                      | 0.5                     |
    | YOLO27s-cls | 224                         | 3.1                         | 0.31                       | 6.9                      | 1.8                     |
    | YOLO27m-cls | 224                         | 6.0                         | 0.39                       | 12.4                     | 6.1                     |
    | YOLO27l-cls | 224                         | 8.9                         | 0.59                       | 15.5                     | 8.3                     |
    | YOLO27x-cls | 224                         | 16.7                        | 0.69                       | 32.8                     | 18.6                    |

=== "Pose (COCO)"

    Measured at a 640-pixel input.

    | Model        | Size<br><sup>(pixels)</sup> | mAP<sup>pose<br>50-95</sup> | mAP<sup>pose<br>50</sup> | CPU ONNX<br><sup>(ms)</sup> | TensorRT<br><sup>(ms)</sup> | Params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
    | ------------ | --------------------------- | --------------------------- | ------------------------ | --------------------------- | -------------------------- | ------------------------ | ----------------------- |
    | YOLO27n-pose | 640                         | **58.0**                    | **84.1**                 | 19.6                        | 0.67                       | 3.7                      | 12.1                    |
    | YOLO27s-pose | 640                         | **64.3**                    | **87.0**                 | 36.6                        | 0.80                       | 12.6                     | 37.3                    |
    | YOLO27m-pose | 640                         | **69.0**                    | **89.9**                 | 76.6                        | 1.29                       | 25.4                     | 102.6                   |
    | YOLO27l-pose | 640                         | **70.7**                    | **90.3**                 | 95.2                        | 1.74                       | 30.1                     | 124.5                   |
    | YOLO27x-pose | 640                         | **72.2**                    | **91.3**                 | 181.1                       | 2.45                       | 67.6                     | 278.8                   |

=== "OBB (DOTAv1)"

    Measured at a 1024-pixel input on the DOTAv1 test set.

    | Model       | Size<br><sup>(pixels)</sup> | mAP<sup>test<br>50-95</sup> | mAP<sup>test<br>50</sup> | CPU ONNX<br><sup>(ms)</sup> | TensorRT<br><sup>(ms)</sup> | Params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
    | ----------- | --------------------------- | --------------------------- | ------------------------ | --------------------------- | -------------------------- | ------------------------ | ----------------------- |
    | YOLO27n-obb | 1024                        | **54.0**                    | **80.3**                 | 42.6                        | 0.83                       | 3.3                      | 22.9                    |
    | YOLO27s-obb | 1024                        | **55.8**                    | **81.6**                 | 88.2                        | 1.30                       | 11.9                     | 88.3                    |
    | YOLO27m-obb | 1024                        | **55.9**                    | **82.6**                 | 188.0                       | 1.92                       | 25.4                     | 263.7                   |
    | YOLO27l-obb | 1024                        | **56.5**                    | **82.7**                 | 227.9                       | 2.48                       | 30.0                     | 320.3                   |
    | YOLO27x-obb | 1024                        | **57.2**                    | **82.7**                 | 455.8                       | 4.33                       | 74.5                     | 715.9                   |

---

## Usage Examples

This section provides simple YOLO27 training and inference examples. For full documentation on these and other
[modes](../modes/index.md), see the [Predict](../modes/predict.md), [Train](../modes/train.md),
[Val](../modes/val.md), and [Export](../modes/export.md) docs pages.

Note that the example below is for YOLO27 [Detect](../tasks/detect.md) models for [object
detection](https://www.ultralytics.com/glossary/object-detection). For additional supported tasks, see the
[Segment](../tasks/segment.md) and [Classify](../tasks/classify.md) docs.

!!! example

    === "Python"

        [PyTorch](https://www.ultralytics.com/glossary/pytorch) pretrained `*.pt` models as well as configuration
        `*.yaml` files can be passed to the `YOLO()` class to create a model instance in Python:

        ```python
        from ultralytics import YOLO

        # Load a COCO-pretrained YOLO27n model
        model = YOLO("yolo27n.pt")

        # Run inference with the YOLO27n model on the 'bus.jpg' image
        results = model("path/to/bus.jpg")

        # Train the model on the COCO8 example dataset for 100 epochs
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        CLI commands are available to directly run the models:

        ```bash
        # Load a COCO-pretrained YOLO27n model and run inference on the 'bus.jpg' image
        yolo predict model=yolo27n.pt source=path/to/bus.jpg

        # Load a COCO-pretrained YOLO27n model and train it on the COCO8 example dataset for 100 epochs
        yolo train model=yolo27n.pt data=coco8.yaml epochs=100 imgsz=640
        ```

YOLO27 code, models, and documentation are available in the [Ultralytics GitHub
repository](https://github.com/ultralytics/ultralytics) and [Ultralytics Docs](../index.md) under
[AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) and
[Enterprise](https://www.ultralytics.com/license) licenses.

---

## FAQ

### What are the key improvements in YOLO27?

- **Dual-scale detection (N/S)**: drops the medium prediction map for a faster head with competitive accuracy
- **Stronger small-object detection (N/S)**: a widened early feature stage improves small-object localization
- **Foreground alignment supervision (N/S)**: cuts the one-to-many vs one-to-one accuracy gap from 0.9/0.8 mAP on
  YOLO26n/s to 0.4 mAP on YOLO27n/s, at zero inference cost
- **Query-based NMS-free detection (M/L/X)**: a transformer decoder outputs final detections directly
- **One simple interface**: both architectures run through the same `YOLO` class

### Should I upgrade from YOLO26?

Yes, for most use cases. YOLO27 improves end-to-end accuracy at every scale: +1.5/+1.4 mAP for the compact n/s
models at essentially the same speed, and +3.2/+3.3/+3.5 mAP for m/l/x. YOLO27x is the first Ultralytics model to
surpass 60 mAP on COCO.

### Is YOLO27 a drop-in replacement for YOLO26?

Yes. All YOLO27 models use the same `YOLO` class and the same train/val/predict/export API as YOLO26 — the correct
pipeline (CNN or query-based) is selected automatically from the model. Swapping `yolo26n.pt` for `yolo27n.pt` is
the only change required.

### Why do YOLO27 N and S predict on only two scales?

Most detectors predict on three feature maps at different resolutions. YOLO27 N and S keep the fine map that small
objects depend on and the coarse map that large objects need, and skip the medium one. This cuts a significant share
of detection-head computation, and the training improvements above keep the accuracy-latency tradeoff competitive.

### What makes the YOLO27x result notable?

YOLO27x is the first Ultralytics model to surpass 60 mAP on COCO, reaching 60.4 mAP at a 640-pixel input (11.4 ms on
an NVIDIA T4) and 61.2 mAP at an 800-pixel input (16.8 ms). It combines the UltraViT backbone, multi-scale feature
fusion, and a query-based detector that produces final detections directly, without NMS.

### How do I get started with YOLO27?

YOLO27 models are available through the `ultralytics` package. Install or update the package and load a model:

```python
from ultralytics import YOLO

# Load a pretrained YOLO27 nano model
model = YOLO("yolo27n.pt")

# Run inference on an image
results = model("image.jpg")
```

See the [Usage Examples](#usage-examples) section for training, validation, and export instructions.

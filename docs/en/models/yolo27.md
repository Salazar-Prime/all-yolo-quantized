---
comments: true
description: YOLO27 from Ultralytics pairs a streamlined dual-scale CNN design for compact models with query-based NMS-free detection for large models, and is the first Ultralytics family to surpass 60 mAP on COCO.
keywords: YOLO27, Ultralytics YOLO, object detection, NMS-free, end-to-end detection, small object detection, computer vision, AI, real-time inference
---

# Ultralytics YOLO27

## Overview

[Ultralytics](https://www.ultralytics.com) YOLO27 is a family of real-time vision models built around two
complementary designs: a streamlined CNN architecture for the compact N and S models, and a query-based, NMS-free
architecture for the larger M and L models. Both designs are end-to-end and deploy through the same interface.

Across its four detection scales, YOLO27 reaches **41.6-60.4 mAP on COCO** at **0.62-2.32 ms latency on an NVIDIA
RTX PRO 6000** — and up to **61.2 mAP** with YOLO27l at a larger 800-pixel input. YOLO27l is the **first Ultralytics
model to surpass 60 mAP on COCO**, while the compact YOLO27n/s improve on YOLO26n/s accuracy at essentially the same
speed.

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
  and directly outputs the final detections — no non-maximum suppression post-processing needed. YOLO27m pairs this
  decoder with the proven YOLO26-style convolutional backbone, while YOLO27l keeps the same FPN/PAN neck and swaps
  in an UltraViT backbone that uses self-attention in its deepest stage to capture global context.

- **One simple interface**
  Both architectures are used through the same `YOLO` class. The right training, validation, prediction, and export
  pipeline is selected automatically from the model, so code written for one YOLO27 scale works unchanged for the
  others.

## Which YOLO27 Should I Use?

- **YOLO27n / YOLO27s** — edge devices, drones, and real-time video: the fastest models in the family, with improved
  small-object detection from the dual-scale design.
- **YOLO27m** — the accuracy-speed sweet spot on GPUs: improves on YOLO26m by 3.2 mAP at the same latency,
  making it the default choice for production GPU deployment.
- **YOLO27l** — accuracy-critical applications: the first Ultralytics model above 60 mAP on COCO, reaching 61.2 mAP
  at a larger input size while staying real-time on GPU.

---

## Supported Tasks and Modes

YOLO27 supports the following tasks across its four model scales, all with training, validation, inference, and
export support:

| Model        | Filenames                                                                   | Task                                          | Training | Validation | Inference | Export |
| ------------ | --------------------------------------------------------------------------- | --------------------------------------------- | -------- | ---------- | --------- | ------ |
| YOLO27       | `yolo27n.pt` `yolo27s.pt` `yolo27m.pt` `yolo27l.pt`                         | [Detection](../tasks/detect.md)               | ✅       | ✅         | ✅        | ✅     |
| YOLO27-seg   | `yolo27n-seg.pt` `yolo27s-seg.pt` `yolo27m-seg.pt` `yolo27l-seg.pt`         | [Instance Segmentation](../tasks/segment.md)  | ✅       | ✅         | ✅        | ✅     |
| YOLO27-sem   | `yolo27n-sem.pt` `yolo27s-sem.pt` `yolo27m-sem.pt` `yolo27l-sem.pt`         | [Semantic Segmentation](../tasks/semantic.md) | ✅       | ✅         | ✅        | ✅     |
| YOLO27-depth | `yolo27n-depth.pt` `yolo27s-depth.pt` `yolo27m-depth.pt` `yolo27l-depth.pt` | [Depth Estimation](../tasks/depth.md)         | ✅       | ✅         | ✅        | ✅     |
| YOLO27-cls   | `yolo27n-cls.pt` `yolo27s-cls.pt` `yolo27m-cls.pt` `yolo27l-cls.pt`         | [Classification](../tasks/classify.md)        | ✅       | ✅         | ✅        | ✅     |
| YOLO27-pose  | `yolo27n-pose.pt` `yolo27s-pose.pt` `yolo27m-pose.pt` `yolo27l-pose.pt`     | [Pose/Keypoints](../tasks/pose.md)            | ✅       | ✅         | ✅        | ✅     |
| YOLO27-obb   | `yolo27n-obb.pt` `yolo27s-obb.pt` `yolo27m-obb.pt` `yolo27l-obb.pt`         | [Oriented Detection](../tasks/obb.md)         | ✅       | ✅         | ✅        | ✅     |

!!! note "Two architecture paths"

    YOLO27 detection uses two designs under one interface: the N and S scales use the streamlined CNN architecture,
    while the M and L scales use the query-based NMS-free architecture. All other tasks use the CNN architecture.

---

## Performance Metrics

Detection accuracy is reported on the COCO validation set. Inference speed is measured on an NVIDIA RTX PRO 6000
(TensorRT 11, FP16) for GPU and an AMD EPYC 9655 (ONNX Runtime, FP32) for CPU. Accuracy numbers can be reproduced
with `yolo val model=yolo27n.pt data=coco.yaml`.

!!! tip "Performance"

    === "Detection (COCO)"

        See [Detection Docs](../tasks/detect.md) for usage examples with these models trained on [COCO](../datasets/detect/coco.md), which include 80 pretrained classes. YOLO27l is additionally reported at an 800-pixel input.

        --8<-- "docs/macros/yolo-det-perf.md"

    === "Segmentation (COCO)"

        See [Segmentation Docs](../tasks/segment.md) for usage examples with these models trained on [COCO](../datasets/segment/coco.md), which include 80 pretrained classes.

        --8<-- "docs/macros/yolo-seg-perf.md"

    === "Semantic Segmentation (Cityscapes)"

        See [Semantic Segmentation Docs](../tasks/semantic.md) for usage examples with these models trained on [Cityscapes](../datasets/semantic/cityscapes.md), which include 19 pretrained classes.

        --8<-- "docs/macros/yolo-semantic-perf.md"

    === "Depth Estimation (NYU Depth V2)"

        See [Depth Estimation Docs](../tasks/depth.md) for usage examples with these models pretrained on a broad multi-dataset mix and evaluated on [NYU Depth V2](../datasets/depth/nyu-depth-v2.md).

        --8<-- "docs/macros/yolo-depth-perf.md"

    === "Classification (ImageNet)"

        See [Classification Docs](../tasks/classify.md) for usage examples with these models trained on [ImageNet](../datasets/classify/imagenet.md), which include 1000 pretrained classes.

        --8<-- "docs/macros/yolo-cls-perf.md"

    === "Pose (COCO)"

        See [Pose Estimation Docs](../tasks/pose.md) for usage examples with these models trained on [COCO](../datasets/pose/coco.md), which include 1 pretrained class, 'person'.

        --8<-- "docs/macros/yolo-pose-perf.md"

    === "OBB (DOTAv1)"

        See [Oriented Detection Docs](../tasks/obb.md) for usage examples with these models trained on [DOTAv1](../datasets/obb/dota-v2.md#dota-v10), which include 15 pretrained classes.

        --8<-- "docs/macros/yolo-obb-perf.md"

_Params and FLOPs values are for the fused model after Conv/BatchNorm folding and removal of the unused detection branch. Speed measurements select the NMS-free head with `nms=False`. Pretrained checkpoints retain the full training architecture and may show higher counts._

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
- **Query-based NMS-free detection (M/L)**: a transformer decoder outputs final detections directly
- **One simple interface**: both architectures run through the same `YOLO` class

### Should I upgrade from YOLO26?

Yes, for most use cases. YOLO27 improves end-to-end accuracy at every scale: +1.5/+1.4 mAP for the compact n/s
models at essentially the same speed, +3.2 mAP for m, and +3.5 mAP for l over the largest YOLO26 scale. YOLO27l is
the first Ultralytics model to surpass 60 mAP on COCO.

### Is YOLO27 a drop-in replacement for YOLO26?

Yes. All YOLO27 models use the same `YOLO` class and the same train/val/predict/export API as YOLO26 — the correct
pipeline (CNN or query-based) is selected automatically from the model. Swapping `yolo26n.pt` for `yolo27n.pt` is
the only change required.

### Why do YOLO27 N and S predict on only two scales?

Most detectors predict on three feature maps at different resolutions. YOLO27 N and S keep the fine map that small
objects depend on and the coarse map that large objects need, and skip the medium one. This cuts a significant share
of detection-head computation, and the training improvements above keep the accuracy-latency tradeoff competitive.

### What makes the YOLO27l result notable?

YOLO27l is the first Ultralytics model to surpass 60 mAP on COCO, reaching 60.4 mAP at a 640-pixel input (2.3 ms on
an NVIDIA RTX PRO 6000) and 61.2 mAP at an 800-pixel input (2.9 ms). It combines the UltraViT backbone, multi-scale
feature fusion, and a query-based detector that produces final detections directly, without NMS.

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

# QAT720 Exp1 manifest workflow

This example preserves the manifest adapter used by the QAT720 Exp2 and Exp2.5 Azure experiments. It is based on
Ultralytics commit `5b522d606abb81d6e51f3bf0008a08c8511686c2` and keeps the core Ultralytics package unchanged.

The workflow accepts the combined `datasetMetadata.json` produced by Exp1, creates deterministic train/validation/test
splits, writes native YOLO labels from the nested object records, and symlinks images into a standard Ultralytics
dataset layout.

## Supported Exp1 schema

Each image record must contain `filePath` or `fileName` and an `objects` list. Each object must contain `classId`,
`xCenter`, `yCenter`, `boxWidth`, and `boxHeight`. The optional `cocoSizeCategory` field enables small, medium, or large
object filtering. Other Exp1 metadata fields are retained in source manifests but ignored during YOLO label generation.

Use the combined `datasetMetadata.json` to reproduce the experiments. `imageMetadata.json` has no objects and cannot be
used for training. Although `objectMetadata.json` is structurally valid, it omits background images and therefore
changes deterministic split membership.

## Install

From the repository root:

```bash
python -m pip install -e .
```

## Generate deterministic JSON manifests

If only JPEG images and YOLO text labels are available, first rebuild the combined Exp1 annotation file:

```bash
python examples/QAT720-Exp1-Manifest/build_exp1_manifest.py \
    --dataset-root /path/to/day0
```

This includes background images, calculates pixel boxes and COCO size categories, and writes
`/path/to/day0/datasetMetadata.json`. Object UUIDs are deterministic so regenerating unchanged data produces stable
metadata.

Then create the deterministic split manifests:

```bash
python examples/QAT720-Exp1-Manifest/splitDatasetManifests.py 42 \
    --datasetPath /path/to/day0 \
    --ratios 30 20 50
```

## Generate a native Ultralytics dataset

The generated dataset uses image symlinks, so it does not duplicate the image payload.

```bash
python examples/QAT720-Exp1-Manifest/generate_dataset.py \
    --manifest /path/to/day0/datasetMetadata.json \
    --dataset-root /path/to/day0 \
    --class-names weed \
    --split-seed 42 \
    --split-ratios 30 20 50 \
    --prepared-dir /path/to/prepared/seed42-all
```

Add `--skip-empty-images` to reproduce Exp2.5. Without that flag, background-only images are retained as in Exp2.

## Train and evaluate

`train.py` evaluates the best saved checkpoint on both validation and held-out test data after training.

```bash
python examples/QAT720-Exp1-Manifest/train.py \
    --manifest /path/to/day0/datasetMetadata.json \
    --dataset-root /path/to/day0 \
    --class-names weed \
    --split-seed 42 \
    --split-ratios 30 20 50 \
    --model yolo26n.pt \
    --epochs 100 \
    --device 0
```

Use `--prepare-only` to validate and generate data without loading a model or starting training.

Evaluate an existing checkpoint:

```bash
python examples/QAT720-Exp1-Manifest/test.py \
    --manifest /path/to/day0/datasetMetadata.json \
    --dataset-root /path/to/day0 \
    --class-names weed \
    --model /path/to/best.pt \
    --splits val test \
    --device 0
```

Run detection on the held-out split:

```bash
python examples/QAT720-Exp1-Manifest/detect.py \
    --manifest /path/to/day0/datasetMetadata.json \
    --dataset-root /path/to/day0 \
    --class-names weed \
    --model /path/to/best.pt \
    --split test \
    --device 0
```

## Recovered experiment behavior

- Exp2 retained background images: 2,748 train, 1,832 validation, and 4,579 test images.
- Exp2.5 used `--skip-empty-images`: 843 train, 577 validation, and 1,399 test images.
- Both selections contain 1,629 train, 1,422 validation, and 3,596 test objects.

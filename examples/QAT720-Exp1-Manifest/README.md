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
UUID coverage additionally requires a nonempty `objectUuid` that is unique within each evaluated split.

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

## GT instance coverage

Add `--gt-coverage` to `test.py` or `train.py` to report which individual ground-truth objects were detected in each final
validation/test evaluation. Precision, recall, and mAP still use the standard validator. Coverage reuses its predictions
in the same inference pass. It is optional so manifests without object UUIDs remain usable.

```bash
python examples/QAT720-Exp1-Manifest/test.py \
    --manifest /path/to/day0/datasetMetadata.json \
    --dataset-root /path/to/day0 \
    --class-names weed \
    --model /path/to/best.pt \
    --splits val test \
    --gt-coverage --gt-conf 0.25 --gt-iou 0.5 \
    --device 0
```

Matching is performed independently for every image:

1. Keep predictions with confidence **at least** `--gt-conf` (default `0.25`).
2. Consider pairs with the **same class** and IoU **strictly greater than** `--gt-iou` (default `0.5`). This is a matching
   threshold, independent of the validator's NMS IoU threshold.
3. Sort eligible pairs by descending IoU, breaking ties by descending prediction confidence, then manifest object order
   and prediction order. Accept a pair only if neither its GT nor its prediction has already been matched.
4. Discard remaining predictions that have an eligible GT as duplicate conflicts. Count remaining predictions with no
   eligible GT as extra boxes. Predictions below the confidence cutoff are neither duplicates nor extras.

A prediction can detect only one GT, and each GT counts once. Matching is greedy, so it does not guarantee the maximum
possible number of matches. For example, three eligible predictions around one GT produce one detected GT and two
discarded duplicates, with zero extra boxes. A confident prediction on an included background image is an extra box.

Each split's result directory contains `gt_coverage.json` with:

- `gtCoveragePercent`: `100 * gtDetected / gtTotal`, aggregated over all selected objects, not averaged over images.
- `gtTotal`, `gtDetected`, `extraPredictions`, and `discardedDuplicatePredictions` counts.
- `missedObjectUuids`: all unmatched GT UUIDs for that split.
- `images`: per-image detected and missed UUIDs, discarded duplicate counts, and extra prediction records containing
  `bboxXYXY` in original-image pixels, `classId`, and `confidence`.
- The evaluated model, split, and confidence/IoU cutoffs.

The scalar values are also included under `gt/` in the final JSON printed by `test.py` and in the final metrics logged to
W&B by `train.py`. Training-epoch validation and checkpoint selection continue to use the standard metrics.

Coverage uses the exact selected split and object-size filter, including background images unless excluded by the dataset
options. It reads the original manifest objects so distinct UUIDs with identical boxes are still separate GT instances.
If the validation loader skips a prepared image, evaluation fails instead of publishing incomplete coverage. The report
is based on predictions after normal validator postprocessing, including NMS where applicable and its detection limit.

## Recovered experiment behavior

- Exp2 retained background images: 2,748 train, 1,832 validation, and 4,579 test images.
- Exp2.5 used `--skip-empty-images`: 843 train, 577 validation, and 1,399 test images.
- Both selections contain 1,629 train, 1,422 validation, and 3,596 test objects.

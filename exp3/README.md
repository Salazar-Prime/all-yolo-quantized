# Experiment 3: instance coverage beside mAP@0.5

Evaluate the 28 best checkpoints from Exp2 and the 28 best checkpoints from Exp2.5 on the **same full held-out test set**.
The model inventory is in `checkpoints.csv`; each entry records the original training job and expected checkpoint SHA-256.
This experiment performs evaluation only.

The September 11, 2026 run completed **56/56 checkpoints** on Rainbow and passed the full UUID audit.
See [results and all-model comparison](RESULTS.md), [CSV](results.csv), and [verification](verification.json).

All source, inputs, environments, and collected outputs stay under this `all-yolo-quantized` repository. Execution uses a
mirror at `/home/varun/work/all-yolo-quantized` on **Rainbow**, physical GPU **1**. No experiment runs on Anvil.

## Protocol

- Original seed-42 manifests, preserving the original object UUIDs.
- Test set: **4,579 images**, **3,596 GT objects**, including **3,180 background images**.
- Single class: `weed`; image size 640; batch size 16; eight data-loader workers; FP32 validation.
- GT matching: confidence **>= 0.25**, IoU **> 0.5**, same class, greedy highest-IoU-first, one prediction per GT.
- Remaining eligible predictions that conflict over a GT are discarded duplicates, not extra boxes.
- Extra boxes have no eligible GT match. Every missed GT UUID is retained in the per-model report.
- mAP@0.5 is recomputed in the same validator pass, using its normal low-confidence floor and external NMS defaults.
  It is not copied from the earlier experiments, whose evaluation image selections differed.

The current fork's default `nms=None` selects the one-to-many head with external NMS for dual-head YOLO26 and YOLOv10
checkpoints. The archived training source preserved their native end-to-end mode by default. This inference-head
difference means the new YOLO26/YOLOv10 mAP values are not exact reproductions of the archived scores. Every row in this
experiment uses the same current-fork validation policy, with NMS IoU 0.7, at most 300 detections per image, and confidence
floor 0.001 for mAP. The separate GT metric then applies its 0.25 confidence cutoff and strict 0.5 matching IoU cutoff.

## Files

```text
exp3/
  checkpoints.csv                 56 checkpoints and their expected hashes
  dataset.json                    portable references to the three manifests
  run.py                          matrix evaluation using the existing coverage validator
  run_rainbow.sh                  Rainbow GPU 1 launcher, logs, PID, and exit status
  RESULTS.md                      completed findings, protocol, and all-model table
  results.csv                     versioned metrics for every checkpoint
  comparison.png / .svg           versioned comparison figures
  verification.json               completed UUID accounting audit
  inputs/checkpoints/exp2*/       local copies of the original best checkpoints
  inputs/images/                 original images
  inputs/manifests/               original train/val/test UUID manifests
  .venv/                         isolated Rainbow environment
  prepared/                      generated YOLO labels and image symlinks
  runs/<run-name>/
    smoke.log                    eight-image compatibility checks for all 56 checkpoints
    evaluation.log               full test evaluation progress
    full/results.csv             mAP@0.5, GT coverage, counts, hashes, and elapsed time
    full/results.md              all models in one comparison table
    full/comparison.png          paired mAP@0.5 and GT coverage chart
    full/comparison.svg          vector version of the comparison chart
    full/verification.json       per-image UUID and aggregate-count verification
    full/exp2*/<model>/
      metrics.json               standard metrics and GT coverage scalars
      gt_coverage.json           detected/missed UUIDs and extra prediction boxes
```

The `inputs`, runtime environments, prepared data, and raw runs are excluded from Git. They remain within this repository
folder. The completed summary, figures, verification, and experiment documentation are versioned.

## Run on Rainbow

Create an isolated environment under `exp3/.venv` using the existing Rainbow Python/CUDA environment as its base, then
install this checkout into it. The existing CUDA 11.8 build of PyTorch is compatible with Rainbow's NVIDIA 535 driver.

```bash
cd /home/varun/work/all-yolo-quantized
/home/varun/work/QAT720_withLLM/exp4/quantization/.venv/bin/python -m venv --system-site-packages exp3/.venv
exp3/.venv/bin/python -m pip install --no-deps -e .
nohup bash exp3/run_rainbow.sh exp3-20260911 > exp3/launcher.log 2>&1 < /dev/null &
```

The launcher first checks every checkpoint's SHA-256 and runs all 56 models on eight real test images. The full matrix
starts only after those checks succeed. It runs sequentially on GPU 1. Each completed model immediately saves its metrics,
UUID report, and an updated comparison table. Environment versions, source revision, input hashes, and GPU identity are
recorded in provenance files.

The final report verifies that every original test image is present and that every GT UUID appears exactly once in its
source image, as either detected or missed. Per-image counts must agree with the reported totals. It then generates a
paired comparison chart showing Exp2 and Exp2.5 for every architecture. To regenerate only this report on Rainbow:

```bash
CUDA_VISIBLE_DEVICES=1 YOLO_CONFIG_DIR="$PWD/exp3/.cache/settings" \
    MPLCONFIGDIR="$PWD/exp3/.cache/matplotlib" exp3/.venv/bin/python exp3/run.py \
    --manifest exp3/dataset.json --output exp3/runs/exp3-20260911 --report-only
```

After an interrupted evaluation, use `run.py --start-index N` with the same output directory and fixed protocol to resume
at the first unfinished row of `checkpoints.csv`. The row index is printed before each checkpoint is evaluated.

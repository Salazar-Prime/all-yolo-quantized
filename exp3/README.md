# Experiment 3: instance coverage beside mAP@0.5

Evaluate the 28 best checkpoints from Exp2 and the 28 best checkpoints from Exp2.5 on the **same full held-out test set**.
The model inventory is in `checkpoints.csv`; each entry records the original training job and expected checkpoint SHA-256.
This experiment performs evaluation only.

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

## Files

```text
exp3/
  checkpoints.csv                 56 checkpoints and their expected hashes
  dataset.json                    portable references to the three manifests
  run.py                          matrix evaluation using the existing coverage validator
  run_rainbow.sh                  Rainbow GPU 1 launcher, logs, PID, and exit status
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
    full/exp2*/<model>/
      metrics.json               standard metrics and GT coverage scalars
      gt_coverage.json           detected/missed UUIDs and extra prediction boxes
```

The `inputs`, runtime environments, prepared data, and raw runs are excluded from Git. They remain within this repository
folder. The summary and experiment documentation are versioned once the run completes.

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

After an interrupted evaluation, use `run.py --start-index N` with the same output directory and fixed protocol to resume
at the first unfinished row of `checkpoints.csv`. The row index is printed before each checkpoint is evaluated.

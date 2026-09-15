# Exp5 figure generation

`generate.py` renders the 28 paired INT8 PTQ/QAT results from the existing experiment summaries. It reads the archived
metrics and per-pass measurements; it does not launch inference or recompute the detector metrics. Render on Rainbow
and collect the outputs back into this directory on Anvil.

The first output set is in `output/exp5-20260914/`:

| Output                          | Contents                                                                                                                                      |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `map_ptq_vs_qat.{png,svg,pdf}`  | PTQ mAP50–95 on x, QAT on y; paired QAT − PTQ differences in percentage points                                                                |
| `fps_ptq_vs_qat.{png,svg,pdf}`  | Synchronized inference FPS and preprocessing + inference + postprocessing FPS                                                                 |
| `resources_ptq.{png,svg,pdf}`   | PTQ temperature, input power, energy, shared RAM, CPU, GPU, and memory-controller utilization                                                 |
| `resources_qat.{png,svg,pdf}`   | The same resource metrics for QAT, using the same column scales                                                                               |
| `explore.html`                  | Offline interactive accuracy scatter and resource explorer; hover for model details, select a metric, and click the legend to isolate PTQ/QAT |
| `metrics.csv`, `map_deltas.csv` | Numeric data behind the figures; mAP in `metrics.csv` remains on the original 0–1 scale                                                       |
| `source/`, `provenance.json`    | Frozen input snapshot, input SHA-256 hashes, renderer versions, and data-quality flags                                                        |

Regenerate this snapshot from the repository root **on Rainbow**:

```bash
exp5/.venv/bin/python exp5/figure_generation/generate.py \
    exp5/figure_generation/output/exp5-20260914/source \
    --output exp5/figure_generation/output/exp5-20260914
```

The source packet contains `results.csv`, `models.csv`, `protocol.json`, and each model's
`ptq/measurements.json` and `qat/measurements.json`, with device metadata where available. The supplied snapshot was
collected under the experiment's archive lock. Use a consistent snapshot when refreshing it while benchmarks run.
All 56 INT8 entries must have three successful full-test passes. Outputs include their own source copies and hashes.

Dependencies used for the first render are recorded in `requirements.txt`. They are already installed in Rainbow's
`exp5/.venv`. Generated outputs are kept locally in the repository and excluded from Git.

Interpretation:

- Positive mAP delta favors QAT; negative delta favors PTQ. The primary metric is mAP50–95; the HTML explorer also offers
  mAP50. A difference of `0.02` on the original 0–1 scale is **2 percentage points**, not a 2% relative change.
- FPS dots use the experiment's aggregate throughput. Whiskers show the minimum and maximum of three passes over the
  same test set; they are not confidence intervals or independent accuracy replications. Pipeline FPS excludes data
  loading and metric updates.
- Temperature is the `thermal@` sensor from the existing summaries, not the separate GPU temperature sensor.
  Temperature and memory-controller means are weighted by recorded sample counts; peaks are maxima across all passes.
- Power is the measured `VDD_IN` input rail. Power and energy cover full evaluation windows, including loading and metric
  updates; they are not isolated inference-only energy. RAM is shared system memory, not dedicated GPU memory.
- Resource heatmaps show absolute values with units. Each column has its own color scale, shared between PTQ and QAT;
  colors cannot be compared across different columns. Darker means higher, not necessarily better.
- YOLO11m PTQ has a 0.561-second telemetry gap in pass 3. Its recorded sensor summaries carry a dagger, and aggregate
  energy is unavailable. Accuracy and FPS measurements remain present.
- YOLOv9 uses `ubuntu`, YOLOv10 uses `ubuntu-1`, and the other families use `soysan`. L4T versions differ and clocks are
  automatic, so cross-family hardware comparisons require care. Early YOLO26n/s/m archives lack per-variant device
  metadata; their recorded device assignments are retained and the missing metadata is listed in provenance.

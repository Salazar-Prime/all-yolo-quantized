# Exp5 figure generation

`generate.py` renders the 28 paired INT8 PTQ/QAT results and, with `--report`, the complete 140-case comparison across
ONNX FP32, TensorRT FP32/FP16, and TensorRT INT8 PTQ/QAT. It reads the archived
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

The completed-campaign output set is in `output/exp5-complete-20260918/`. It includes the original comparisons,
resource heatmaps for all five formats, the interactive explorer with all 140 records, and:

| Output                                                                        | Purpose                                                                                               |
| ----------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `exp5_takeaways.pdf`                                                          | Eight landscape slides with graphs, numeric takeaways, and interpretation limits                      |
| `paired_format_heatmap.{png,svg,pdf}`                                         | All 28 models: accuracy change and pipeline speedup relative to the same model's TensorRT FP32 result |
| `format_overview.{png,svg,pdf}`                                               | Paired speedup medians and individual accuracy changes across all five formats                        |
| `int8_recovery.{png,svg,pdf}`                                                 | QAT recovery relative to PTQ, with FP32 as the reference                                              |
| `accuracy_speed_frontiers.{png,svg,pdf}`                                      | Accuracy–pipeline-FPS tradeoffs and observed frontiers, separately for each device                    |
| `latency_breakdown.{png,svg,pdf}`                                             | Preprocessing, inference, and postprocessing shares; inference/pipeline/full-evaluation FPS           |
| `energy_power_memory.{png,svg,pdf}`                                           | Paired energy ratios, board power, and peak shared system RAM                                         |
| `device_thermal_conditions.{png,svg,pdf}`                                     | Recorded thermal conditions and GPU clocks, with workload/device differences explicit                 |
| `effective_operation_rates.{png,svg,pdf}`                                     | Reference-model operation counts and derived operation rates                                          |
| `deployment_candidates.{png,svg,pdf}`                                         | FP16 candidates to study further on their measured devices                                            |
| `format_summary.csv`, `paired_comparisons.csv`, `accuracy_speed_frontier.csv` | Numeric comparisons and observed frontier membership                                                  |
| `findings.json`, `takeaways.md`                                               | Slide takeaways and supporting summary values in reusable formats                                     |

Regenerate the complete report **on Rainbow**:

```bash
exp5/.venv/bin/python exp5/figure_generation/generate.py \
    exp5/figure_generation/output/exp5-complete-20260918/source \
    --output exp5/figure_generation/output/exp5-complete-20260918 --report
```

Regenerate this snapshot from the repository root **on Rainbow**:

```bash
exp5/.venv/bin/python exp5/figure_generation/generate.py \
    exp5/figure_generation/output/exp5-20260914/source \
    --output exp5/figure_generation/output/exp5-20260914
```

The source packet contains `results.csv`, `models.csv`, `protocol.json`, and each model's
`ptq/measurements.json` and `qat/measurements.json`, with device metadata where available. The supplied snapshot was
collected under the experiment's archive lock. Use a consistent snapshot when refreshing it while benchmarks run.
The default requires all 56 INT8 entries; `--report` requires all 140 entries and their per-variant measurements.
Every entry must have three successful full-test passes. Outputs include their own source copies and hashes.

Dependencies used for the first render are recorded in `requirements.txt`. They are already installed in Rainbow's
`exp5/.venv`. Generated outputs are kept locally in the repository and excluded from Git.

For the requested Exp5 versus Exp6 comparison, use `--fallback-source` with a second source packet instead of `--report`.
This reuses the snapshot reader and exporter to pair `ptq`/`qat` with `ptq_fp16`/`qat_fp16`. It produces three graphs
(inference FPS, pipeline FPS, and paired speedups), PNG/SVG/PDF exports, a combined PDF, comparison CSVs, and source hashes.
Each packet includes the usual metrics and a `device.json` for every selected variant; historical device identities must
label campaign-level provenance when per-variant records are unavailable. See the
[Exp6 results and reproduction command](../../exp6/README.md#results-int8-with-fp32-fallback-versus-fp16-enabled).

Interpretation:

- Positive mAP delta favors QAT; negative delta favors PTQ. The primary metric is mAP50–95; the HTML explorer also offers
  mAP50. A difference of `0.02` on the original 0–1 scale is **2 percentage points**, not a 2% relative change.
- FPS dots use the experiment's aggregate throughput. Whiskers show the minimum and maximum of three passes over the
  same test set; they are not confidence intervals or independent accuracy replications. Pipeline FPS excludes data
  loading and metric updates.
- Report speedups, accuracy differences, and energy ratios are calculated within each model relative to TensorRT FP32,
  then summarized across models. A median of paired ratios is not a ratio of population medians. Energy statistics
  exclude missing pairs and report their counts. INT8 engines permit FP32 fallback, with FP16 disabled; the observed
  comparison is specific to this recipe and TensorRT version.
- Latency-share bars average each model's preprocessing/inference/postprocessing time fractions, giving models equal
  weight. Full-evaluation FPS divides processed images by measured evaluation wall time; it includes loading and metric
  updates but excludes warmup and initialization. It is not a field camera-stream benchmark.
- Accuracy–speed frontiers are exact observed nondominated points within each device: no other measured point is at
  least as accurate and fast, with one strict improvement. Tiny accuracy differences can create near-duplicate frontier
  points; they do not establish statistical significance. Boxplots summarize models, not confidence intervals.
- Operation rates are reference GFLOPs/image times synchronized inference FPS. They are equivalent model work rates,
  not hardware instruction counters; INT8 work is integer arithmetic and some functional operators are uncounted.
- Temperature is the `thermal@` sensor from the existing summaries, not the separate GPU temperature sensor.
  Temperature and memory-controller means are weighted by recorded sample counts; peaks are maxima across all passes.
- Power is the measured `VDD_IN` input rail. Power and energy cover full evaluation windows, including loading and metric
  updates; they are not isolated inference-only energy. RAM is shared system memory, not dedicated GPU memory.
- Resource heatmaps show absolute values with units. Each column has its own color scale, shared between PTQ and QAT;
  colors cannot be compared across different columns. Darker means higher, not necessarily better.
- YOLO11m PTQ has a 0.561-second telemetry gap in pass 3. Its recorded sensor summaries carry a dagger, and aggregate
  energy is unavailable. Accuracy and FPS measurements remain present.
- The completed campaign also has incomplete telemetry for YOLO12x FP16 pass 3. Its energy aggregate is likewise
  unavailable. The report retains both flagged rows, with 138/140 energy aggregates available.
- YOLOv9 uses `ubuntu`, YOLOv10 uses `ubuntu-1`, and the other families use `soysan`. L4T versions differ and clocks are
  automatic, so cross-family hardware comparisons require care. Early YOLO26n/s/m archives lack per-variant device
  metadata; their recorded device assignments are retained and the missing metadata is listed in provenance.
- The thermal slide's dots are per-model/format sample-weighted means; black marks are device-wide sample-weighted
  means. Different model families, software versions, run order, and automatic clocks prevent attributing temperature
  differences to cooling alone or concluding that thermal throttling occurred. The shortlist needs validation on the
  intended deployment board, camera pipeline, and unseen field data; it does not change the fixed training recipe.

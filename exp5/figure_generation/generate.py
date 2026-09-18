"""Render Exp5 accuracy, throughput, resource figures, and a complete-campaign PDF briefing."""

import argparse
import csv
import hashlib
import io
import json
import platform
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import plotly
import plotly.graph_objects as go

HERE = Path(__file__).resolve().parent
COLORS = {"ptq": "#3568BA", "qat": "#DE7545"}
ALL_COLORS = {"onnx": "#8995A6", "fp32": "#805AA3", "fp16": "#14877E", **COLORS}
RESOURCES = {
    "temperature_mean_c": "Mean thermal\nsensor (°C)",
    "temperature_peak_c": "Peak thermal\nsensor (°C)",
    "power_mean_w": "Mean input\npower (W)",
    "power_peak_w": "Peak input\npower (W)",
    "board_energy_j_per_image": "Board energy\n(J/image)",
    "ram_peak_mb": "Peak shared\nRAM (MiB)",
    "cpu_mean_pct": "Mean CPU\n(% across cores)",
    "gpu_mean_pct": "Mean GPU\nutilization (%)",
    "emc_mean_pct": "Mean memory\ncontroller (%)",
}
DEVICE_NOTE = "YOLOv9: ubuntu · YOLOv10: ubuntu-1 · others: soysan | Three Xavier NX devices; L4T versions differ."
PROTOCOL_NOTE = "4,579 test images × 3 passes · 640 × 640 · batch 1 · NMS IoU 0.5 · 20 W / automatic clocks"


def load(source, output, variants=COLORS):
    """Snapshot the inputs and reuse the experiment's existing aggregates without reparsing sensor logs."""
    hashes = {}

    def read(relative):
        data = (source / relative).read_bytes()
        path = output / "source" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        hashes[str(relative)] = hashlib.sha256(data).hexdigest()
        return data.decode()

    rows = list(csv.DictReader(io.StringIO(read(Path("results.csv")))))
    models = list(csv.DictReader(io.StringIO(read(Path("models.csv")))))
    read(Path("protocol.json"))
    families = ("yolo26", "yolov9", "yolov10", "yolo11", "yolo12")
    models.sort(
        key=lambda r: (next(i for i, f in enumerate(families) if r["model"].startswith(f)), int(r["checkpoint_bytes"]))
    )
    records, trials = {}, {}
    for item in models:
        model = item["model"]
        for variant in variants:
            row = next(r.copy() for r in rows if r["model"] == model and r["variant"] == variant)
            if row["status"] != "passed" or row["passes"] != "3":
                raise ValueError("Three completed passes required: " + model + "/" + variant)
            directory = Path(model) / variant
            passes = json.loads(read(directory / "measurements.json"))
            if len(passes) != 3 or any(p["test_images"] != 4579 or p["nms_iou"] != 0.5 for p in passes):
                raise ValueError("Measurement protocol mismatch: " + model + "/" + variant)
            if (source / directory / "device.json").exists():
                read(directory / "device.json")
            for key in row.keys() - {"model", "variant", "device", "status", "telemetry_complete"}:
                row[key] = float(row[key]) if row[key] else np.nan
            row["telemetry_complete"] = all(p["telemetry"]["telemetry_complete"] for p in passes)
            for field, target in (("temperature_c_mean", "temperature_mean_c"), ("emc_pct_mean", "emc_mean_pct")):
                counts = [p["telemetry"]["telemetry_samples"] for p in passes]
                row[target] = float(np.average([p["telemetry"][field] for p in passes], weights=counts))
            row["temperature_peak_c"] = max(p["telemetry"]["temperature_c_peak"] for p in passes)
            row["evaluation_fps"] = sum(p["test_images"] for p in passes) / sum(p["pass_wall_seconds"] for p in passes)
            row["gpu_clock_mean_mhz"] = float(
                np.average([p["telemetry"]["gpu_clock_mhz_mean"] for p in passes], weights=counts)
            )
            records[model, variant], trials[model, variant] = row, passes
    with (output / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(next(iter(records.values()))))
        writer.writeheader()
        writer.writerows(records.values())
    return [m["model"] for m in models], records, trials, hashes


def save(fig, output, name, title, note, slides=None, takeaway=None):
    fig.suptitle(title, x=0.055, y=0.98, ha="left", fontsize=20, weight="bold", color="#162A43")
    fig.text(0.055, 0.937, PROTOCOL_NOTE, fontsize=9, color="#4E6075")
    fig.text(0.055, 0.037, note + "\n" + DEVICE_NOTE, fontsize=9, color="#4E6075", linespacing=1.7)
    fig.subplots_adjust(top=0.88, bottom=0.12, left=0.15, right=0.97, wspace=0.4)
    if slides is not None:
        fig.subplots_adjust(top=0.84, bottom=0.30, left=0.075, right=0.975, wspace=0.40)
        fig.text(0.055, 0.16, textwrap.fill(takeaway, 112), fontsize=16, color="#162A43", weight="bold", va="center")
        slides.savefig(fig, facecolor="white")
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(output / (name + "." + suffix), dpi=180, facecolor="white")
    plt.close(fig)


def report(models, records, trials, output):
    """Compare matched formats and export a briefing with numerically derived takeaways."""
    variants, devices = list(ALL_COLORS), ("soysan", "ubuntu", "ubuntu-1")
    keys = (
        "map50_95",
        "pipeline_fps",
        "board_energy_j_per_image",
        "power_mean_w",
        "ram_peak_mb",
        "temperature_mean_c",
        "gpu_clock_mean_mhz",
        "reference_gflops_per_image",
        "effective_gops_per_second",
    )
    arrays = {k: np.array([[records[m, v][k] for v in variants] for m in models]) for k in keys}
    accuracy = arrays["map50_95"] * 100
    speedup = arrays["pipeline_fps"] / arrays["pipeline_fps"][:, [1]]
    loss = accuracy - accuracy[:, [1]]
    energy = arrays["board_energy_j_per_image"] / arrays["board_energy_j_per_image"][:, [1]]
    quality_gain = accuracy[:, 4] - accuracy[:, 3]
    device_mask = {d: np.array([records[m, "fp32"]["device"] == d for m in models]) for d in devices}
    summary = [
        {
            "variant": v,
            "models": len(models),
            "median_pipeline_speedup_vs_fp32": float(np.median(speedup[:, j])),
            "median_map_delta_pp_vs_fp32": float(np.median(loss[:, j])),
            "median_energy_reduction_percent_vs_fp32": float(np.nanmedian(100 * (1 - energy[:, j]))),
            "energy_pairs": int(np.isfinite(energy[:, j]).sum()),
        }
        for j, v in enumerate(variants)
    ]
    with (output / "format_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    paired = [
        {
            "model": m,
            "variant": v,
            "device": records[m, v]["device"],
            "map_delta_pp_vs_fp32": loss[i, j],
            "pipeline_speedup_vs_fp32": speedup[i, j],
            "energy_ratio_vs_fp32": energy[i, j],
        }
        for i, m in enumerate(models)
        for j, v in enumerate(variants)
    ]
    with (output / "paired_comparisons.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)

    # Dense study figure complements the larger type and smaller plots in the slides.
    fig, axes = plt.subplots(1, 2, figsize=(17, 12))
    for ax, matrix, title, cmap, bounds in (
        (axes[0], loss[:, [0, 2, 3, 4]], "mAP50–95 change vs TRT FP32 (pp)", "RdBu", (-31, 31)),
        (axes[1], speedup[:, [0, 2, 3, 4]], "Pipeline FPS / TRT FP32 FPS", "YlGnBu", (0, float(speedup.max()))),
    ):
        ax.imshow(matrix, cmap=cmap, vmin=bounds[0], vmax=bounds[1], aspect="auto")
        ax.set(
            xticks=range(4), xticklabels=["ONNX", "FP16", "PTQ", "QAT"], yticks=range(len(models)), yticklabels=models
        )
        ax.set_title(title, pad=15)
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                fraction = (value - bounds[0]) / (bounds[1] - bounds[0])
                white = fraction > 0.67 or (ax is axes[0] and fraction < 0.25)
                ax.text(
                    j,
                    i,
                    f"{value:+.2f}" if ax is axes[0] else f"{value:.2f}×",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if white else "#162A43",
                )
    save(
        fig,
        output,
        "paired_format_heatmap",
        "Every model | accuracy and speed relative to TensorRT FP32",
        "Paired within the same model and assigned device. ONNX uses FP32; INT8 engines allow FP32 fallback, but no FP16.",
    )

    takeaways, frontier_rows = [], []
    with PdfPages(output / "exp5_takeaways.pdf") as slides:

        def finish(fig, name, title, takeaway, note):
            takeaways.append({"figure": name, "title": title, "takeaway": takeaway, "note": note})
            save(fig, output, name, title, note, slides, takeaway)

        fig, axes = plt.subplots(1, 2, figsize=(16, 9))
        for j, v in enumerate(variants):
            med = np.median(speedup[:, j])
            axes[0].barh(j, med, color=ALL_COLORS[v], height=0.6)
            axes[0].text(med + 0.04, j, f"{med:.2f}×", va="center")
            axes[1].scatter(loss[:, j], np.full(len(models), j), color=ALL_COLORS[v], alpha=0.7, s=22)
            axes[1].plot(np.median(loss[:, j]), j, "|", color="black", markersize=20, markeredgewidth=2)
        for ax in axes:
            ax.set(yticks=range(5), yticklabels=[v.upper() for v in variants])
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=0.15)
        axes[0].set(xlabel="Median paired pipeline speedup vs TRT FP32", xlim=(0, 3.0))
        axes[0].axvline(1, color="gray", ls="--")
        axes[1].set(xlabel="mAP50–95 difference from TRT FP32 (pp)")
        axes[1].axvline(0, color="gray", ls="--")
        fp16_wins = int((arrays["pipeline_fps"][:, 2] > arrays["pipeline_fps"][:, 3]).sum())
        finish(
            fig,
            "format_overview",
            "01 | FP16 offers the strongest observed speed–accuracy balance",
            f"FP16: {summary[2]['median_pipeline_speedup_vs_fp32']:.2f}× median pipeline speed, {summary[2]['median_map_delta_pp_vs_fp32']:+.2f} pp median mAP change. Faster than PTQ on {fp16_wins}/{len(models)} models.",
            "140/140 combinations; 420 test passes. Dots: models; black marks: medians. INT8 permits FP32 fallback, with FP16 disabled; results are recipe-specific.",
        )

        fig, axes = plt.subplots(1, 2, figsize=(16, 9))
        low = min(loss[:, 3:].min(), -1) - 2
        axes[0].plot([low, 2], [low, 2], "--", color="gray")
        axes[0].scatter(loss[:, 3], loss[:, 4], s=40, color="#147C80")
        axes[0].set(xlabel="PTQ change vs FP32 (pp)", ylabel="QAT change vs FP32 (pp)", xlim=(low, 2), ylim=(low, 2))
        for i in set(np.argsort(abs(quality_gain))[-3:]) | {int(np.argmin(loss[:, 4]))}:
            axes[0].annotate(models[i], (loss[i, 3], loss[i, 4]), xytext=(5, 6), textcoords="offset points", fontsize=9)
        chosen = sorted(
            set(np.argsort(quality_gain)[:3]) | set(np.argsort(quality_gain)[-4:]), key=lambda i: quality_gain[i]
        )
        axes[1].barh(
            range(len(chosen)),
            quality_gain[chosen],
            color=[COLORS["qat"] if quality_gain[i] > 0 else COLORS["ptq"] for i in chosen],
        )
        axes[1].set(
            yticks=range(len(chosen)), yticklabels=[models[i] for i in chosen], xlabel="QAT − PTQ mAP50–95 (pp)"
        )
        axes[1].axvline(0, color="gray")
        for ax in axes:
            ax.grid(axis="x", alpha=0.15)
        wins = int((quality_gain > 0).sum())
        finish(
            fig,
            "int8_recovery",
            "02 | QAT helps some models, but does not reliably recover FP32 accuracy",
            f"QAT improves {wins}/{len(models)} models over PTQ; {int((loss[:, 4] < 0).sum())}/{len(models)} QAT results remain below FP32. Largest QAT loss: {models[int(np.argmin(loss[:, 4]))]} ({loss[:, 4].min():.1f} pp).",
            "Left: all models; above the diagonal favors QAT. Right: four largest gains and three largest losses. Ten fixed QAT epochs; no test-set tuning.",
        )

        fig, axes = plt.subplots(1, 3, figsize=(16, 9))
        for ax, device in zip(axes, devices):
            points = [(i, j) for i in np.flatnonzero(device_mask[device]) for j in range(5)]
            frontier = [
                (i, j)
                for i, j in points
                if not any(
                    accuracy[k, other] >= accuracy[i, j]
                    and arrays["pipeline_fps"][k, other] >= arrays["pipeline_fps"][i, j]
                    and (
                        accuracy[k, other] > accuracy[i, j]
                        or arrays["pipeline_fps"][k, other] > arrays["pipeline_fps"][i, j]
                    )
                    for k, other in points
                )
            ]
            frontier.sort(key=lambda ij: arrays["pipeline_fps"][ij])
            for j, v in enumerate(variants):
                ax.scatter(
                    arrays["pipeline_fps"][device_mask[device], j],
                    accuracy[device_mask[device], j],
                    s=28,
                    color=ALL_COLORS[v],
                    label=v.upper(),
                    alpha=0.8,
                )
            ax.plot(
                [arrays["pipeline_fps"][ij] for ij in frontier],
                [accuracy[ij] for ij in frontier],
                color="#283B50",
                lw=1,
            )
            for i, j in frontier:
                frontier_rows.append(
                    {
                        "device": device,
                        "model": models[i],
                        "variant": variants[j],
                        "map50_95_percent": accuracy[i, j],
                        "pipeline_fps": arrays["pipeline_fps"][i, j],
                    }
                )
            for n, (i, j) in enumerate(ij for ij in frontier if ij[1] == 2):
                ax.annotate(
                    models[i] + "/FP16",
                    (arrays["pipeline_fps"][i, j], accuracy[i, j]),
                    xytext=(-4, -18 - 26 * n),
                    textcoords="offset points",
                    ha="right",
                    fontsize=8,
                    arrowprops={"arrowstyle": "-", "color": "#526579", "lw": 0.6},
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1},
                )
            ax.set(title=device, xlabel="Pipeline FPS", ylim=(-1, 40))
            ax.grid(alpha=0.15)
        axes[0].set_ylabel("mAP50–95 (%)")
        axes[0].legend(loc="lower right", fontsize=8, frameon=False)
        fast = np.unravel_index(np.argmax(arrays["pipeline_fps"]), accuracy.shape)
        finish(
            fig,
            "accuracy_speed_frontiers",
            "03 | Choose candidates within a device's measured workloads",
            f"Fastest observed: {models[fast[0]]}/{variants[fast[1]].upper()} at {arrays['pipeline_fps'][fast]:.1f} pipeline FPS and {accuracy[fast]:.2f}% mAP. YOLO26n/FP16: {records['yolo26n', 'fp16']['map50_95'] * 100:.2f}% at {records['yolo26n', 'fp16']['pipeline_fps']:.1f} FPS.",
            "Lines: exact observed nondominated points (higher accuracy and FPS). Tiny accuracy differences are not evidence of significance; families differ by device.",
        )

        fig, axes = plt.subplots(1, 2, figsize=(16, 9))
        latency = np.array(
            [
                [
                    [
                        sum(p["speed_ms_per_image"][k] * p["test_images"] for p in trials[m, v])
                        / sum(p["test_images"] for p in trials[m, v])
                        for k in ("preprocess", "inference", "postprocess")
                    ]
                    for v in variants
                ]
                for m in models
            ]
        )
        shares = (latency / latency.sum(axis=2, keepdims=True)).mean(axis=0) * 100
        bottom = np.zeros(5)
        for k, (name, color) in enumerate(
            zip(("Preprocess", "Inference", "Postprocess"), ("#8995A6", "#14877E", "#DE7545"))
        ):
            axes[0].bar(range(5), shares[:, k], bottom=bottom, label=name, color=color)
            bottom += shares[:, k]
        axes[0].set(
            xticks=range(5),
            xticklabels=[v.upper() for v in variants],
            ylabel="Mean per-model pipeline time share (%)",
            ylim=(0, 112),
        )
        axes[0].legend(loc="upper center", ncol=3, fontsize=8)
        example = models.index("yolo11n")
        x = np.arange(5)
        for offset, key, label, color in [
            (-0.25, "inference_fps", "Inference", "#14877E"),
            (0, "pipeline_fps", "Pipeline", "#3568BA"),
            (0.25, "evaluation_fps", "Full evaluation", "#DE7545"),
        ]:
            axes[1].bar(
                x + offset, [records[models[example], v][key] for v in variants], width=0.24, label=label, color=color
            )
        axes[1].set(xticks=x, xticklabels=[v.upper() for v in variants], ylabel="YOLO11n throughput (FPS)")
        axes[1].legend(fontsize=9, frameon=False)
        finish(
            fig,
            "latency_breakdown",
            "04 | Inference FPS overstates application throughput",
            f"YOLO11n/FP16: {records['yolo11n', 'fp16']['inference_fps']:.1f} inference FPS becomes {records['yolo11n', 'fp16']['pipeline_fps']:.1f} pipeline FPS and {records['yolo11n', 'fp16']['evaluation_fps']:.1f} full-evaluation FPS.",
            "Pipeline = preprocessing + synchronized inference + postprocessing. Full evaluation also includes loading/metrics; it is not a camera-stream benchmark.",
        )

        fig, axes = plt.subplots(1, 3, figsize=(16, 9))
        for ax, matrix, label in zip(
            axes,
            (energy, arrays["power_mean_w"], arrays["ram_peak_mb"] / 1024),
            ("Board J/image / same-model FP32", "Mean board input power (W)", "Peak shared system RAM (GiB)"),
        ):
            boxes = ax.boxplot(
                [col[np.isfinite(col)] for col in matrix.T],
                patch_artist=True,
                tick_labels=[v.upper() for v in variants],
                widths=0.55,
            )
            for patch, v in zip(boxes["boxes"], variants):
                patch.set_facecolor(ALL_COLORS[v])
                patch.set_alpha(0.75)
            ax.set_ylabel(label)
            ax.grid(axis="y", alpha=0.15)
            ax.tick_params(axis="x", labelsize=9)
        axes[0].axhline(1, color="gray", ls="--")
        finish(
            fig,
            "energy_power_memory",
            "05 | Faster evaluation reduces energy more than power alone suggests",
            f"FP16 cuts board energy/image by a median {summary[2]['median_energy_reduction_percent_vs_fp32']:.1f}% vs FP32 ({summary[2]['energy_pairs']} pairs). All 140 runs fit; peak observed system RAM is {arrays['ram_peak_mb'].max() / 1024:.2f} GiB.",
            "Boxplots summarize models, not confidence intervals. Energy covers full evaluation, not inference alone. Missing energy: YOLO11m/PTQ and YOLO12x/FP16.",
        )

        fig, axes = plt.subplots(1, 2, figsize=(16, 9))
        thermal = {}
        for j, device in enumerate(devices):
            color = ["#14877E", "#3568BA", "#DE7545"][j]
            pp = [p for i in np.flatnonzero(device_mask[device]) for v in variants for p in trials[models[i], v]]
            count = sum(p["telemetry"]["telemetry_samples"] for p in pp)
            mean = sum(p["telemetry"]["temperature_c_mean"] * p["telemetry"]["telemetry_samples"] for p in pp) / count
            peak = max(p["telemetry"]["temperature_c_peak"] for p in pp)
            thermal[device] = {
                "sample_weighted_mean_c": mean,
                "peak_c": peak,
                "combinations": int(device_mask[device].sum()) * 5,
            }
            axes[0].scatter(
                np.full(device_mask[device].sum() * 5, j),
                arrays["temperature_mean_c"][device_mask[device]].ravel(),
                alpha=0.5,
                s=28,
                color=color,
            )
            axes[0].plot(j, mean, "_", color="black", markersize=25, markeredgewidth=3)
            axes[0].text(j, peak + 2, f"peak {peak:.1f}°C", ha="center", fontsize=10)
            axes[1].scatter(
                arrays["temperature_mean_c"][device_mask[device]].ravel(),
                arrays["gpu_clock_mean_mhz"][device_mask[device]].ravel(),
                s=30,
                label=device,
                alpha=0.65,
                color=color,
            )
        axes[0].set(
            xticks=range(3),
            xticklabels=[f"{d}\n{thermal[d]['combinations']} runs" for d in devices],
            ylabel="Mean thermal@ temperature per run (°C)",
            xlim=(-0.5, 2.5),
            ylim=(40, 97),
        )
        axes[1].set(xlabel="Mean thermal@ temperature (°C)", ylabel="Mean recorded GPU clock (MHz)")
        axes[1].legend(frameon=False)
        finish(
            fig,
            "device_thermal_conditions",
            "06 | Device operating conditions differ substantially",
            f"Sample-weighted temperatures: soysan {thermal['soysan']['sample_weighted_mean_c']:.1f}°C; ubuntu {thermal['ubuntu']['sample_weighted_mean_c']:.1f}°C; ubuntu-1 {thermal['ubuntu-1']['sample_weighted_mean_c']:.1f}°C. Cross-device comparisons need this context.",
            "thermal@ is not GPU@. Different families, software, run order and automatic clocks confound causes; these plots do not prove throttling or explain reboots.",
        )

        fig, axes = plt.subplots(1, 3, figsize=(16, 9))
        for ax, device in zip(axes, devices):
            for j, v in enumerate(variants):
                mask = device_mask[device]
                ax.scatter(
                    arrays["reference_gflops_per_image"][mask, j],
                    arrays["effective_gops_per_second"][mask, j],
                    color=ALL_COLORS[v],
                    s=30,
                    label=v.upper(),
                    alpha=0.8,
                )
            ax.set(title=device, xlabel="Reference GFLOPs / image", xscale="log")
            ax.grid(alpha=0.15)
        axes[0].set_ylabel("Reference-equivalent operation rate (GOP/s)")
        axes[0].legend(fontsize=8, frameon=False)
        finish(
            fig,
            "effective_operation_rates",
            "07 | Operation rate and useful image throughput answer different questions",
            "Use measured pipeline FPS for deployment decisions. Derived GOP/s describes reference model work per second, not executed instructions or hardware utilization.",
            "Rate = reference GFLOPs/image × synchronized inference FPS; MAC = 2 operations. Quantized engines use integer work; profiling misses some functional operators.",
        )

        fig, ax = plt.subplots(figsize=(16, 9))
        ax.axis("off")
        selections = [(r["device"], r["model"], r["variant"]) for r in frontier_rows if r["variant"] == "fp16"]
        table = ax.table(
            cellText=[
                [
                    d,
                    m + "/" + v.upper(),
                    f"{records[m, v]['map50_95'] * 100:.2f}",
                    f"{records[m, v]['pipeline_fps']:.1f}",
                    f"{records[m, v]['board_energy_j_per_image']:.3f}",
                ]
                for d, m, v in selections
            ],
            colLabels=["Device", "FP16 study candidate", "mAP50–95 (%)", "Pipeline FPS", "Board J/image"],
            cellLoc="center",
            loc="center",
            colWidths=[0.13, 0.29, 0.19, 0.18, 0.19],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(14)
        table.scale(1, 3.4)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("white")
            cell.set_facecolor("#162A43" if row == 0 else "#EDF2F6" if row % 2 else "#F7F9FB")
            if row == 0:
                cell.get_text().set_color("white")
        finish(
            fig,
            "deployment_candidates",
            "08 | Start field validation with measured FP16 candidates",
            "Choose the accuracy/FPS budget first, then validate on the intended board and camera pipeline. Treat large INT8 accuracy losses as failures to investigate before deployment.",
            "Descriptive shortlist, not a new test-tuned model recipe or a claim of statistical superiority. Field latency, thermals and unseen-data accuracy remain to be checked.",
        )
    with (output / "accuracy_speed_frontier.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frontier_rows[0]))
        writer.writeheader()
        writer.writerows(frontier_rows)
    (output / "findings.json").write_text(
        json.dumps(
            {
                "format_summary": summary,
                "qat_beats_ptq_models": wins,
                "fp16_faster_than_ptq_models": fp16_wins,
                "thermal": thermal,
                "slides": takeaways,
            },
            indent=2,
        )
        + "\n"
    )
    (output / "takeaways.md").write_text(
        "# Exp5 complete-campaign findings\n\n"
        + "\n\n".join(
            f"## {t['title']}\n\n{t['takeaway']}\n\n{t['note']}\n\n[Figure]({t['figure']}.png)" for t in takeaways
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source", type=Path, help="Snapshot with results.csv, models.csv, protocol.json, and model/variant JSON files"
    )
    parser.add_argument("--output", type=Path, default=HERE / "output" / "latest")
    parser.add_argument(
        "--report", action="store_true", help="Include all five formats and an eight-slide PDF briefing"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    variants = ALL_COLORS if args.report else COLORS
    models, records, trials, hashes = load(args.source, args.output, variants)

    def values(key, variant):
        return np.array([records[m, variant][key] for m in models])

    incomplete = [m + "/" + v for (m, v), r in records.items() if not r["telemetry_complete"]]
    ptq, qat = values("map50_95", "ptq") * 100, values("map50_95", "qat") * 100
    delta, positions = qat - ptq, np.arange(len(models))
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )

    fig, (scatter, bars) = plt.subplots(1, 2, figsize=(18, 11), gridspec_kw={"width_ratios": [1, 1.2]})
    limit = float(np.ceil(max(ptq.max(), qat.max()) / 5) * 5)
    scatter.plot([0, limit], [0, limit], "--", color="#98A6B5", lw=1.2, label="Equal mAP")
    for device, marker in (("soysan", "o"), ("ubuntu", "s"), ("ubuntu-1", "^")):
        selected = np.array([records[m, "ptq"]["device"] == device for m in models])
        scatter.scatter(
            ptq[selected],
            qat[selected],
            s=65,
            marker=marker,
            color="#147C80",
            alpha=0.8,
            edgecolor="white",
            label=device,
        )
    for i in np.argsort(np.abs(delta))[-8:]:
        scatter.annotate(
            models[i], (ptq[i], qat[i]), xytext=(5, 6 if i % 2 else -13), textcoords="offset points", fontsize=8
        )
    scatter.set(xlabel="PTQ mAP50–95 (%)", ylabel="QAT mAP50–95 (%)", xlim=(-1, limit), ylim=(-1, limit))
    scatter.set_aspect("equal", adjustable="box")
    scatter.set_title("Above the diagonal: QAT has higher mAP", fontsize=11, pad=16)
    scatter.grid(alpha=0.15)
    scatter.legend(loc="lower right", frameon=False, fontsize=9)
    bars.barh(positions, delta, color=[COLORS["qat"] if v >= 0 else COLORS["ptq"] for v in delta], height=0.65)
    bars.axvline(0, color="#718096", lw=1)
    for i, d in enumerate(delta):
        bars.text(
            d + (0.3 if d >= 0 else -0.3), i, f"{d:+.2f}", va="center", ha="left" if d >= 0 else "right", fontsize=8
        )
    bars.set(
        yticks=positions,
        yticklabels=models,
        xlabel="Δ mAP50–95 = QAT − PTQ (percentage points)",
        xlim=(min(delta.min() - 3, -3), delta.max() + 4),
    )
    bars.invert_yaxis()
    bars.grid(axis="x", alpha=0.15)
    save(
        fig,
        args.output,
        "map_ptq_vs_qat",
        "Accuracy | PTQ versus QAT",
        "Positive Δ favors QAT; negative Δ favors PTQ. Percent-point differences, not relative percentage changes.",
    )

    fig, axes = plt.subplots(1, 2, figsize=(17, 12), sharey=True)
    for ax, key, label in zip(
        axes,
        ("inference_fps", "pipeline_fps"),
        ("Synchronized inference FPS", "Pipeline FPS: preprocess + inference + postprocess"),
    ):
        for variant, offset in (("ptq", -0.13), ("qat", 0.13)):
            center = values(key, variant)
            bounds = np.array([[p[key] for p in trials[m, variant]] for m in models])
            ax.errorbar(
                center,
                positions + offset,
                xerr=[center - bounds.min(axis=1), bounds.max(axis=1) - center],
                fmt="o",
                ms=4,
                capsize=2,
                color=COLORS[variant],
                label=variant.upper(),
            )
        ax.set(xlabel=label, yticks=positions, yticklabels=models, xlim=(0, None))
        ax.grid(axis="x", alpha=0.15)
        ax.legend(loc="lower right", frameon=False)
    axes[0].invert_yaxis()
    save(
        fig,
        args.output,
        "fps_ptq_vs_qat",
        "Throughput | PTQ versus QAT",
        "Dots: aggregate FPS across 3 passes. Whiskers: observed pass minimum–maximum, not confidence intervals. Pipeline excludes loading/metrics.",
    )

    matrices = {v: np.array([[records[m, v][k] for k in RESOURCES] for m in models]) for v in variants}
    combined = np.vstack(list(matrices.values()))
    low, high = np.nanmin(combined, axis=0), np.nanmax(combined, axis=0)
    for variant, matrix in matrices.items():
        fig, ax = plt.subplots(figsize=(17, 12))
        normalized = (matrix - low) / np.where(high > low, high - low, 1)
        cmap = plt.get_cmap("YlGnBu").copy()
        cmap.set_bad("#E9ECEF")
        ax.imshow(np.ma.masked_invalid(normalized), cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                text = (
                    "N/A"
                    if not np.isfinite(value)
                    else (f"{value:.2f}" if j == 4 else f"{value:.0f}" if j == 5 else f"{value:.1f}")
                )
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if normalized[i, j] > 0.65 else "#18334D",
                )
        labels = [m + (" †" if not records[m, variant]["telemetry_complete"] else "") for m in models]
        ax.set(
            xticks=np.arange(len(RESOURCES)), xticklabels=list(RESOURCES.values()), yticks=positions, yticklabels=labels
        )
        ax.xaxis.tick_top()
        ax.tick_params(length=0, pad=9, labelsize=9)
        save(
            fig,
            args.output,
            "resources_" + variant,
            "Resource profile | " + variant.upper(),
            "Darker = higher within each column; shared scales across formats. Thermal sensor = tegrastats thermal@. † Incomplete telemetry; energy omitted.",
        )

    custom = [[records[m, "ptq"]["device"], float(delta[i])] for i, m in enumerate(models)]
    accuracy = go.Figure(
        go.Scatter(
            x=ptq,
            y=qat,
            text=models,
            customdata=custom,
            mode="markers",
            marker={
                "size": 11,
                "color": delta,
                "colorscale": "RdBu_r",
                "cmin": -max(abs(delta)),
                "cmax": max(abs(delta)),
                "colorbar": {"title": "Δ mAP (pp)"},
            },
            hovertemplate="%{text}<br>Device: %{customdata[0]}<br>PTQ: %{x:.3f}%<br>QAT: %{y:.3f}%<br>Δ: %{customdata[1]:+.3f} pp<extra></extra>",
        )
    )
    accuracy.add_shape(type="line", x0=0, y0=0, x1=limit, y1=limit, line={"dash": "dash", "color": "gray"})
    accuracy.update_layout(
        template="plotly_white",
        title="PTQ vs QAT mAP50–95 — hover to identify every model",
        height=650,
        xaxis_title="PTQ mAP50–95 (%)",
        yaxis_title="QAT mAP50–95 (%)",
        yaxis={"scaleanchor": "x", "scaleratio": 1},
    )
    metrics = {
        "inference_fps": "Inference FPS",
        "pipeline_fps": "Pipeline FPS",
        "evaluation_fps": "Full evaluation FPS (includes loading / metrics)",
        "effective_gops_per_second": "Reference-equivalent GOP/s (derived, not hardware counters)",
        "gpu_clock_mean_mhz": "Mean GPU clock (MHz)",
        **RESOURCES,
        "map50": "mAP50 (%)",
        "map50_95": "mAP50–95 (%)",
    }
    metric_values = {k: [values(k, v) * (100 if k.startswith("map") else 1) for v in variants] for k in metrics}
    explorer = go.Figure()
    for variant in variants:
        quality = [
            [
                records[m, variant]["device"],
                "complete" if records[m, variant]["telemetry_complete"] else "incomplete telemetry; energy unavailable",
            ]
            for m in models
        ]
        explorer.add_bar(
            x=values("inference_fps", variant),
            y=models,
            orientation="h",
            name=variant.upper(),
            marker_color=variants[variant],
            customdata=quality,
            hovertemplate="%{y}<br>%{x:.3f}<br>Device: %{customdata[0]}<br>Telemetry: %{customdata[1]}<extra>%{fullData.name}</extra>",
        )
    buttons = [
        {
            "label": label.replace("\n", " "),
            "method": "update",
            "args": [{"x": metric_values[key]}, {"xaxis.title.text": label.replace("\n", " ")}],
        }
        for key, label in metrics.items()
    ]
    explorer.update_layout(
        template="plotly_white",
        title="Model resource explorer — select a metric; click a format in the legend to hide/show it",
        height=1600 if args.report else 1100,
        barmode="group",
        xaxis_title="Inference FPS",
        yaxis={"autorange": "reversed"},
        margin={"l": 110, "t": 135},
        updatemenus=[{"buttons": buttons, "x": 0, "y": 1.08, "xanchor": "left"}],
    )
    config = {"displaylogo": False, "toImageButtonOptions": {"format": "svg"}}
    html = "<!doctype html><meta charset='utf-8'><title>Exp5 figures</title><style>body{font:16px system-ui;max-width:1300px;margin:40px auto;padding:0 24px;color:#162a43}p{line-height:1.6}</style>"
    html += (
        "<h1>Exp5 · "
        + " / ".join(v.upper() for v in variants)
        + "</h1><p>"
        + PROTOCOL_NOTE
        + "<br>"
        + DEVICE_NOTE
        + "</p>"
    )
    html += (
        "<p>Temperature is the thermal@ sensor. Board power/energy cover the complete test evaluation windows. Incomplete telemetry: "
        + (", ".join(incomplete) or "none")
        + ". Recorded sensor summaries are shown; incomplete energy is unavailable. FPS whiskers in the static figure describe repeat variation, not statistical uncertainty.</p>"
    )
    html += accuracy.to_html(full_html=False, include_plotlyjs=True, config=config)
    html += explorer.to_html(full_html=False, include_plotlyjs=False, config=config)
    (args.output / "explore.html").write_text(html)
    with (args.output / "map_deltas.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "device", "ptq_map50_95_percent", "qat_map50_95_percent", "qat_minus_ptq_pp"])
        writer.writerows([m, records[m, "ptq"]["device"], ptq[i], qat[i], delta[i]] for i, m in enumerate(models))
    provenance = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "models": len(models),
        "variants": list(variants),
        "metric": "map50_95",
        "delta": "100 * (QAT - PTQ), percentage points",
        "input_sha256": hashes,
        "versions": {
            "python": platform.python_version(),
            "matplotlib": matplotlib.__version__,
            "numpy": np.__version__,
            "plotly": plotly.__version__,
        },
        "incomplete_telemetry": incomplete,
        "missing_device_metadata": [
            m + "/" + v for m in models for v in variants if str(Path(m) / v / "device.json") not in hashes
        ],
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    if args.report:
        report(models, records, trials, args.output)
    print(
        f"Saved figures (PNG/SVG/PDF), interactive HTML, CSV data and provenance for {len(models)} models to {args.output}"
    )


if __name__ == "__main__":
    main()

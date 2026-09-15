"""Render paired INT8 accuracy, throughput, and resource figures from archived Exp5 measurements."""

import argparse
import csv
import hashlib
import io
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly
import plotly.graph_objects as go

HERE = Path(__file__).resolve().parent
COLORS = {"ptq": "#3568BA", "qat": "#DE7545"}
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


def load(source, output):
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
        for variant in COLORS:
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
            records[model, variant], trials[model, variant] = row, passes
    with (output / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(next(iter(records.values()))))
        writer.writeheader()
        writer.writerows(records.values())
    return [m["model"] for m in models], records, trials, hashes


def save(fig, output, name, title, note):
    fig.suptitle(title, x=0.055, y=0.98, ha="left", fontsize=20, weight="bold", color="#162A43")
    fig.text(0.055, 0.937, PROTOCOL_NOTE, fontsize=9, color="#4E6075")
    fig.text(0.055, 0.037, note + "\n" + DEVICE_NOTE, fontsize=9, color="#4E6075", linespacing=1.7)
    fig.subplots_adjust(top=0.88, bottom=0.12, left=0.15, right=0.97, wspace=0.4)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(output / (name + "." + suffix), dpi=180, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source", type=Path, help="Snapshot with results.csv, models.csv, protocol.json, and model/variant JSON files"
    )
    parser.add_argument("--output", type=Path, default=HERE / "output" / "latest")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    models, records, trials, hashes = load(args.source, args.output)

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

    matrices = {v: np.array([[records[m, v][k] for k in RESOURCES] for m in models]) for v in COLORS}
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
            "Darker = higher within each column; identical scales for PTQ/QAT. Thermal sensor = tegrastats thermal@. † Incomplete telemetry; energy omitted.",
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
        **RESOURCES,
        "map50": "mAP50 (%)",
        "map50_95": "mAP50–95 (%)",
    }
    metric_values = {k: [values(k, v) * (100 if k.startswith("map") else 1) for v in COLORS] for k in metrics}
    explorer = go.Figure()
    for variant in COLORS:
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
            marker_color=COLORS[variant],
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
        title="Model resource explorer — select a metric; click PTQ/QAT in the legend to isolate a method",
        height=1100,
        barmode="group",
        xaxis_title="Inference FPS",
        yaxis={"autorange": "reversed"},
        margin={"l": 110, "t": 135},
        updatemenus=[{"buttons": buttons, "x": 0, "y": 1.08, "xanchor": "left"}],
    )
    config = {"displaylogo": False, "toImageButtonOptions": {"format": "svg"}}
    html = "<!doctype html><meta charset='utf-8'><title>Exp5 INT8 figures</title><style>body{font:16px system-ui;max-width:1300px;margin:40px auto;padding:0 24px;color:#162a43}p{line-height:1.6}</style>"
    html += "<h1>Exp5 · PTQ and QAT</h1><p>" + PROTOCOL_NOTE + "<br>" + DEVICE_NOTE + "</p>"
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
            m + "/" + v for m in models for v in COLORS if str(Path(m) / v / "device.json") not in hashes
        ],
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(
        f"Saved 4 figures (PNG/SVG/PDF), interactive HTML, CSV data and provenance for {len(models)} models to {args.output}"
    )


if __name__ == "__main__":
    main()

"""Join Xavier measurements with raw tegrastats samples and retain failed deployment rows."""

import argparse
import bisect
import csv
import json
import re
from collections import Counter
from pathlib import Path

from run import deployment_for


def telemetry_window(samples, timestamps, start, end):
    selected = samples[max(0, bisect.bisect_left(timestamps, start) - 1) : bisect.bisect_right(timestamps, end) + 1]
    inside = [s for s in selected if start <= s["monotonic"] <= end]
    if len(inside) < 2:
        return {"telemetry_samples": len(inside), "telemetry_complete": False}
    result = {
        "telemetry_samples": len(inside),
        "telemetry_complete": selected[0]["monotonic"] <= start
        and selected[-1]["monotonic"] >= end
        and all(b["monotonic"] - a["monotonic"] < 0.5 for a, b in zip(selected, selected[1:])),
    }
    fields = {
        "power_w": r"VDD_IN (\d+)(?:mW)?/",
        "ram_mb": r"RAM (\d+)/",
        "swap_mb": r"SWAP (\d+)/",
        "gpu_pct": r"GR3D_FREQ (\d+)%",
        "emc_pct": r"EMC_FREQ (\d+)%",
        "temperature_c": r"thermal@([\d.]+)C",
        "gpu_clock_mhz": r"GR3D_FREQ \d+%@\[?(\d+)",
        "cpu_gpu_cv_power_w": r"VDD_CPU_GPU_CV (\d+)(?:mW)?/",
        "soc_power_w": r"VDD_SOC (\d+)(?:mW)?/",
    }
    for name, pattern in fields.items():
        values = [float(m.group(1)) for s in inside for m in [re.search(pattern, s["raw"])] if m]
        if values:
            scale = 0.001 if name.endswith("power_w") else 1
            result[name + "_mean"] = sum(values) / len(values) * scale
            result[name + "_peak"] = max(values) * scale
    cpu = []
    for sample in inside:
        match = re.search(r"CPU \[([^]]+)\]", sample["raw"])
        if match:
            cpu.append([float(v.split("%")[0]) if "%" in v else 0.0 for v in match.group(1).split(",")])
    if cpu:
        result["cpu_mean_pct_per_core"] = [sum(c[i] for c in cpu) / len(cpu) for i in range(len(cpu[0]))]
    cpu_clocks = [
        float(v) for s in inside for v in re.findall(r"\d+%@(\d+)", s["raw"].split("CPU [", 1)[-1].split("]", 1)[0])
    ]
    if cpu_clocks:
        result["cpu_clock_mhz_mean"] = sum(cpu_clocks) / len(cpu_clocks)
    energy = 0.0
    covered = 0.0
    for first, second in zip(selected, selected[1:]):
        a, b = max(start, first["monotonic"]), min(end, second["monotonic"])
        p = re.search(fields["power_w"], first["raw"])
        q = re.search(fields["power_w"], second["raw"])
        if b > a and p and q:
            dt = second["monotonic"] - first["monotonic"]
            left, right = float(p.group(1)), float(q.group(1))
            pa = left + (right - left) * (a - first["monotonic"]) / dt
            pb = left + (right - left) * (b - first["monotonic"]) / dt
            energy += (pa + pb) * 0.0005 * (b - a)
            covered += b - a
    result["telemetry_complete"] &= abs(covered - (end - start)) < 0.001
    result.update(
        board_energy_j=energy if result["telemetry_complete"] else None,
        power_coverage_seconds=covered,
        power_rail="VDD_IN",
        power_scope="complete test evaluation window including loading and metric updates",
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    rows = []
    cases = list(csv.DictReader((Path(__file__).resolve().parent / "models.csv").open()))
    for case in cases:
        model = args.run / case["model"]
        identity = model / "device.json"
        device = json.loads(identity.read_text())["device"] if identity.exists() else deployment_for(model.name)
        telemetry = model / "telemetry.jsonl"
        if not telemetry.exists() and model.is_dir():
            telemetry = args.run / "telemetry.jsonl"
        samples = [json.loads(line) for line in telemetry.open() if line.endswith("\n")] if telemetry.exists() else []
        timestamps = [s["monotonic"] for s in samples]
        for variant in ("onnx", "fp32", "fp16", "ptq", "qat"):
            directory = model / variant
            passes = [json.loads(p.read_text()) for p in sorted(directory.glob("pass[123].json"))]
            for trial in passes:
                trial["telemetry"] = telemetry_window(
                    samples, timestamps, trial["start_monotonic"], trial["end_monotonic"]
                )
                energy = trial["telemetry"].get("board_energy_j")
                trial["telemetry"]["board_energy_j_per_image"] = (
                    energy / trial["test_images"] if energy is not None else None
                )
            if directory.exists():
                (directory / "measurements.json").write_text(json.dumps(passes, indent=2) + "\n")
            for phase in ("build", "idle"):
                if (directory / (phase + ".json")).exists():
                    measurement = json.loads((directory / (phase + ".json")).read_text())
                    measurement["telemetry"] = telemetry_window(
                        samples, timestamps, measurement["start_monotonic"], measurement["end_monotonic"]
                    )
                    measurement["telemetry"]["power_scope"] = phase
                    (directory / (phase + "-measurements.json")).write_text(json.dumps(measurement, indent=2) + "\n")
            row = {
                "model": model.name,
                "device": device,
                "variant": variant,
                "status": "pending",
                "passes": len(passes),
                "map50": None,
                "map50_95": None,
                "inference_fps": None,
                "pipeline_fps": None,
                "reference_gflops_per_image": None,
                "effective_gops_per_second": None,
                "power_mean_w": None,
                "power_peak_w": None,
                "board_energy_j_per_image": None,
                "ram_peak_mb": None,
                "gpu_mean_pct": None,
                "cpu_mean_pct": None,
                "process_peak_rss_mib": None,
                "telemetry_complete": None,
            }
            if passes:
                seconds = sum(p["speed_ms_per_image"]["inference"] * p["test_images"] / 1000 for p in passes)
                count = sum(p["test_images"] for p in passes)
                pipe_seconds = sum(
                    sum(p["speed_ms_per_image"][k] for k in ("preprocess", "inference", "postprocess"))
                    * p["test_images"]
                    / 1000
                    for p in passes
                )
                row.update(
                    status="passed" if len(passes) == 3 else "partial",
                    map50=sum(p["metrics"]["metrics/mAP50(B)"] for p in passes) / len(passes),
                    map50_95=sum(p["metrics"]["metrics/mAP50-95(B)"] for p in passes) / len(passes),
                    inference_fps=count / seconds,
                    pipeline_fps=count / pipe_seconds,
                )
                if any(p["test_images"] != 4579 for p in passes):
                    row["status"] = "smoke_only"
                row["process_peak_rss_mib"] = max(p["process_peak_rss_kib"] for p in passes) / 1024
                row["reference_gflops_per_image"] = passes[0]["reference_gflops_per_image"]
                row["telemetry_complete"] = all(p["telemetry"].get("telemetry_complete") for p in passes)
                cpu = [
                    sum(p["telemetry"]["cpu_mean_pct_per_core"]) / len(p["telemetry"]["cpu_mean_pct_per_core"])
                    for p in passes
                    if "cpu_mean_pct_per_core" in p["telemetry"]
                ]
                row["cpu_mean_pct"] = sum(cpu) / len(cpu) if cpu else None
                if passes[0]["reference_gflops_per_image"]:
                    row["effective_gops_per_second"] = passes[0]["reference_gflops_per_image"] * count / seconds
                for key, field, reducer in (
                    ("power_mean_w", "power_w_mean", lambda v: sum(v) / len(v)),
                    ("power_peak_w", "power_w_peak", max),
                    ("ram_peak_mb", "ram_mb_peak", max),
                    ("gpu_mean_pct", "gpu_pct_mean", lambda v: sum(v) / len(v)),
                ):
                    values = [p["telemetry"][field] for p in passes if field in p["telemetry"]]
                    row[key] = reducer(values) if values else None
                if all(p["telemetry"].get("telemetry_complete") for p in passes):
                    total_energy = sum(p["telemetry"]["board_energy_j"] for p in passes)
                    row["board_energy_j_per_image"] = total_energy / count
                    row["power_mean_w"] = total_energy / sum(p["end_monotonic"] - p["start_monotonic"] for p in passes)
            for stage in ("build", "evaluate", "profile"):
                status = directory / (stage + ".exit")
                if status.exists() and status.read_text().strip() != "0":
                    log = (directory / (stage + ".log")).read_text(errors="replace")
                    phase = "runtime" if stage == "evaluate" else stage
                    if re.search(r"out of memory|outofmemory|cudaErrorMemoryAllocation|bad_alloc", log, re.I):
                        row["status"] = phase + "_oom"
                    elif "No space left on device" in log:
                        row["status"] = "storage_exhausted"
                    elif re.search(r"No importer registered for op|UNSUPPORTED_NODE", log):
                        row["status"] = "unsupported_operator"
                    else:
                        row["status"] = phase + "_failed"
                    break
            source_phase = variant if variant in {"ptq", "qat"} else "fp32"
            status = model / ("prepare-" + source_phase + ".exit")
            if (model / "preparation-failed.json").exists() or (status.exists() and status.read_text().strip() != "0"):
                row["status"] = "preparation_failed"
            rows.append(row)
    if rows:
        with (args.run / "results.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        (args.run / "status.json").write_text(
            json.dumps(
                {
                    "cases": len(rows),
                    "statuses": dict(Counter(row["status"] for row in rows)),
                    "archived_models": sum((args.run / case["model"] / "archived.json").exists() for case in cases),
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()

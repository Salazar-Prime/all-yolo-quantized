"""Evaluate the 56 Exp2/Exp2.5 best checkpoints on Rainbow's common seed-42 test set."""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.metadata
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "examples/QAT720-Exp1-Manifest"))

from train import (
    addEvaluationArguments,
    addManifestArguments,
    evaluateModel,
    hashFile,
    loadDatasetSplits,
    prepareDataset,
)


def summarize(project: Path, cases: list[dict]) -> list[dict]:
    """Write one comparison table from completed per-checkpoint measurements."""
    rows = []
    for case in cases:
        path = project / case["experiment"] / case["model"] / "metrics.json"
        if path.is_file():
            record = json.loads(path.read_text())
            metrics = record["metrics"]
            rows.append(
                {
                    "experiment": case["experiment"],
                    "model": case["model"],
                    "map50": metrics["metrics/mAP50(B)"],
                    "map50_95": metrics["metrics/mAP50-95(B)"],
                    "gt_coverage_percent": metrics["gt/gtCoveragePercent"],
                    "gt_detected": metrics["gt/gtDetected"],
                    "gt_total": metrics["gt/gtTotal"],
                    "gt_missed": metrics["gt/gtTotal"] - metrics["gt/gtDetected"],
                    "extra_predictions": metrics["gt/extraPredictions"],
                    "discarded_duplicates": metrics["gt/discardedDuplicatePredictions"],
                    "elapsed_seconds": record["elapsed_seconds"],
                    "checkpoint_sha256": case["sha256"],
                }
            )
    if not rows:
        return rows
    with (project / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Experiment 3: GT coverage and mAP@0.5",
        "",
        f"Completed checkpoints: {len(rows)}/{len(cases)}. Both groups use the same held-out test images.",
        "",
        "GT matching: confidence >= 0.25, IoU > 0.5, same class, highest IoU first; one prediction per GT.",
        "Conflicting duplicates are excluded from extra boxes. mAP@0.5 uses the standard low-confidence validator pass.",
        "mAP is stored on a 0–1 scale in CSV and shown as a percentage below, alongside GT coverage.",
        "",
        "| Training | Model | mAP@0.5 (%) | GT detected (%) | Detected / total | Missed | Extras | Duplicates |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['experiment']} | {row['model']} | {100 * row['map50']:.3f} | "
            f"{row['gt_coverage_percent']:.3f} | {row['gt_detected']} / {row['gt_total']} | "
            f"{row['gt_missed']} | {row['extra_predictions']} | {row['discarded_duplicates']} |"
        )
    (project / "results.md").write_text("\n".join(lines) + "\n")
    return rows


def main() -> None:
    """Prepare the portable manifest dataset and run the smoke or full matrix on Rainbow GPU 1."""
    parser = argparse.ArgumentParser(description=__doc__)
    addManifestArguments(parser)
    addEvaluationArguments(parser)
    parser.set_defaults(datasetRoot=HERE / "inputs/images", preparedDir=HERE / "prepared/full", gtCoverage=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="Use four positive and four background images per split.")
    parser.add_argument(
        "--start-index", type=int, default=0, help="Resume at this zero-based checkpoint inventory row."
    )
    parser.add_argument("--batch-size", dest="batchSize", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if socket.gethostname().split(".")[0] != "digital-ag" or os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        parser.error("Exp3 must run on Rainbow with CUDA_VISIBLE_DEVICES=1")
    if args.gtConf != 0.25 or args.gtIou != 0.5:
        parser.error("This experiment's fixed protocol requires --gt-conf 0.25 --gt-iou 0.5")
    if args.objectSize is not None or args.skipEmptyImages:
        parser.error("Exp3 uses all object sizes and includes background images for both training groups")

    import torch

    from ultralytics import YOLO, settings

    torch.set_num_threads(16)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Rainbow GPU 1 is not available as the sole visible CUDA device")
    settings.update({"wandb": False, "tensorboard": False, "sync": False})
    cases = list(csv.DictReader((HERE / "checkpoints.csv").open()))
    if not 0 <= args.start_index < len(cases):
        parser.error("--start-index must identify an inventory row")
    args.imageSize, args.device, args.existOk, args.disablePlots = 640, "0", True, True
    args.project = args.output.resolve() / ("smoke" if args.smoke else "full")
    args.project.mkdir(parents=True, exist_ok=True)
    dataYaml, statistics = prepareDataset(args)
    if args.prepareOnly:
        return
    if args.smoke:
        splits, _, _, _ = loadDatasetSplits(args.manifest.resolve(), args.splitSeed, args.splitRatios)
        selected = {
            split: [record for record in records if record.get("objects")][:4]
            + [record for record in records if not record.get("objects")][:4]
            for split, records in splits.items()
        }
        args.manifest = HERE / "prepared/smoke-manifest.json"
        args.manifest.write_text(json.dumps({"names": ["weed"], "splits": selected}, indent=2) + "\n")
        args.preparedDir = HERE / "prepared/smoke"
        dataYaml, statistics = prepareDataset(args)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    provenance = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "git_revision": revision,
        "hostname": socket.gethostname(),
        "physical_gpu": 1,
        "gpu_name": torch.cuda.get_device_name(0),
        "python": sys.executable,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "dataset_statistics": statistics,
        "manifest_sha256": {
            str(path.relative_to(HERE)): hashFile(path) for path in sorted((HERE / "inputs/manifests").glob("*.json"))
        },
        "inventory_sha256": hashFile(HERE / "checkpoints.csv"),
        "image_size": 640,
        "batch_size": args.batchSize,
        "confidence_threshold": args.gtConf,
        "iou_threshold": args.gtIou,
    }
    (args.project / f"provenance-{os.getpid()}.json").write_text(json.dumps(provenance, indent=2) + "\n")
    for index, case in enumerate(cases[args.start_index :], args.start_index):
        print(f"CASE {index}/{len(cases) - 1}: {case['experiment']}/{case['model']}", flush=True)
        checkpoint = HERE / case["checkpoint"]
        if hashFile(checkpoint) != case["sha256"]:
            raise ValueError(f"Checkpoint does not match its recovered training inventory: {checkpoint}")
        started = time.monotonic()
        model = YOLO(str(checkpoint))
        metrics = evaluateModel(model, args, dataYaml, "test", f"{case['experiment']}/{case['model']}")
        if metrics["gt/gtTotal"] != statistics["test"]["objects"]:
            raise ValueError("The coverage denominator differs from the prepared test manifest")
        record = {
            **case,
            "git_revision": revision,
            "metrics": metrics,
            "elapsed_seconds": time.monotonic() - started,
            "speed_ms_per_image": model.metrics.speed,
        }
        path = args.project / case["experiment"] / case["model"] / "metrics.json"
        path.write_text(json.dumps(record, indent=2) + "\n")
        summarize(args.project, cases)
        print(f"COMPLETED {index}: mAP50={metrics['metrics/mAP50(B)']:.6f}, GT={metrics['gt/gtCoveragePercent']:.3f}%")
        del model
        gc.collect()
        torch.cuda.empty_cache()
    rows = summarize(args.project, cases)
    if len(rows) != len(cases):
        raise RuntimeError(f"Only {len(rows)}/{len(cases)} checkpoints have results")
    print(f"COMPLETE: {len(rows)} checkpoints; {args.project / 'results.md'}", flush=True)


if __name__ == "__main__":
    main()

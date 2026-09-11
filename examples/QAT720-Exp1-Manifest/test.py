# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Evaluate a YOLO checkpoint on validation and/or test records from an Exp1 manifest."""

import argparse
import json
from pathlib import Path

from train import SCRIPT_DIR, addManifestArguments, prepareDataset, printStatistics


def parseArguments() -> argparse.Namespace:
    """Parse dataset preparation and evaluation arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    addManifestArguments(parser)
    parser.add_argument("--model", required=True, help="Detection checkpoint to evaluate.")
    parser.add_argument("--splits", nargs="+", choices=("val", "test"), default=("val", "test"))
    parser.add_argument("--image-size", dest="imageSize", type=int, default=640)
    parser.add_argument("--batch-size", dest="batchSize", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", type=Path, default=SCRIPT_DIR / "runs" / "test")
    parser.add_argument("--name", default=None)
    parser.add_argument("--exist-ok", dest="existOk", action="store_true")
    parser.add_argument("--disable-plots", dest="disablePlots", action="store_true")
    return parser.parse_args()


def serializableMetrics(results) -> dict:
    """Return numeric metrics from one Ultralytics validation result."""
    metrics = {}
    for key, value in results.results_dict.items():
        try:
            metrics[key] = float(value)
        except (TypeError, ValueError):
            continue
    return metrics


def main() -> None:
    """Prepare the requested data and evaluate the selected checkpoint."""
    args = parseArguments()
    dataYaml, statistics = prepareDataset(args)
    print(f"Ultralytics data config: {dataYaml}")
    printStatistics(statistics, args.objectSize, args.skipEmptyImages)
    if args.prepareOnly:
        print("Preparation-only mode: no model was loaded and no evaluation was started.")
        return

    from ultralytics import YOLO

    model = YOLO(args.model)
    runName = args.name or f"{Path(args.model).stem}_exp1"
    evaluationArguments = {
        "data": str(dataYaml),
        "imgsz": args.imageSize,
        "batch": args.batchSize,
        "workers": args.workers,
        "project": str(args.project.expanduser().resolve()),
        "exist_ok": args.existOk,
        "plots": not args.disablePlots,
    }
    if args.device is not None:
        evaluationArguments["device"] = args.device

    metrics = {}
    for splitName in args.splits:
        results = model.val(split=splitName, name=f"{runName}_{splitName}", **evaluationArguments)
        metrics[splitName] = serializableMetrics(results)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

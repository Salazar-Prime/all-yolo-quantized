# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Run YOLO detection on one prepared split selected from an Exp1 manifest."""

import argparse
from pathlib import Path

from train import SCRIPT_DIR, addManifestArguments, prepareDataset, printStatistics


def parseArguments() -> argparse.Namespace:
    """Parse dataset preparation and prediction arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    addManifestArguments(parser)
    parser.add_argument("--model", required=True, help="Detection checkpoint to use.")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--image-size", dest="imageSize", type=int, default=640)
    parser.add_argument("--batch-size", dest="batchSize", type=int, default=16)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", dest="maxDet", type=int, default=300)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", type=Path, default=SCRIPT_DIR / "runs" / "detect")
    parser.add_argument("--name", default=None)
    parser.add_argument("--exist-ok", dest="existOk", action="store_true")
    parser.add_argument("--save-txt", dest="saveTxt", action="store_true")
    parser.add_argument("--save-conf", dest="saveConf", action="store_true")
    parser.add_argument("--no-save", dest="noSave", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Prepare the requested split and run prediction on its image directory."""
    args = parseArguments()
    dataYaml, statistics = prepareDataset(args)
    print(f"Ultralytics data config: {dataYaml}")
    printStatistics(statistics, args.objectSize, args.skipEmptyImages)
    if args.prepareOnly:
        print("Preparation-only mode: no model was loaded and no detection was started.")
        return

    from ultralytics import YOLO

    source = dataYaml.parent / "images" / args.split
    runName = args.name or f"{Path(args.model).stem}_exp1_{args.split}"
    predictArguments = {
        "source": str(source),
        "imgsz": args.imageSize,
        "batch": args.batchSize,
        "conf": args.conf,
        "iou": args.iou,
        "max_det": args.maxDet,
        "project": str(args.project.expanduser().resolve()),
        "name": runName,
        "exist_ok": args.existOk,
        "save": not args.noSave,
        "save_txt": args.saveTxt,
        "save_conf": args.saveConf,
    }
    if args.device is not None:
        predictArguments["device"] = args.device
    results = YOLO(args.model).predict(**predictArguments)
    print(f"Processed {len(results)} images from {args.split} split.")


if __name__ == "__main__":
    main()

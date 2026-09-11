# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Prepare Exp1 manifest data and run a YOLO train/validation/test experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import uuid
from pathlib import Path
from typing import Sequence

from splitDatasetManifests import loadMetadata, splitRecords, validateRatios

SCRIPT_DIR = Path(__file__).resolve().parent
SPLIT_NAMES = ("train", "val", "test")
SIZE_ALIASES = {
    "all": None,
    "s": "small",
    "small": "small",
    "m": "medium",
    "medium": "medium",
    "l": "large",
    "large": "large",
}
PREPARATION_MARKER = ".manifestPreparation.json"


def normalizeObjectSize(value: str) -> str | None:
    """Return the canonical COCO size name selected on the command line."""
    normalized = SIZE_ALIASES.get(value.strip().lower())
    if value.strip().lower() not in SIZE_ALIASES:
        raise argparse.ArgumentTypeError("object size must be one of: all, small/S, medium/M, large/L")
    return normalized


def loadJson(path: Path):
    """Load a JSON file and provide a path-specific parse error."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Manifest is not valid JSON: {path} ({error})")


def loadReferencedSplit(value, splitName: str, manifestPath: Path):
    """Load one inline split or one split manifest referenced by the input file."""
    if isinstance(value, list):
        records = value
        sourcePath = None
    elif isinstance(value, str) and value:
        sourcePath = Path(value).expanduser()
        if not sourcePath.is_absolute():
            sourcePath = manifestPath.parent / sourcePath
        sourcePath = sourcePath.resolve()
        if not sourcePath.is_file():
            raise FileNotFoundError(f"{splitName} manifest does not exist: {sourcePath}")
        records = loadMetadata(sourcePath)
    else:
        raise ValueError(f"{splitName} split must be a list of records or a manifest path")

    validateRecords(records, f"{splitName} split")
    return records, sourcePath


def validateRecords(records, description: str) -> None:
    """Validate the image-record structure needed by this training adapter."""
    if not isinstance(records, list):
        raise TypeError(f"{description} must contain a JSON list")

    seenPaths = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise TypeError(f"{description} record {index} is not a JSON object")
        filePath = record.get("filePath")
        fileName = record.get("fileName")
        if (not isinstance(filePath, str) or not filePath) and (not isinstance(fileName, str) or not fileName):
            raise ValueError(f"{description} record {index} has no valid filePath or fileName")
        identity = filePath or fileName
        if identity in seenPaths:
            raise ValueError(f"Duplicate image in {description}: {identity}")
        seenPaths.add(identity)

        objects = record.get("objects", [])
        if not isinstance(objects, list):
            raise TypeError(f"{description} record {index} has a non-list objects value")


def extractConfiguredClassNames(payload) -> list[str] | None:
    """Read class names from supported manifest/config shapes."""
    if not isinstance(payload, dict):
        return None

    classes = payload.get("classes")
    names = classes.get("names") if isinstance(classes, dict) else payload.get("names")
    if isinstance(names, dict):
        try:
            names = [names[str(index)] for index in range(len(names))]
        except KeyError:
            return None
    if isinstance(names, list) and all(isinstance(name, str) and name for name in names):
        return names
    return None


def loadDatasetSplits(
    manifestPath: Path,
    splitSeed: int,
    splitRatios: Sequence[float],
) -> tuple[dict[str, list[dict]], list[str] | None, list[Path], str]:
    """Load explicit splits or split one combined image-record manifest."""
    payload = loadJson(manifestPath)
    configuredNames = extractConfiguredClassNames(payload)
    sourcePaths = [manifestPath]

    if isinstance(payload, list):
        validateRecords(payload, "combined manifest")
        ratios = validateRatios(splitRatios)
        if any(value <= 0 for value in ratios):
            raise ValueError("Train, validation, and test ratios must all be greater than zero")
        splits = splitRecords(payload, splitSeed, ratios)
        inputMode = "combined"
    elif isinstance(payload, dict):
        splitMapping = payload.get("splits")
        if splitMapping is None:
            splitMapping = payload.get("manifests")
        if not isinstance(splitMapping, dict):
            raise TypeError("JSON object input must contain a 'splits' or 'manifests' mapping")

        splits = {}
        for splitName in SPLIT_NAMES:
            if splitName not in splitMapping:
                raise ValueError(f"Input manifest has no '{splitName}' split")
            records, sourcePath = loadReferencedSplit(splitMapping[splitName], splitName, manifestPath)
            splits[splitName] = records
            if sourcePath is not None:
                sourcePaths.append(sourcePath)
        inputMode = "explicit"
    else:
        raise TypeError("Input must be a combined JSON list or an object with split manifests")

    validateSplitMembership(splits)
    return splits, configuredNames, sourcePaths, inputMode


def validateSplitMembership(splits: dict[str, list[dict]]) -> None:
    """Require nonempty, mutually disjoint train/validation/test splits."""
    seen = {}
    for splitName in SPLIT_NAMES:
        records = splits[splitName]
        if not records:
            raise ValueError(f"{splitName} split is empty")
        for record in records:
            identity = record.get("filePath") or record.get("fileName")
            previousSplit = seen.get(identity)
            if previousSplit is not None:
                raise ValueError(f"Image occurs in both {previousSplit} and {splitName} splits: {identity}")
            seen[identity] = splitName


def inferClassNames(
    splits: dict[str, list[dict]],
    requestedNames: Sequence[str] | None,
    configuredNames: Sequence[str] | None,
) -> list[str]:
    """Choose class names and verify every class ID can be represented."""
    classIds = set()
    for records in splits.values():
        for record in records:
            for obj in record.get("objects", []):
                classId = parseClassId(obj)
                classIds.add(classId)

    if not classIds:
        raise ValueError("The input manifest contains no objects")

    names = list(requestedNames or configuredNames or [])
    if not names:
        names = [f"class_{index}" for index in range(max(classIds) + 1)]
    if max(classIds) >= len(names):
        raise ValueError(
            f"Manifest class ID {max(classIds)} requires at least {max(classIds) + 1} class names, but {len(names)} were provided"
        )
    return names


def parseClassId(obj: dict) -> int:
    """Return a validated nonnegative integer class ID."""
    value = obj.get("classId")
    if isinstance(value, bool):
        raise TypeError("Object classId must be a nonnegative integer")
    try:
        classId = int(value)
    except (TypeError, ValueError):
        raise ValueError("Object classId must be a nonnegative integer")
    if classId < 0 or value != classId:
        raise ValueError("Object classId must be a nonnegative integer")
    return classId


def yoloLabelLine(obj: dict) -> str:
    """Validate and serialize one object as a YOLO detection label."""
    classId = parseClassId(obj)
    values = []
    for key in ("xCenter", "yCenter", "boxWidth", "boxHeight"):
        try:
            value = float(obj.get(key))
        except (TypeError, ValueError):
            raise ValueError(f"Object {key} must be numeric")
        if not math.isfinite(value):
            raise ValueError(f"Object {key} must be finite")
        if key in ("boxWidth", "boxHeight"):
            if value <= 0 or value > 1:
                raise ValueError(f"Object {key} must be in (0, 1]")
        elif value < 0 or value > 1:
            raise ValueError(f"Object {key} must be in [0, 1]")
        values.append(value)
    return "{} {} {} {} {}\n".format(classId, *(format(value, ".10g") for value in values))


def selectObjects(objects: Sequence[dict], objectSize: str | None) -> list[dict]:
    """Return all objects or only objects in one canonical COCO size category."""
    if objectSize is None:
        return list(objects)

    selected = []
    for obj in objects:
        category = obj.get("cocoSizeCategory")
        normalizedCategory = category.strip().lower() if isinstance(category, str) else ""
        if normalizedCategory not in (
            "small",
            "medium",
            "large",
        ):
            raise ValueError("Every object must have a valid cocoSizeCategory when --object-size is used")
        if normalizedCategory == objectSize:
            selected.append(obj)
    return selected


def resolveImagePath(record: dict, datasetRoot: Path | None, manifestPath: Path) -> Path:
    """Resolve a record image, allowing an explicit root to replace stale paths."""
    filePathValue = record.get("filePath")
    fileNameValue = record.get("fileName")
    candidates = []

    if datasetRoot is not None and isinstance(fileNameValue, str) and fileNameValue:
        candidates.append(datasetRoot / Path(fileNameValue).name)

    if isinstance(filePathValue, str) and filePathValue:
        filePath = Path(filePathValue).expanduser()
        if filePath.is_absolute():
            candidates.append(filePath)
        else:
            if datasetRoot is not None:
                candidates.append(datasetRoot / filePath)
            candidates.append(manifestPath.parent / filePath)

    if isinstance(fileNameValue, str) and fileNameValue:
        candidates.append(manifestPath.parent / fileNameValue)

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved

    raise FileNotFoundError(
        f"Image from manifest record was not found (filePath={filePathValue!r}, fileName={fileNameValue!r}); "
        "use --dataset-root if the dataset moved"
    )


def hashFile(path: Path) -> str:
    """Return a streaming SHA-256 digest for one source manifest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def preparationSignature(
    sourcePaths: Sequence[Path],
    inputMode: str,
    splitSeed: int,
    splitRatios: Sequence[float],
    objectSize: str | None,
    skipEmptyImages: bool,
    datasetRoot: Path | None,
    classNames: Sequence[str],
) -> tuple[str, dict]:
    """Build the reproducibility metadata and its stable signature."""
    specification = {
        "formatVersion": 2,
        "inputMode": inputMode,
        "sourceManifests": [
            {"path": str(path), "sha256": hashFile(path)} for path in sorted(set(sourcePaths), key=str)
        ],
        "splitSeed": splitSeed if inputMode == "combined" else None,
        "splitRatios": list(splitRatios) if inputMode == "combined" else None,
        "objectSize": objectSize or "all",
        "skipEmptyImages": skipEmptyImages,
        "datasetRoot": str(datasetRoot) if datasetRoot is not None else None,
        "classNames": list(classNames),
    }
    encoded = json.dumps(specification, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), specification


def writeDataYaml(path: Path, datasetPath: Path, classNames: Sequence[str]) -> None:
    """Write the standard Ultralytics detection dataset configuration."""
    lines = [
        f"path: {json.dumps(str(datasetPath.resolve()))}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    lines.extend(f"  {index}: {json.dumps(name)}" for index, name in enumerate(classNames))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def buildPreparedDataset(
    outputPath: Path,
    splits: dict[str, list[dict]],
    manifestPath: Path,
    datasetRoot: Path | None,
    objectSize: str | None,
    skipEmptyImages: bool,
    classNames: Sequence[str],
    signature: str,
    specification: dict,
) -> dict:
    """Build a lightweight Ultralytics dataset using symlinked images."""
    temporaryPath = outputPath.with_name(f".{outputPath.name}.tmp-{uuid.uuid4().hex}")
    statistics = {}

    try:
        for splitName in SPLIT_NAMES:
            imageDir = temporaryPath / "images" / splitName
            labelDir = temporaryPath / "labels" / splitName
            imageDir.mkdir(parents=True, exist_ok=True)
            labelDir.mkdir(parents=True, exist_ok=True)

            loadedImages = 0
            loadedObjects = 0
            skippedImages = 0
            for recordIndex, record in enumerate(splits[splitName]):
                objects = record.get("objects", [])
                selectedObjects = selectObjects(objects, objectSize)
                if (objectSize is not None and not selectedObjects) or (
                    objectSize is None and skipEmptyImages and not objects
                ):
                    skippedImages += 1
                    continue

                imagePath = resolveImagePath(record, datasetRoot, manifestPath)
                outputStem = f"{recordIndex:08d}_{imagePath.stem}"
                outputImagePath = imageDir / (outputStem + imagePath.suffix)
                outputLabelPath = labelDir / (outputStem + ".txt")
                outputImagePath.symlink_to(imagePath)
                outputLabelPath.write_text(
                    "".join(yoloLabelLine(obj) for obj in selectedObjects),
                    encoding="utf-8",
                )
                loadedImages += 1
                loadedObjects += len(selectedObjects)

            if loadedImages == 0:
                raise ValueError(f"{splitName} split has no images after filtering")
            if loadedObjects == 0:
                raise ValueError(f"{splitName} split has no objects after filtering")
            statistics[splitName] = {
                "images": loadedImages,
                "objects": loadedObjects,
                "skippedImages": skippedImages,
            }

        writeDataYaml(temporaryPath / "data.yaml", outputPath, classNames)
        marker = {
            "generatedBy": "examples/QAT720-Exp1-Manifest/train.py",
            "signature": signature,
            "specification": specification,
            "statistics": statistics,
        }
        (temporaryPath / PREPARATION_MARKER).write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporaryPath.replace(outputPath)
    finally:
        if temporaryPath.exists():
            shutil.rmtree(str(temporaryPath))

    return statistics


def prepareDataset(args) -> tuple[Path, dict]:
    """Load one manifest input and create or reuse its YOLO dataset view."""
    manifestPath = args.manifest.expanduser().resolve()
    if not manifestPath.is_file():
        raise FileNotFoundError(f"Input manifest does not exist: {manifestPath}")

    datasetRoot = args.datasetRoot.expanduser().resolve() if args.datasetRoot is not None else None
    if datasetRoot is not None and not datasetRoot.is_dir():
        raise NotADirectoryError(f"Dataset root does not exist: {datasetRoot}")

    splitRatios = validateRatios(args.splitRatios)
    splits, configuredNames, sourcePaths, inputMode = loadDatasetSplits(manifestPath, args.splitSeed, splitRatios)
    classNames = inferClassNames(splits, args.classNames, configuredNames)
    signature, specification = preparationSignature(
        sourcePaths,
        inputMode,
        args.splitSeed,
        splitRatios,
        args.objectSize,
        args.skipEmptyImages,
        datasetRoot,
        classNames,
    )

    inputTag = f"seed_{args.splitSeed}" if inputMode == "combined" else "explicit"
    preparedSelection = args.objectSize or "all"
    if args.skipEmptyImages:
        preparedSelection += "_skip_empty"
    outputPath = (
        args.preparedDir.expanduser().resolve()
        if args.preparedDir is not None
        else (SCRIPT_DIR / "prepared" / manifestPath.stem / inputTag / preparedSelection)
    )
    markerPath = outputPath / PREPARATION_MARKER

    if outputPath.exists():
        existingMarker = loadJson(markerPath) if markerPath.is_file() else None
        if (
            isinstance(existingMarker, dict)
            and existingMarker.get("signature") == signature
            and (outputPath / "data.yaml").is_file()
            and not args.rebuildPrepared
        ):
            print(f"Reusing prepared dataset: {outputPath}")
            return outputPath / "data.yaml", existingMarker["statistics"]

        if not args.rebuildPrepared:
            raise FileExistsError(
                f"Prepared directory exists but does not match this request: {outputPath}. "
                "Use --rebuild-prepared to replace it."
            )
        if (
            not isinstance(existingMarker, dict)
            or existingMarker.get("generatedBy") != "examples/QAT720-Exp1-Manifest/train.py"
        ):
            raise ValueError(f"Refusing to replace an unrecognized directory: {outputPath}")
        shutil.rmtree(str(outputPath))

    outputPath.parent.mkdir(parents=True, exist_ok=True)
    statistics = buildPreparedDataset(
        outputPath,
        splits,
        manifestPath,
        datasetRoot,
        args.objectSize,
        args.skipEmptyImages,
        classNames,
        signature,
        specification,
    )
    print(f"Prepared dataset: {outputPath}")
    return outputPath / "data.yaml", statistics


def printStatistics(
    statistics: dict,
    objectSize: str | None,
    skipEmptyImages: bool,
) -> None:
    """Print the exact inputs that the three native dataloaders will receive."""
    print("Object size: {}".format(objectSize or "all"))
    print("Skip empty images in train/val/test: {}".format("true" if skipEmptyImages else "false"))
    for splitName in SPLIT_NAMES:
        values = statistics[splitName]
        print(
            "{}: {} images, {} objects, {} images filtered out".format(
                splitName,
                values["images"],
                values["objects"],
                values["skippedImages"],
            )
        )


def runExperiment(args, dataYaml: Path) -> None:
    """Train YOLO, then evaluate its best checkpoint on val and test."""
    try:
        from ultralytics import YOLO, settings
    except ImportError as error:
        raise RuntimeError("Ultralytics dependencies are unavailable. Install this fork before training.") from error

    runName = args.name or "{}_{}_{}".format(Path(args.model).stem, args.manifest.stem, args.objectSize or "all")
    project = args.project.expanduser().resolve()
    wandbEnabled = args.wandbMode != "disabled"
    wandbModule = None
    settings.update({"tensorboard": True, "wandb": wandbEnabled})
    if wandbEnabled:
        os.environ.pop("WANDB_DISABLED", None)
        os.environ["WANDB_MODE"] = args.wandbMode
        os.environ["WANDB_PROJECT"] = args.wandbProject
        os.environ["WANDB_NAME"] = runName
        os.environ["WANDB_JOB_TYPE"] = "training"
        if args.wandbEntity:
            os.environ["WANDB_ENTITY"] = args.wandbEntity
        try:
            import wandb as wandbModule
        except ImportError as error:
            raise RuntimeError(
                "W&B logging is enabled but the wandb package is unavailable. "
                "Install the optional W&B dependency or pass --wandb-mode disabled."
            ) from error
    print(
        "Experiment logging: TensorBoard=enabled, W&B={} (project={}, mode={})".format(
            "enabled" if wandbEnabled else "disabled",
            args.wandbProject,
            args.wandbMode,
        )
    )

    trainArguments = {
        "data": str(dataYaml),
        "epochs": args.epochs,
        "imgsz": args.imageSize,
        "batch": args.batchSize,
        "workers": args.workers,
        "seed": args.trainingSeed,
        "deterministic": True,
        "patience": args.patience,
        "project": str(project),
        "name": runName,
        "exist_ok": args.existOk,
        "plots": not args.disablePlots,
    }
    if args.device is not None:
        trainArguments["device"] = args.device

    if wandbModule is not None:
        wandbRun = wandbModule.init(
            project=args.wandbProject,
            entity=args.wandbEntity,
            name=runName,
            config=trainArguments,
            mode=args.wandbMode,
            dir=str(project),
            sync_tensorboard=True,
        )
        print(
            "W&B run initialized explicitly: project={}, run={}, url={}".format(
                wandbRun.project,
                wandbRun.name,
                wandbRun.url or "offline",
            )
        )

    model = YOLO(args.model)
    model.train(**trainArguments)

    bestCheckpoint = Path(model.trainer.best)
    if not bestCheckpoint.is_file():
        bestCheckpoint = Path(model.trainer.last)
    if not bestCheckpoint.is_file():
        raise FileNotFoundError("Training completed without a saved checkpoint")

    evaluationModel = YOLO(str(bestCheckpoint))
    evaluationArguments = {
        "data": str(dataYaml),
        "imgsz": args.imageSize,
        "batch": args.batchSize,
        "workers": args.workers,
        "project": str(project),
        "exist_ok": args.existOk,
        "plots": not args.disablePlots,
    }
    if args.device is not None:
        evaluationArguments["device"] = args.device

    finalValidation = evaluationModel.val(split="val", name=runName + "_final_val", **evaluationArguments)
    finalTest = evaluationModel.val(split="test", name=runName + "_final_test", **evaluationArguments)

    if wandbModule is not None:
        finalRun = wandbModule.run
        if finalRun is None:
            finalRun = wandbModule.init(
                project=args.wandbProject,
                entity=args.wandbEntity,
                name=runName,
                id=wandbRun.id,
                resume="allow",
                mode=args.wandbMode,
                dir=str(project),
            )
        finalMetrics = {}
        for prefix, metrics in (
            ("final_val", finalValidation),
            ("final_test", finalTest),
        ):
            for key, value in metrics.results_dict.items():
                try:
                    finalMetrics[f"{prefix}/{key}"] = float(value)
                except (TypeError, ValueError):
                    continue
        finalRun.log(finalMetrics)
        finalRun.finish()


def addManifestArguments(parser: argparse.ArgumentParser) -> None:
    """Add the shared Exp1 manifest conversion arguments to a parser."""
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help=(
            "One combined image-record manifest, or one JSON config whose "
            "'splits'/'manifests' mapping defines train, val, and test."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        dest="datasetRoot",
        type=Path,
        default=None,
        help="Replacement image root for manifests containing stale filePath values.",
    )
    parser.add_argument(
        "--object-size",
        dest="objectSize",
        type=normalizeObjectSize,
        default=None,
        metavar="{all,small/S,medium/M,large/L}",
        help=(
            "Load only objects in this COCO size category. Images without a matching object are excluded. Default: all."
        ),
    )
    parser.add_argument(
        "--class-names",
        dest="classNames",
        nargs="+",
        default=None,
        help=("Class names in class-ID order. Defaults to names in a config input, then class_0, class_1, etc."),
    )
    parser.add_argument(
        "--split-seed",
        dest="splitSeed",
        type=int,
        default=42,
        help="Seed used when a combined list manifest must be split. Default: 42.",
    )
    parser.add_argument(
        "--split-ratios",
        dest="splitRatios",
        type=float,
        nargs=3,
        metavar=("TRAIN", "VAL", "TEST"),
        default=(30.0, 20.0, 50.0),
        help="Split weights for a combined list manifest. Default: 30 20 50.",
    )
    parser.add_argument(
        "--skip-empty-images",
        dest="skipEmptyImages",
        action="store_true",
        default=False,
        help=(
            "Exclude manifest records with no labels from train, validation, and "
            "test. Default: false, so background-only images are retained. "
            "Size-filtered runs also exclude images without a matching-size object "
            "to avoid treating other labeled objects as background."
        ),
    )
    parser.add_argument(
        "--prepared-dir",
        dest="preparedDir",
        type=Path,
        default=None,
        help="Prepared YOLO dataset directory. Defaults under this example's prepared/ directory.",
    )
    parser.add_argument(
        "--rebuild-prepared",
        dest="rebuildPrepared",
        action="store_true",
        help="Replace a prepared directory previously generated by this script.",
    )
    parser.add_argument(
        "--prepare-only",
        dest="prepareOnly",
        action="store_true",
        help="Prepare and validate data, then exit without loading or training a model.",
    )


def parseArguments() -> argparse.Namespace:
    """Parse manifest preparation and YOLO experiment arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    addManifestArguments(parser)

    parser.add_argument(
        "--model",
        default="yolo26n.pt",
        help="YOLO detection checkpoint or model YAML. Default: yolo26n.pt.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--image-size", dest="imageSize", type=int, default=640)
    parser.add_argument("--batch-size", dest="batchSize", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--training-seed", dest="trainingSeed", type=int, default=0)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--project",
        type=Path,
        default=SCRIPT_DIR / "runs",
        help="Training/evaluation result root. Default: this example's runs/ directory.",
    )
    parser.add_argument("--name", default=None)
    parser.add_argument("--exist-ok", dest="existOk", action="store_true")
    parser.add_argument(
        "--disable-plots",
        dest="disablePlots",
        action="store_true",
        help=(
            "Disable Ultralytics plot-image generation and logger uploads while "
            "retaining scalar metrics and TensorBoard logging."
        ),
    )
    parser.add_argument(
        "--wandb-mode",
        dest="wandbMode",
        choices=("offline", "online", "disabled"),
        default="offline",
        help=("W&B logging mode. TensorBoard remains enabled in every mode. Default: offline."),
    )
    parser.add_argument(
        "--wandb-project",
        dest="wandbProject",
        default="qat720-exp1-manifest",
        help="W&B project name. Default: qat720-exp1-manifest.",
    )
    parser.add_argument(
        "--wandb-entity",
        dest="wandbEntity",
        default=None,
        help="Optional W&B user or team entity. Defaults to the authenticated entity.",
    )
    return parser.parse_args()


def main() -> None:
    args = parseArguments()
    dataYaml, statistics = prepareDataset(args)
    print(f"Ultralytics data config: {dataYaml}")
    printStatistics(statistics, args.objectSize, args.skipEmptyImages)
    if args.prepareOnly:
        print("Preparation-only mode: no model was loaded and no training was started.")
        return
    runExperiment(args, dataYaml)


if __name__ == "__main__":
    main()

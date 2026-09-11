# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Split datasetMetadata.json into deterministic train, validation, and test manifests.

The split seed is the only required argument. By default, the script uses the
current day-0 corn dataset, a 30:20:50 train:val:test ratio, and writes three
JSON manifests under:

    <datasetPath>/manifests/seed_<seed>/

Each image record, including its complete nested object metadata, is assigned
to exactly one split. Detailed JSON statistics are generated for every split,
along with one combined manifestStatistics.csv comparison table.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Sequence

from manifestStatistics import createManifestStatistics, writeStatisticsCsv

DEFAULT_DATASET_PATH = Path(".")
DEFAULT_RATIOS = (30.0, 20.0, 50.0)
SPLIT_NAMES = ("train", "val", "test")
MANIFEST_FILENAMES = {
    "train": "trainManifest.json",
    "val": "valManifest.json",
    "test": "testManifest.json",
}


def validateRatios(ratios: Sequence[float]) -> tuple[float, float, float]:
    if len(ratios) != 3:
        raise ValueError("Exactly three ratios are required: train val test")

    parsedRatios = tuple(float(value) for value in ratios)
    if any(not math.isfinite(value) for value in parsedRatios):
        raise ValueError("Split ratios must be finite numbers")
    if any(value < 0 for value in parsedRatios):
        raise ValueError("Split ratios cannot be negative")
    if sum(parsedRatios) <= 0:
        raise ValueError("At least one split ratio must be greater than zero")
    return parsedRatios


def allocateSplitCounts(
    numberOfRecords: int,
    ratios: Sequence[float],
) -> tuple[int, int, int]:
    """
    Convert ratio weights into exact integer counts using largest remainders.

    The returned counts always sum to numberOfRecords. Ties are resolved in
    train, validation, then test order.
    """
    trainRatio, valRatio, testRatio = validateRatios(ratios)
    ratioValues = (trainRatio, valRatio, testRatio)
    ratioTotal = sum(ratioValues)
    exactCounts = [numberOfRecords * ratioValue / ratioTotal for ratioValue in ratioValues]
    counts = [math.floor(value) for value in exactCounts]
    remaining = numberOfRecords - sum(counts)

    remainderOrder = sorted(
        range(len(SPLIT_NAMES)),
        key=lambda index: (-(exactCounts[index] - counts[index]), index),
    )
    for index in remainderOrder[:remaining]:
        counts[index] += 1

    return counts[0], counts[1], counts[2]


def loadMetadata(metadataPath: Path) -> list[dict[str, object]]:
    try:
        payload = json.loads(metadataPath.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Input manifest is not valid JSON: {metadataPath} ({error})")

    if not isinstance(payload, list):
        raise TypeError("Input manifest must contain a JSON list of image records")

    records = []
    seenFilePaths = set()
    for index, record in enumerate(payload):
        if not isinstance(record, dict):
            raise TypeError(f"Manifest record {index} is not a JSON object")

        filePath = record.get("filePath")
        if not isinstance(filePath, str) or not filePath:
            raise ValueError(f"Manifest record {index} has no valid filePath")
        if filePath in seenFilePaths:
            raise ValueError(f"Duplicate filePath in input manifest: {filePath}")

        seenFilePaths.add(filePath)
        records.append(record)

    return records


def splitRecords(
    records: Sequence[dict[str, object]],
    seed: int,
    ratios: Sequence[float],
) -> dict[str, list[dict[str, object]]]:
    """
    Return deterministic, disjoint train/val/test record lists.

    Sorting by filePath before shuffling makes the result independent of the
    record order in the input JSON.
    """
    shuffledRecords = sorted(
        records,
        key=lambda record: str(record.get("filePath") or record.get("fileName")).casefold(),
    )
    random.Random(seed).shuffle(shuffledRecords)

    trainCount, valCount, testCount = allocateSplitCounts(
        len(shuffledRecords),
        ratios,
    )
    trainEnd = trainCount
    valEnd = trainEnd + valCount
    testEnd = valEnd + testCount

    if testEnd != len(shuffledRecords):
        raise RuntimeError("Internal error: split counts do not cover all records")

    return {
        "train": shuffledRecords[:trainEnd],
        "val": shuffledRecords[trainEnd:valEnd],
        "test": shuffledRecords[valEnd:testEnd],
    }


def validateSplits(
    inputRecords: Sequence[dict[str, object]],
    splitRecordsByName: dict[str, list[dict[str, object]]],
) -> None:
    inputPaths = {str(record["filePath"]) for record in inputRecords}
    outputPaths = set()

    for splitName in SPLIT_NAMES:
        splitPaths = {str(record["filePath"]) for record in splitRecordsByName[splitName]}
        if outputPaths.intersection(splitPaths):
            raise RuntimeError("The generated splits are not disjoint")
        outputPaths.update(splitPaths)

    if outputPaths != inputPaths:
        missingCount = len(inputPaths - outputPaths)
        unexpectedCount = len(outputPaths - inputPaths)
        raise RuntimeError(
            f"The generated splits do not match the input (missing={missingCount}, unexpected={unexpectedCount})"
        )


def writeManifestsAtomically(
    outputDir: Path,
    splitRecordsByName: dict[str, list[dict[str, object]]],
) -> dict[str, Path]:
    outputDir.mkdir(parents=True, exist_ok=True)
    temporaryPaths = {}
    finalPaths = {}

    try:
        for splitName in SPLIT_NAMES:
            finalPath = outputDir / MANIFEST_FILENAMES[splitName]
            temporaryPath = finalPath.with_name(finalPath.name + ".tmp")
            temporaryPath.write_text(
                json.dumps(splitRecordsByName[splitName], indent=2) + "\n",
                encoding="utf-8",
            )
            temporaryPaths[splitName] = temporaryPath
            finalPaths[splitName] = finalPath

        for splitName in SPLIT_NAMES:
            temporaryPaths[splitName].replace(finalPaths[splitName])
    finally:
        for temporaryPath in temporaryPaths.values():
            if temporaryPath.exists():
                temporaryPath.unlink()

    return finalPaths


def countObjects(records: Sequence[dict[str, object]]) -> int:
    objectCount = 0
    for record in records:
        objects = record.get("objects", [])
        if not isinstance(objects, list):
            raise TypeError("Record {} has an invalid objects value".format(record["filePath"]))
        objectCount += len(objects)
    return objectCount


def parseArguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "seed",
        type=int,
        help="Integer random seed used to assign image records to splits.",
    )
    parser.add_argument(
        "--datasetPath",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"Dataset root. Default: {DEFAULT_DATASET_PATH}",
    )
    parser.add_argument(
        "--metadataPath",
        type=Path,
        default=None,
        help="Input metadata JSON. Default: <datasetPath>/datasetMetadata.json.",
    )
    parser.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        metavar=("TRAIN", "VAL", "TEST"),
        default=DEFAULT_RATIOS,
        help="Train/val/test weights. Default: 30 20 50.",
    )
    parser.add_argument(
        "--outputDir",
        type=Path,
        default=None,
        help="Output directory. Default: <datasetPath>/manifests/seed_<seed>.",
    )
    return parser.parse_args()


def main() -> None:
    args = parseArguments()
    datasetPath = args.datasetPath.expanduser().resolve()
    metadataPath = (
        args.metadataPath.expanduser().resolve()
        if args.metadataPath is not None
        else datasetPath / "datasetMetadata.json"
    )
    outputDir = (
        args.outputDir.expanduser().resolve()
        if args.outputDir is not None
        else datasetPath / "manifests" / f"seed_{args.seed}"
    )
    ratios = validateRatios(args.ratios)

    if not datasetPath.is_dir():
        raise NotADirectoryError(f"Dataset directory does not exist: {datasetPath}")
    if not metadataPath.is_file():
        raise FileNotFoundError(f"Input metadata manifest does not exist: {metadataPath}")

    records = loadMetadata(metadataPath)
    splits = splitRecords(records, args.seed, ratios)
    validateSplits(records, splits)
    writtenPaths = writeManifestsAtomically(outputDir, splits)
    statisticsPaths = {splitName: createManifestStatistics(writtenPaths[splitName]) for splitName in SPLIT_NAMES}
    statisticsPayloads = {
        splitName: json.loads(statisticsPaths[splitName].read_text(encoding="utf-8")) for splitName in SPLIT_NAMES
    }
    combinedStatisticsCsvPath = writeStatisticsCsv(
        outputDir / "manifestStatistics.csv",
        statisticsPayloads,
        SPLIT_NAMES,
    )

    print(f"Input manifest: {metadataPath}")
    print(f"Split seed: {args.seed}")
    print(f"Requested train:val:test ratio: {ratios[0]:g}:{ratios[1]:g}:{ratios[2]:g}")
    print(f"Output directory: {outputDir}")

    totalImages = len(records)
    totalObjects = countObjects(records)
    for splitName in SPLIT_NAMES:
        splitRecordList = splits[splitName]
        splitImages = len(splitRecordList)
        splitObjects = countObjects(splitRecordList)
        imagePercent = 100.0 * splitImages / totalImages if totalImages else 0.0
        objectPercent = 100.0 * splitObjects / totalObjects if totalObjects else 0.0
        print(
            f"{splitName}: {splitImages} images ({imagePercent:.2f}%), {splitObjects} objects ({objectPercent:.2f}%) -> {writtenPaths[splitName]}"
        )
        print(f"{splitName} statistics -> {statisticsPaths[splitName]}")
    print(f"Combined statistics CSV -> {combinedStatisticsCsvPath}")


if __name__ == "__main__":
    main()

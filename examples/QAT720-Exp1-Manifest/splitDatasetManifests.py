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

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from manifestStatistics import createManifestStatistics, writeStatisticsCsv


DEFAULT_DATASET_PATH = Path(".")
DEFAULT_RATIOS = (30.0, 20.0, 50.0)
SPLIT_NAMES = ("train", "val", "test")
MANIFEST_FILENAMES = {
    "train": "trainManifest.json",
    "val": "valManifest.json",
    "test": "testManifest.json",
}


def validateRatios(ratios: Sequence[float]) -> Tuple[float, float, float]:
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
) -> Tuple[int, int, int]:
    """
    Convert ratio weights into exact integer counts using largest remainders.

    The returned counts always sum to numberOfRecords. Ties are resolved in
    train, validation, then test order.
    """
    trainRatio, valRatio, testRatio = validateRatios(ratios)
    ratioValues = (trainRatio, valRatio, testRatio)
    ratioTotal = sum(ratioValues)
    exactCounts = [
        numberOfRecords * ratioValue / ratioTotal for ratioValue in ratioValues
    ]
    counts = [int(math.floor(value)) for value in exactCounts]
    remaining = numberOfRecords - sum(counts)

    remainderOrder = sorted(
        range(len(SPLIT_NAMES)),
        key=lambda index: (-(exactCounts[index] - counts[index]), index),
    )
    for index in remainderOrder[:remaining]:
        counts[index] += 1

    return counts[0], counts[1], counts[2]


def loadMetadata(metadataPath: Path) -> List[Dict[str, object]]:
    try:
        payload = json.loads(metadataPath.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            "Input manifest is not valid JSON: {} ({})".format(metadataPath, error)
        )

    if not isinstance(payload, list):
        raise ValueError("Input manifest must contain a JSON list of image records")

    records = []
    seenFilePaths = set()
    for index, record in enumerate(payload):
        if not isinstance(record, dict):
            raise ValueError("Manifest record {} is not a JSON object".format(index))

        filePath = record.get("filePath")
        if not isinstance(filePath, str) or not filePath:
            raise ValueError(
                "Manifest record {} has no valid filePath".format(index)
            )
        if filePath in seenFilePaths:
            raise ValueError("Duplicate filePath in input manifest: {}".format(filePath))

        seenFilePaths.add(filePath)
        records.append(record)

    return records


def splitRecords(
    records: Sequence[Dict[str, object]],
    seed: int,
    ratios: Sequence[float],
) -> Dict[str, List[Dict[str, object]]]:
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
    inputRecords: Sequence[Dict[str, object]],
    splitRecordsByName: Dict[str, List[Dict[str, object]]],
) -> None:
    inputPaths = {str(record["filePath"]) for record in inputRecords}
    outputPaths = set()

    for splitName in SPLIT_NAMES:
        splitPaths = {
            str(record["filePath"])
            for record in splitRecordsByName[splitName]
        }
        if outputPaths.intersection(splitPaths):
            raise RuntimeError("The generated splits are not disjoint")
        outputPaths.update(splitPaths)

    if outputPaths != inputPaths:
        missingCount = len(inputPaths - outputPaths)
        unexpectedCount = len(outputPaths - inputPaths)
        raise RuntimeError(
            "The generated splits do not match the input "
            "(missing={}, unexpected={})".format(missingCount, unexpectedCount)
        )


def writeManifestsAtomically(
    outputDir: Path,
    splitRecordsByName: Dict[str, List[Dict[str, object]]],
) -> Dict[str, Path]:
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


def countObjects(records: Sequence[Dict[str, object]]) -> int:
    objectCount = 0
    for record in records:
        objects = record.get("objects", [])
        if not isinstance(objects, list):
            raise ValueError(
                "Record {} has an invalid objects value".format(record["filePath"])
            )
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
        help="Dataset root. Default: {}".format(DEFAULT_DATASET_PATH),
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
        else datasetPath / "manifests" / "seed_{}".format(args.seed)
    )
    ratios = validateRatios(args.ratios)

    if not datasetPath.is_dir():
        raise NotADirectoryError(
            "Dataset directory does not exist: {}".format(datasetPath)
        )
    if not metadataPath.is_file():
        raise FileNotFoundError(
            "Input metadata manifest does not exist: {}".format(metadataPath)
        )

    records = loadMetadata(metadataPath)
    splits = splitRecords(records, args.seed, ratios)
    validateSplits(records, splits)
    writtenPaths = writeManifestsAtomically(outputDir, splits)
    statisticsPaths = {
        splitName: createManifestStatistics(writtenPaths[splitName])
        for splitName in SPLIT_NAMES
    }
    statisticsPayloads = {
        splitName: json.loads(
            statisticsPaths[splitName].read_text(encoding="utf-8")
        )
        for splitName in SPLIT_NAMES
    }
    combinedStatisticsCsvPath = writeStatisticsCsv(
        outputDir / "manifestStatistics.csv",
        statisticsPayloads,
        SPLIT_NAMES,
    )

    print("Input manifest: {}".format(metadataPath))
    print("Split seed: {}".format(args.seed))
    print(
        "Requested train:val:test ratio: {:g}:{:g}:{:g}".format(
            ratios[0],
            ratios[1],
            ratios[2],
        )
    )
    print("Output directory: {}".format(outputDir))

    totalImages = len(records)
    totalObjects = countObjects(records)
    for splitName in SPLIT_NAMES:
        splitRecordList = splits[splitName]
        splitImages = len(splitRecordList)
        splitObjects = countObjects(splitRecordList)
        imagePercent = 100.0 * splitImages / totalImages if totalImages else 0.0
        objectPercent = 100.0 * splitObjects / totalObjects if totalObjects else 0.0
        print(
            "{}: {} images ({:.2f}%), {} objects ({:.2f}%) -> {}".format(
                splitName,
                splitImages,
                imagePercent,
                splitObjects,
                objectPercent,
                writtenPaths[splitName],
            )
        )
        print(
            "{} statistics -> {}".format(
                splitName,
                statisticsPaths[splitName],
            )
        )
    print("Combined statistics CSV -> {}".format(combinedStatisticsCsvPath))


if __name__ == "__main__":
    main()

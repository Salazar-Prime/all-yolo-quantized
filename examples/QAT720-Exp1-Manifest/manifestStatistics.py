# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Calculate descriptive statistics for one dataset manifest.

The manifest must be a JSON list of image records in the format produced by
exp1/generateMetadata.py. By default, JSON and CSV reports are written next to
the input manifest as <manifestStem>Statistics.json and
<manifestStem>Statistics.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Sequence

COCO_SMALL_AREA_MAX_PX2 = 32 * 32
COCO_MEDIUM_AREA_MAX_PX2 = 96 * 96


def loadManifest(manifestPath: Path) -> list[dict[str, object]]:
    try:
        payload = json.loads(manifestPath.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Manifest is not valid JSON: {manifestPath} ({error})")

    if not isinstance(payload, list):
        raise TypeError("Manifest must contain a JSON list of image records")

    records = []
    seenFilePaths = set()
    for index, record in enumerate(payload):
        if not isinstance(record, dict):
            raise TypeError(f"Manifest record {index} is not a JSON object")

        filePath = record.get("filePath")
        if not isinstance(filePath, str) or not filePath:
            raise ValueError(f"Manifest record {index} has no valid filePath")
        if filePath in seenFilePaths:
            raise ValueError(f"Duplicate filePath in manifest: {filePath}")

        objects = record.get("objects")
        if not isinstance(objects, list):
            raise TypeError(f"Manifest record {index} has no valid objects list")

        seenFilePaths.add(filePath)
        records.append(record)

    return records


def requireFiniteNumber(value: object, fieldName: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{fieldName} must be numeric, found {value!r}")
    if not math.isfinite(number):
        raise ValueError(f"{fieldName} must be finite, found {value!r}")
    return number


def percentile(sortedValues: Sequence[float], fraction: float) -> float | None:
    if not sortedValues:
        return None
    if len(sortedValues) == 1:
        return float(sortedValues[0])

    position = (len(sortedValues) - 1) * fraction
    lowerIndex = math.floor(position)
    upperIndex = math.ceil(position)
    if lowerIndex == upperIndex:
        return float(sortedValues[lowerIndex])

    lowerValue = sortedValues[lowerIndex]
    upperValue = sortedValues[upperIndex]
    return float(lowerValue + (upperValue - lowerValue) * (position - lowerIndex))


def summarizeValues(values: Sequence[float]) -> dict[str, object]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "populationStdDev": None,
            "p10": None,
            "p25": None,
            "p75": None,
            "p90": None,
            "p95": None,
        }

    sortedValues = sorted(float(value) for value in values)
    return {
        "count": len(sortedValues),
        "min": min(sortedValues),
        "max": max(sortedValues),
        "mean": statistics.mean(sortedValues),
        "median": statistics.median(sortedValues),
        "populationStdDev": statistics.pstdev(sortedValues),
        "p10": percentile(sortedValues, 0.10),
        "p25": percentile(sortedValues, 0.25),
        "p75": percentile(sortedValues, 0.75),
        "p90": percentile(sortedValues, 0.90),
        "p95": percentile(sortedValues, 0.95),
    }


def percentage(count: int, total: int) -> float:
    return 100.0 * count / total if total else 0.0


def cocoSizeCategory(areaPx2: float) -> str:
    if areaPx2 < COCO_SMALL_AREA_MAX_PX2:
        return "small"
    if areaPx2 < COCO_MEDIUM_AREA_MAX_PX2:
        return "medium"
    return "large"


def distributionPayload(
    counts: Counter,
    total: int,
    orderedKeys: Sequence[object] | None = None,
) -> dict[str, dict[str, object]]:
    keys = orderedKeys if orderedKeys is not None else sorted(counts, key=str)
    return {
        str(key): {
            "count": int(counts.get(key, 0)),
            "percent": percentage(int(counts.get(key, 0)), total),
        }
        for key in keys
    }


def buildManifestStatistics(
    records: Sequence[dict[str, object]],
    manifestPath: Path | None = None,
) -> dict[str, object]:
    imageWidths = []
    imageHeights = []
    imageAspectRatios = []
    objectsPerImage = []

    boxWidthsPx = []
    boxHeightsPx = []
    boxAreasPx2 = []
    boxWidthsNormalized = []
    boxHeightsNormalized = []
    boxAreasNormalized = []

    classCounts = Counter()
    cocoSizeCounts = Counter()
    objectsPerImageFrequency = Counter()
    plotImageCounts = Counter()
    plotObjectCounts = Counter()

    storedObjectCountMismatches = 0
    storedCocoCategoryMismatches = 0
    totalObjects = 0

    for recordIndex, record in enumerate(records):
        imageWidth = requireFiniteNumber(
            record.get("imageWidth"),
            f"record[{recordIndex}].imageWidth",
        )
        imageHeight = requireFiniteNumber(
            record.get("imageHeight"),
            f"record[{recordIndex}].imageHeight",
        )
        if imageWidth <= 0 or imageHeight <= 0:
            raise ValueError("Image dimensions must be positive for {}".format(record["filePath"]))

        objects = record["objects"]
        objectCount = len(objects)
        storedObjectCount = record.get("numberOfObjects")
        if storedObjectCount is None or int(storedObjectCount) != objectCount:
            storedObjectCountMismatches += 1

        imageWidths.append(imageWidth)
        imageHeights.append(imageHeight)
        imageAspectRatios.append(imageWidth / imageHeight)
        objectsPerImage.append(float(objectCount))
        objectsPerImageFrequency[objectCount] += 1
        totalObjects += objectCount

        plotNumber = record.get("plotNumber")
        plotKey = str(plotNumber) if plotNumber is not None else "unknown"
        plotImageCounts[plotKey] += 1
        plotObjectCounts[plotKey] += objectCount

        for objectIndex, objectRecord in enumerate(objects):
            if not isinstance(objectRecord, dict):
                raise TypeError(
                    "Object {} in {} is not a JSON object".format(
                        objectIndex,
                        record["filePath"],
                    )
                )

            fieldPrefix = f"record[{recordIndex}].objects[{objectIndex}]"
            widthPx = requireFiniteNumber(
                objectRecord.get("widthPx"),
                fieldPrefix + ".widthPx",
            )
            heightPx = requireFiniteNumber(
                objectRecord.get("heightPx"),
                fieldPrefix + ".heightPx",
            )
            areaPx2 = requireFiniteNumber(
                objectRecord.get("areaPx2"),
                fieldPrefix + ".areaPx2",
            )
            widthNormalized = requireFiniteNumber(
                objectRecord.get("boxWidth"),
                fieldPrefix + ".boxWidth",
            )
            heightNormalized = requireFiniteNumber(
                objectRecord.get("boxHeight"),
                fieldPrefix + ".boxHeight",
            )
            if widthPx <= 0 or heightPx <= 0 or areaPx2 <= 0:
                raise ValueError(f"Pixel box dimensions and area must be positive for {fieldPrefix}")
            if widthNormalized <= 0 or heightNormalized <= 0:
                raise ValueError(f"Normalized box dimensions must be positive for {fieldPrefix}")

            classId = objectRecord.get("classId")
            if classId is None:
                raise ValueError(f"{fieldPrefix} has no classId")

            derivedCategory = cocoSizeCategory(areaPx2)
            storedCategory = objectRecord.get("cocoSizeCategory")
            if storedCategory != derivedCategory:
                storedCocoCategoryMismatches += 1

            classCounts[str(classId)] += 1
            cocoSizeCounts[derivedCategory] += 1
            boxWidthsPx.append(widthPx)
            boxHeightsPx.append(heightPx)
            boxAreasPx2.append(areaPx2)
            boxWidthsNormalized.append(widthNormalized)
            boxHeightsNormalized.append(heightNormalized)
            boxAreasNormalized.append(widthNormalized * heightNormalized)

    numberOfImages = len(records)
    imagesWithObjects = sum(count > 0 for count in objectsPerImage)
    imagesWithoutObjects = numberOfImages - imagesWithObjects

    plotKeys = sorted(
        set(plotImageCounts) | set(plotObjectCounts),
        key=lambda key: (key == "unknown", key),
    )
    plotDistribution = {
        plotKey: {
            "images": int(plotImageCounts.get(plotKey, 0)),
            "objects": int(plotObjectCounts.get(plotKey, 0)),
        }
        for plotKey in plotKeys
    }

    payload = {
        "sourceManifest": (str(manifestPath.resolve()) if manifestPath is not None else None),
        "images": {
            "total": numberOfImages,
            "withObjects": imagesWithObjects,
            "withObjectsPercent": percentage(imagesWithObjects, numberOfImages),
            "withoutObjects": imagesWithoutObjects,
            "withoutObjectsPercent": percentage(
                imagesWithoutObjects,
                numberOfImages,
            ),
            "storedObjectCountMismatches": storedObjectCountMismatches,
            "widthPx": summarizeValues(imageWidths),
            "heightPx": summarizeValues(imageHeights),
            "aspectRatio": summarizeValues(imageAspectRatios),
            "objectsPerImage": summarizeValues(objectsPerImage),
            "objectCountFrequency": {
                str(key): int(objectsPerImageFrequency[key]) for key in sorted(objectsPerImageFrequency)
            },
        },
        "objects": {
            "total": totalObjects,
            "classDistribution": distributionPayload(classCounts, totalObjects),
            "cocoSizeDistribution": distributionPayload(
                cocoSizeCounts,
                totalObjects,
                orderedKeys=("small", "medium", "large"),
            ),
            "storedCocoCategoryMismatches": storedCocoCategoryMismatches,
            "pixelBoundingBoxes": {
                "widthPx": summarizeValues(boxWidthsPx),
                "heightPx": summarizeValues(boxHeightsPx),
                "areaPx2": summarizeValues(boxAreasPx2),
            },
            "normalizedBoundingBoxes": {
                "width": summarizeValues(boxWidthsNormalized),
                "height": summarizeValues(boxHeightsNormalized),
                "area": summarizeValues(boxAreasNormalized),
            },
        },
        "plotDistribution": plotDistribution,
    }
    return payload


def defaultStatisticsPath(manifestPath: Path) -> Path:
    return manifestPath.with_name(manifestPath.stem + "Statistics.json")


def writeJsonAtomically(outputPath: Path, payload: object) -> None:
    outputPath.parent.mkdir(parents=True, exist_ok=True)
    temporaryPath = outputPath.with_name(outputPath.name + ".tmp")
    try:
        temporaryPath.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporaryPath.replace(outputPath)
    finally:
        if temporaryPath.exists():
            temporaryPath.unlink()


def flattenStatistics(
    payload: dict[str, object],
    prefix: str = "",
) -> dict[str, object]:
    """
    Flatten nested statistics into dot-delimited metric names for tabular output.
    """
    flattened = {}
    for key in sorted(payload):
        value = payload[key]
        metricName = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(flattenStatistics(value, metricName))
        elif isinstance(value, list):
            flattened[metricName] = json.dumps(value, separators=(",", ":"))
        else:
            flattened[metricName] = value
    return flattened


def writeStatisticsCsv(
    outputPath: Path,
    statisticsByColumn: dict[str, dict[str, object]],
    columnOrder: Sequence[str],
) -> Path:
    """
    Write multiple statistics payloads into one CSV.

    Metrics are rows and columnOrder values (for example train/val/test) are
    columns. The union of all metric names is written so split-specific plot or
    class keys are not lost.
    """
    missingColumns = [columnName for columnName in columnOrder if columnName not in statisticsByColumn]
    if missingColumns:
        raise ValueError("Missing statistics columns: {}".format(", ".join(missingColumns)))

    flattenedByColumn = {columnName: flattenStatistics(statisticsByColumn[columnName]) for columnName in columnOrder}
    metricNames = sorted({metricName for flattened in flattenedByColumn.values() for metricName in flattened})

    outputPath = outputPath.expanduser().resolve()
    outputPath.parent.mkdir(parents=True, exist_ok=True)
    temporaryPath = outputPath.with_name(outputPath.name + ".tmp")
    try:
        with temporaryPath.open("w", newline="", encoding="utf-8") as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(["metric"] + list(columnOrder))
            for metricName in metricNames:
                writer.writerow(
                    [metricName]
                    + [
                        flattenedByColumn[columnName].get(metricName, "")
                        if flattenedByColumn[columnName].get(metricName) is not None
                        else ""
                        for columnName in columnOrder
                    ]
                )
        temporaryPath.replace(outputPath)
    finally:
        if temporaryPath.exists():
            temporaryPath.unlink()

    return outputPath


def createManifestStatistics(
    manifestPath: Path,
    outputPath: Path | None = None,
) -> Path:
    manifestPath = manifestPath.expanduser().resolve()
    if not manifestPath.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifestPath}")

    resolvedOutputPath = (
        outputPath.expanduser().resolve() if outputPath is not None else defaultStatisticsPath(manifestPath)
    )
    records = loadManifest(manifestPath)
    payload = buildManifestStatistics(records, manifestPath)
    writeJsonAtomically(resolvedOutputPath, payload)
    return resolvedOutputPath


def parseArguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifestPath",
        type=Path,
        help="Path to a train, validation, test, or full dataset manifest.",
    )
    parser.add_argument(
        "--outputPath",
        type=Path,
        default=None,
        help="Statistics JSON output. Default: next to the manifest.",
    )
    parser.add_argument(
        "--csvOutputPath",
        type=Path,
        default=None,
        help="Statistics CSV output. Default: same path as JSON with .csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parseArguments()
    writtenPath = createManifestStatistics(args.manifestPath, args.outputPath)
    payload = json.loads(writtenPath.read_text(encoding="utf-8"))
    csvOutputPath = (
        args.csvOutputPath.expanduser().resolve() if args.csvOutputPath is not None else writtenPath.with_suffix(".csv")
    )
    writtenCsvPath = writeStatisticsCsv(
        csvOutputPath,
        {"manifest": payload},
        ("manifest",),
    )

    print(f"Statistics written to: {writtenPath}")
    print(f"CSV statistics written to: {writtenCsvPath}")
    print("Images: {}".format(payload["images"]["total"]))
    print("Objects: {}".format(payload["objects"]["total"]))
    cocoDistribution = payload["objects"]["cocoSizeDistribution"]
    print(
        "COCO sizes: small={} medium={} large={}".format(
            cocoDistribution["small"]["count"],
            cocoDistribution["medium"]["count"],
            cocoDistribution["large"]["count"],
        )
    )


if __name__ == "__main__":
    main()

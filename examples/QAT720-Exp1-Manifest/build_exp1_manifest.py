# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Build an Exp1-compatible datasetMetadata.json from JPEG images and YOLO detection labels."""

import argparse
import base64
import json
import math
import re
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

FILE_NAME_PATTERN = re.compile(r"^plot(?P<plotNumber>\d+)Tile(?P<tileNumber>\d+)\.jpe?g$", re.IGNORECASE)
COCO_SMALL_AREA_MAX = 32 * 32
COCO_MEDIUM_AREA_MAX = 96 * 96


def parsePlotAndTile(fileName: str) -> Tuple[Optional[int], Optional[int]]:
    """Extract plot and tile numbers from a QAT720 image filename."""
    match = FILE_NAME_PATTERN.match(fileName)
    if match is None:
        return None, None
    return int(match.group("plotNumber")), int(match.group("tileNumber"))


def getJpegSize(filePath: Path) -> Optional[Tuple[int, int]]:
    """Read JPEG dimensions without decoding the complete image."""
    try:
        with filePath.open("rb") as imageFile:
            if imageFile.read(2) != b"\xff\xd8":
                return None
            while True:
                markerPrefix = imageFile.read(1)
                if not markerPrefix:
                    return None
                if markerPrefix != b"\xff":
                    continue
                markerType = imageFile.read(1)
                while markerType == b"\xff":
                    markerType = imageFile.read(1)
                if not markerType:
                    return None
                markerValue = markerType[0]
                if markerValue in (0xD8, 0xD9):
                    continue
                lengthBytes = imageFile.read(2)
                if len(lengthBytes) != 2:
                    return None
                segmentLength = int.from_bytes(lengthBytes, "big")
                if segmentLength < 2:
                    return None
                if markerValue in {
                    0xC0,
                    0xC1,
                    0xC2,
                    0xC3,
                    0xC5,
                    0xC6,
                    0xC7,
                    0xC9,
                    0xCA,
                    0xCB,
                    0xCD,
                    0xCE,
                    0xCF,
                }:
                    payload = imageFile.read(segmentLength - 2)
                    if len(payload) < 5:
                        return None
                    imageHeight = int.from_bytes(payload[1:3], "big")
                    imageWidth = int.from_bytes(payload[3:5], "big")
                    return imageWidth, imageHeight
                imageFile.seek(segmentLength - 2, 1)
    except OSError:
        return None


def parseLabel(line: str, labelPath: Path, lineNumber: int) -> Dict[str, float]:
    """Parse and validate one normalized YOLO detection label."""
    parts = line.split()
    if len(parts) != 5:
        raise ValueError("{}:{} must contain class_id x_center y_center width height".format(labelPath, lineNumber))
    try:
        classId = int(parts[0])
        xCenter, yCenter, boxWidth, boxHeight = (float(value) for value in parts[1:])
    except ValueError as error:
        raise ValueError("{}:{} contains a non-numeric label value".format(labelPath, lineNumber)) from error
    if classId < 0:
        raise ValueError("{}:{} class ID must be nonnegative".format(labelPath, lineNumber))
    values = (xCenter, yCenter, boxWidth, boxHeight)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("{}:{} coordinates must be finite".format(labelPath, lineNumber))
    if not (0 <= xCenter <= 1 and 0 <= yCenter <= 1 and 0 < boxWidth <= 1 and 0 < boxHeight <= 1):
        raise ValueError("{}:{} coordinates must be normalized YOLO values".format(labelPath, lineNumber))
    return {
        "classId": classId,
        "xCenter": xCenter,
        "yCenter": yCenter,
        "boxWidth": boxWidth,
        "boxHeight": boxHeight,
    }


def pixelBox(obj: Dict[str, float], imageWidth: int, imageHeight: int) -> Dict[str, int]:
    """Convert one normalized YOLO box to clipped integer pixel coordinates."""
    left = max(0, min(round((obj["xCenter"] - obj["boxWidth"] / 2) * imageWidth), imageWidth - 1))
    top = max(0, min(round((obj["yCenter"] - obj["boxHeight"] / 2) * imageHeight), imageHeight - 1))
    right = max(left + 1, min(round((obj["xCenter"] + obj["boxWidth"] / 2) * imageWidth), imageWidth))
    bottom = max(top + 1, min(round((obj["yCenter"] + obj["boxHeight"] / 2) * imageHeight), imageHeight))
    return {"left": left, "top": top, "right": right, "bottom": bottom}


def cocoSizeCategory(area: int) -> str:
    """Return the standard COCO size category for a pixel area."""
    if area < COCO_SMALL_AREA_MAX:
        return "small"
    if area < COCO_MEDIUM_AREA_MAX:
        return "medium"
    return "large"


def findLabel(imagePath: Path, imagesDir: Path, labelsDir: Path) -> Optional[Path]:
    """Resolve a mirrored or flat YOLO label path for an image."""
    relative = imagePath.relative_to(imagesDir).with_suffix(".txt")
    candidates = (labelsDir / relative, labelsDir / (imagePath.stem + ".txt"))
    return next((path for path in candidates if path.is_file()), None)


def buildObjects(imagePath: Path, labelPath: Optional[Path], imageWidth: int, imageHeight: int) -> List[dict]:
    """Build Exp1 object records for one image and its optional label file."""
    if labelPath is None:
        return []
    objects = []
    lines = labelPath.read_text(encoding="utf-8").splitlines()
    for objectIndex, line in enumerate(lines):
        if not line.strip():
            continue
        obj = parseLabel(line, labelPath, objectIndex + 1)
        box = pixelBox(obj, imageWidth, imageHeight)
        widthPx = box["right"] - box["left"]
        heightPx = box["bottom"] - box["top"]
        areaPx2 = widthPx * heightPx
        identity = "{}:{}:{}".format(imagePath.absolute(), objectIndex, line.strip())
        objectUuid = uuid.uuid5(uuid.NAMESPACE_URL, identity).hex
        value = "{},{},{},{},{}".format(
            obj["xCenter"], obj["yCenter"], obj["boxWidth"], obj["boxHeight"], objectUuid
        )
        objects.append(
            {
                "objectIndex": objectIndex,
                "objectUuid": objectUuid,
                **obj,
                "objectValueBase64": base64.b64encode(value.encode("utf-8")).decode("ascii"),
                "widthPx": widthPx,
                "heightPx": heightPx,
                "areaPx2": areaPx2,
                "cocoSizeCategory": cocoSizeCategory(areaPx2),
                "bboxPixel": box,
            }
        )
    return objects


def buildManifest(imagesDir: Path, labelsDir: Path) -> List[dict]:
    """Return Exp1 image records for every JPEG below an image directory."""
    imagePaths = sorted(
        (path for path in imagesDir.rglob("*") if path.is_file() and path.suffix.casefold() in (".jpg", ".jpeg")),
        key=lambda path: str(path).casefold(),
    )
    records = []
    for imagePath in imagePaths:
        imageSize = getJpegSize(imagePath)
        if imageSize is None:
            raise ValueError("Image is not a readable JPEG: {}".format(imagePath))
        imageWidth, imageHeight = imageSize
        plotNumber, tileNumber = parsePlotAndTile(imagePath.name)
        objects = buildObjects(imagePath, findLabel(imagePath, imagesDir, labelsDir), imageWidth, imageHeight)
        records.append(
            {
                "filePath": str(imagePath.absolute()),
                "fileName": imagePath.name,
                "plotNumber": plotNumber,
                "tileNumber": tileNumber,
                "imageWidth": imageWidth,
                "imageHeight": imageHeight,
                "aspectRatio": imageWidth / imageHeight,
                "numberOfObjects": len(objects),
                "objects": objects,
            }
        )
    if not records:
        raise ValueError("No JPEG images found under {}".format(imagesDir))
    return records


def parseArguments() -> argparse.Namespace:
    """Parse metadata generation arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, default=None, help="Default: DATASET_ROOT.")
    parser.add_argument("--labels-dir", type=Path, default=None, help="Default: DATASET_ROOT/labels.")
    parser.add_argument("--output", type=Path, default=None, help="Default: DATASET_ROOT/datasetMetadata.json.")
    return parser.parse_args()


def main() -> None:
    """Generate and atomically write an Exp1-compatible combined manifest."""
    args = parseArguments()
    datasetRoot = args.dataset_root.expanduser().resolve()
    imagesDir = (args.images_dir or datasetRoot).expanduser().resolve()
    labelsDir = (args.labels_dir or datasetRoot / "labels").expanduser().resolve()
    output = (args.output or datasetRoot / "datasetMetadata.json").expanduser().resolve()
    if not imagesDir.is_dir():
        raise NotADirectoryError("Image directory does not exist: {}".format(imagesDir))
    if not labelsDir.is_dir():
        raise NotADirectoryError("Label directory does not exist: {}".format(labelsDir))

    records = buildManifest(imagesDir, labelsDir)
    temporary = output.with_name(output.name + ".tmp")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    objectCount = sum(len(record["objects"]) for record in records)
    emptyCount = sum(not record["objects"] for record in records)
    print("Images: {} ({} without objects)".format(len(records), emptyCount))
    print("Objects: {}".format(objectCount))
    print("Manifest: {}".format(output))


if __name__ == "__main__":
    main()

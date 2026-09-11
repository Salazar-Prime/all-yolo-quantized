# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convert an Exp1 annotation manifest into a native Ultralytics detection dataset."""

import argparse

from train import addManifestArguments, prepareDataset, printStatistics


def parseArguments() -> argparse.Namespace:
    """Parse Exp1 manifest conversion arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    addManifestArguments(parser)
    return parser.parse_args()


def main() -> None:
    """Generate and summarize a symlink-backed Ultralytics dataset."""
    args = parseArguments()
    dataYaml, statistics = prepareDataset(args)
    print(f"Ultralytics data config: {dataYaml}")
    printStatistics(statistics, args.objectSize, args.skipEmptyImages)


if __name__ == "__main__":
    main()

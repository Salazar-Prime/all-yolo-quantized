"""Capture privileged Jetson sensors with monotonic timestamps until an explicit stop marker appears."""

import argparse
import fcntl
import json
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    stop = args.output.with_suffix(".stop")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", buffering=1) as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        process = subprocess.Popen(["tegrastats", "--interval", "100"], stdout=subprocess.PIPE, universal_newlines=True)
        try:
            for line in process.stdout:
                if stop.exists():
                    break
                handle.write(
                    json.dumps({"monotonic": time.monotonic(), "unix": time.time(), "raw": line.strip()}) + "\n"
                )
        finally:
            process.terminate()
            process.wait(timeout=10)


if __name__ == "__main__":
    main()

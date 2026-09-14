#!/usr/bin/env python3
"""Print current Exp5 preparation, Xavier benchmark, and archival progress."""

import csv
import datetime
import inspect
import json
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def snapshot(directory):
    import subprocess
    from pathlib import Path

    root = Path(directory)
    result = {"files": {}, "active": {}}
    for pattern in (
        "*/prepare-*.exit",
        "*/*/*.exit",
        "*/*/pass[123].json",
        "*/model.exit",
        "*/archived.json",
        "*/preparation-failed.json",
    ):
        for path in root.glob(pattern):
            try:
                result["files"][str(path.relative_to(root))] = (
                    path.read_text().strip() if path.suffix == ".exit" else "present"
                )
            except FileNotFoundError:
                continue  # The controller can archive and remove a model during this snapshot.
    for line in subprocess.check_output(["ps", "-eo", "args="], universal_newlines=True).splitlines():
        args = line.split()
        if len(args) < 4:
            continue
        script, model, variant, phase = args[-4:]
        if script.endswith("/prepare.py") and variant == root.name:
            result["active"][model + "/prepare-" + phase] = "Running"
        elif script.endswith("/benchmark.py") and Path(model).parent == root:
            result["active"][Path(model).name + "/" + variant] = phase
    return result


def remote_snapshot(host, directory):
    source = inspect.getsource(snapshot) + "\nimport json, sys\nprint(json.dumps(snapshot(sys.argv[1])))\n"
    command = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "-o", "ConnectionAttempts=1", host]
    try:
        result = subprocess.run(
            command + ["python3 - " + shlex.quote(directory)],
            input=source,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=20,
            check=True,
        )
        return json.loads(result.stdout)
    except (subprocess.SubprocessError, OSError, ValueError):
        return None


def progress(model, variant, data, available, preparation=False):
    files, active = data["files"], data["active"]
    key = model + "/" + ("prepare-" if preparation else "") + variant
    if preparation:
        code = files.get(key + ".exit")
        return ("Done" if code == "0" else "FAIL") if code else active.get(key, "Wait" if available else "Unknown")
    if model + "/preparation-failed.json" in files:
        return "Prep FAIL"
    for phase in ("build", "evaluate", "profile"):
        code = files.get(key + "/" + phase + ".exit")
        if code and code != "0":
            return "FAIL " + phase[:4]
    passes = sum(key + "/pass{}.json".format(n) in files for n in (1, 2, 3))
    if key in active:
        return {"build": "Building", "idle": "Idle", "evaluate": "Test {}/3".format(passes), "profile": "Profiling"}[
            active[key]
        ]
    if passes == 3 and files.get(key + "/evaluate.exit") == "0":
        if variant != "onnx" or files.get(key + "/profile.exit") == "0":
            return "Done"
    if passes:
        return "Partial {}/3".format(passes)
    if files.get(model + "/model.exit", "0") != "0":
        return "Run FAIL"
    return "Wait" if available else "Unknown"


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "exp5"))
    from run import HOST, RAINBOW, ROOT, XAVIER

    run = json.loads((ROOT / "exp5/protocol.json").read_text())["production_run"]
    models = [row["model"] for row in csv.DictReader((ROOT / "exp5/models.csv").open())]
    print(
        "Exp5 progress: {} | {}".format(run, datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")),
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            label: pool.submit(remote_snapshot, host, root + "/exp5/runs/" + run)
            for label, host, root in (("Rainbow", "rainbow", RAINBOW), ("Xavier", HOST, XAVIER))
        }
        sources = {label: future.result() for label, future in futures.items()}
    available = {label: data is not None for label, data in sources.items()}
    local = snapshot(ROOT / "exp5/runs" / run)
    paused = (ROOT / "exp5/runs" / run / "xavier-paused.json").exists()
    if paused:
        print("Xavier benchmarks are paused for field testing; Rainbow preparation continues.")
    for label, data in sources.items():
        if data is None:
            print("{} unavailable: showing Anvil copies where present; other states are Unknown.".format(label))
            sources[label] = {"files": dict(local["files"]), "active": {}}
        else:
            sources[label] = {"files": dict(local["files"], **data["files"]), "active": data["active"]}
    rows = [["Model", "Export", "Calib", "QAT tune", "ONNX", "TRT32", "TRT16", "INT8 PTQ", "INT8 QAT", "Archive"]]
    for model in models:
        prep = [
            progress(model, phase, sources["Rainbow"], available["Rainbow"], True) for phase in ("fp32", "ptq", "qat")
        ]
        archived = model + "/archived.json" in local["files"]
        benchmarks = [
            progress(model, variant, sources["Xavier"], archived or available["Xavier"])
            for variant in ("onnx", "fp32", "fp16", "ptq", "qat")
        ]
        for index, phase in enumerate((0, 0, 0, 1, 2)):
            if prep[phase] == "FAIL":
                benchmarks[index] = "Prep FAIL"
            elif paused and benchmarks[index] != "Done" and "FAIL" not in benchmarks[index]:
                benchmarks[index] = "Paused"
        rows.append([model] + prep + benchmarks + ["Done" if archived else "Wait"])
    widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]))]
    print()
    for index, row in enumerate(rows):
        print(" | ".join(value.ljust(width) for value, width in zip(row, widths)))
        if index == 0:
            print("-+-".join("-" * width for width in widths))
    print(
        "\nPreparation: {}/{} models | Xavier: {}/{} combinations | Archived: {}/{} models".format(
            sum(all(value == "Done" for value in row[1:4]) for row in rows[1:]),
            len(models),
            sum(value == "Done" for row in rows[1:] for value in row[4:9]),
            len(models) * 5,
            sum(row[-1] == "Done" for row in rows[1:]),
            len(models),
        )
    )
    print("Wait = queued; Test N/3 = completed passes during active evaluation; FAIL = inspect that stage's log.")


if __name__ == "__main__":
    main()

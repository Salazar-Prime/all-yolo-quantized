#!/usr/bin/env python3
"""Print current Xavier benchmark and archival progress for an experiment protocol."""

import argparse
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
    result = {"files": {}, "active": {}, "controllers": []}
    for pattern in (
        "*/prepare-*.exit",
        "*/*/*.exit",
        "*/*/pass[123].json",
        "*/model.exit",
        "*/archived.json",
        "*/*/archived.json",
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
        if "--device" in args and root.name in args and any(a.endswith("/run.py") for a in args):
            result["controllers"].append(args[args.index("--device") + 1])
        for index, arg in enumerate(args[:-3]):
            if arg.endswith("/benchmark.py"):
                model, variant, phase = args[index + 1 : index + 4]
                if Path(model).parent == root:
                    if phase == "evaluate" and "--images" in args:
                        phase = "smoke"
                    result["active"][Path(model).name + "/" + variant] = phase
                break
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


def progress(model, variant, data, available):
    files, active = data["files"], data["active"]
    key = model + "/" + variant
    source_phase = variant.split("_")[0] if variant.startswith(("ptq", "qat")) else "fp32"
    code = files.get(model + "/prepare-" + source_phase + ".exit", "0")
    if model + "/preparation-failed.json" in files or code != "0":
        return "Prep FAIL"
    for phase in ("build", "smoke", "evaluate", "profile"):
        code = files.get(key + "/" + phase + ".exit")
        if code and code != "0":
            return "FAIL " + phase[:4]
    passes = sum(key + "/pass{}.json".format(n) in files for n in (1, 2, 3))
    if key in active:
        return {
            "build": "Building",
            "smoke": "Smoke check",
            "idle": "Idle",
            "evaluate": "Test {}/3".format(passes),
            "profile": "Profiling",
        }[active[key]]
    if passes == 3 and files.get(key + "/evaluate.exit") == "0":
        if variant not in {"onnx", "ptq_fp16", "qat_fp16"} or files.get(key + "/profile.exit") == "0":
            return "Done"
    if passes:
        return "Partial {}/3".format(passes)
    if files.get(model + "/model.exit", "0") != "0":
        return "Run FAIL"
    return "Wait" if available else "Unknown"


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "exp5"))
    from run import ROOT, deployment_for

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("exp5/protocol.json"))
    args = parser.parse_args()
    protocol = json.loads((ROOT / args.protocol).read_text())
    run, deployments = protocol["production_run"], protocol["deployments"]
    labels = {
        "onnx": "ONNX",
        "fp32": "TRT32",
        "fp16": "TRT16",
        "ptq": "INT8 PTQ",
        "qat": "INT8 QAT",
        "ptq_fp16": "PTQ + FP16",
        "qat_fp16": "QAT + FP16",
    }
    variants = [v for v in labels if any(v in group for group in protocol["variant_priority"])]
    models = [row["model"] for row in csv.DictReader((ROOT / protocol["model_manifest"]).open())]
    directory = ROOT / protocol["experiment"] / "runs" / run
    print(
        "Xavier model progress: {} | {}".format(
            run, datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        ),
        flush=True,
    )
    data = snapshot(directory)
    paused = (directory / "xavier-paused.json").exists()
    if paused:
        print(
            "Xavier benchmarks are paused: "
            + json.loads((directory / "xavier-paused.json").read_text()).get("reason", "see pause record")
        )
    with ThreadPoolExecutor(max_workers=len(deployments)) as pool:
        futures = {
            name: pool.submit(
                remote_snapshot, device["ssh_alias"], device["root"] + "/" + protocol["experiment"] + "/runs/" + run
            )
            for name, device in deployments.items()
        }
        remotes = {name: future.result() for name, future in futures.items()}
    for name, remote in remotes.items():
        if remote is None:
            print(name + " unavailable: showing Anvil copies; other states are Unknown.")
        else:
            data["files"].update(remote["files"])
            data["active"].update(remote["active"])
        launch = directory / "setup" / (name + "-launch.json")
        if (
            launch.exists()
            and name not in data["controllers"]
            and not (directory / ("complete-" + name + ".json")).exists()
        ):
            state = json.loads(launch.read_text())
            print(
                name
                + ": "
                + {
                    "waiting_for_runtime_and_data_transfer": "Staging files",
                    "validating_runtime_and_dataset": "Checking runtime/dataset",
                    "controller_started": "Controller stopped; inspect controller log",
                    "failed": "Setup FAIL: " + state.get("error", "see launch record"),
                }.get(state["status"], state["status"])
            )
    rows = [["Model", "Device"] + [labels[v] for v in variants] + ["Archive"]]
    for model in models:
        archived = model + "/archived.json" in data["files"]
        device = deployment_for(model, deployments)
        benchmarks = [progress(model, variant, data, archived or remotes[device] is not None) for variant in variants]
        if paused:
            benchmarks = [value if value == "Done" or "FAIL" in value else "Paused" for value in benchmarks]
        archived_variants = sum(model + "/" + v + "/archived.json" in data["files"] for v in variants)
        rows.append(
            [model, device] + benchmarks + ["Done" if archived else "{}/{}".format(archived_variants, len(variants))]
        )
    widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]))]
    print()
    for index, row in enumerate(rows):
        print(" | ".join(value.ljust(width) for value, width in zip(row, widths)))
        if index == 0:
            print("-+-".join("-" * width for width in widths))
    print(
        "\nXavier: {}/{} combinations | Archived: {}/{} models".format(
            sum(value == "Done" for row in rows[1:] for value in row[2:-1]),
            len(models) * len(variants),
            sum(row[-1] == "Done" for row in rows[1:]),
            len(models),
        )
    )
    print("Wait = queued; Test N/3 = completed passes during active evaluation; FAIL = inspect that stage's log.")


if __name__ == "__main__":
    main()

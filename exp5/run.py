"""Control one Xavier's assigned models and verify their archive on Anvil."""

import argparse
import csv
import fcntl
import hashlib
import json
import shlex
import socket
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAINBOW = "/home/varun/work/all-yolo-quantized"
PROTOCOL = json.loads((ROOT / "exp5/protocol.json").read_text())
DEPLOYMENTS = PROTOCOL["deployments"]


def deployment_for(model):
    return next(name for name, device in DEPLOYMENTS.items() if model.startswith(tuple(device["model_prefixes"])))


def remote(host, command):
    while True:
        try:
            return subprocess.check_output(
                ["ssh", "-o", "BatchMode=yes", host, "bash -c " + shlex.quote(command)], universal_newlines=True
            ).strip()
        except subprocess.CalledProcessError as error:
            if error.returncode != 255:
                raise
            print("SSH unavailable for {}; retrying in 30 seconds".format(host), flush=True)
            time.sleep(30)


def transfer(source, destination):
    while True:
        result = subprocess.run(["rsync", "-a", source, destination])
        if result.returncode == 0:
            return
        if result.returncode not in (12, 255):
            result.check_returncode()
        print("Transfer connection interrupted; retrying in 30 seconds", flush=True)
        time.sleep(30)


def stage(host, root, directory, name, command):
    status = directory + "/" + name + ".exit"
    present = remote(host, "if [ -f {0} ]; then cat {0}; else echo pending; fi".format(shlex.quote(status)))
    if present == "pending":
        script = "exec 9>{4}\nflock -n 9 || exit 0\ntest -f {3} && exit 0\ncd {0}\n{1} >{2} 2>&1\nrc=$?\nprintf '%s\\n' \"$rc\" >{3}\n".format(
            shlex.quote(root),
            command,
            shlex.quote(directory + "/" + name + ".log"),
            shlex.quote(status),
            shlex.quote(directory + "/" + name + ".lock"),
        )
        remote(
            host,
            "mkdir -p {0}; nohup bash -c {1} </dev/null >{2} 2>&1 &".format(
                shlex.quote(directory), shlex.quote(script), shlex.quote(directory + "/" + name + ".launcher.log")
            ),
        )
    while present == "pending":
        time.sleep(30)
        present = remote(
            host,
            "if [ -f {0} ]; then cat {0}; elif flock -n {1} true; then echo interrupted; else echo pending; fi".format(
                shlex.quote(status), shlex.quote(directory + "/" + name + ".lock")
            ),
        )
    if present == "interrupted":
        raise RuntimeError("Remote worker stopped without an exit record; preserve its artifacts before recovery")
    return int(present)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    parser.add_argument("--device", choices=DEPLOYMENTS, default="soysan")
    args = parser.parse_args()
    if not socket.gethostname().startswith("login"):
        raise RuntimeError("The controller must run on Anvil; compute runs remotely")
    if not args.run.replace("-", "").replace("_", "").isalnum():
        raise ValueError("Run ID may contain only letters, digits, hyphens, and underscores")
    output = ROOT / "exp5/runs" / args.run
    output.mkdir(parents=True, exist_ok=True)
    controller_lock = (output / (args.device + ".lock")).open("w")
    fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (output / "xavier-paused.json").exists():
        raise SystemExit("Xavier is paused; review xavier-paused.json before resuming")
    cases = list(csv.DictReader((ROOT / "exp5/models.csv").open()))
    cases = sorted(
        (case for case in cases if deployment_for(case["model"]) == args.device),
        key=lambda case: int(case["checkpoint_bytes"]),
    )
    device = DEPLOYMENTS[args.device]
    HOST, XAVIER = device["ssh_alias"], device["root"]
    telemetry = XAVIER + "/exp5/runs/" + args.run + "/telemetry.jsonl"
    remote(HOST, "test -r " + shlex.quote(telemetry))
    remote(
        HOST,
        "flock -n -E 10 " + shlex.quote(telemetry) + " true; test $? -eq 10",
    )
    queue = ((case, group) for group in PROTOCOL["variant_priority"] for case in cases)
    for case, group in queue:
        model = case["model"]
        local = output / model
        if (local / "archived.json").exists():
            continue
        variants = [variant for variant in group if not (local / variant / "archived.json").exists()]
        if not variants:
            continue
        rainbow_dir = RAINBOW + "/exp5/runs/" + args.run + "/" + model
        xavier_dir = XAVIER + "/exp5/runs/" + args.run + "/" + model
        print("Waiting for prepared " + model, flush=True)
        statuses = [rainbow_dir + "/prepare-" + phase + ".exit" for phase in ("fp32", "ptq", "qat")]
        ready = " && ".join("test -f " + shlex.quote(path) for path in statuses)
        while remote("rainbow", ready + " && echo ready || echo pending") != "ready":
            time.sleep(30)
        local.mkdir(parents=True, exist_ok=True)
        transfer("rainbow:" + rainbow_dir + "/", str(local) + "/")
        if not (local / "reference.json").exists():
            (local / "preparation-failed.json").write_text(json.dumps(case, indent=2) + "\n")
            continue
        remote(HOST, "mkdir -p " + shlex.quote(xavier_dir))
        sources = {variant if variant in {"ptq", "qat"} else "fp32" for variant in variants}
        for name in [source + ".onnx" for source in sorted(sources)] + ["reference.json"]:
            if (local / name).exists():
                transfer(str(local / name), HOST + ":" + xavier_dir + "/")
        remote(
            HOST,
            "cp {0} {1}".format(
                shlex.quote(str(Path(telemetry).parent / "device.json")), shlex.quote(xavier_dir + "/device.json")
            ),
        )
        print("Running {} ({}) on {}".format(model, ", ".join(variants), args.device), flush=True)
        rc = stage(
            HOST,
            XAVIER,
            xavier_dir,
            "model",
            "bash exp5/run_xavier.sh "
            + " ".join(shlex.quote(arg) for arg in [xavier_dir, device["service"]] + variants),
        )
        if rc:
            raise RuntimeError("Xavier model wrapper failed; retain its artifacts for recovery")
        checksums = remote(
            HOST, "cd {0} && find . -type f -print0 | sort -z | xargs -0 sha256sum".format(shlex.quote(xavier_dir))
        )
        archive_lock = (output / "archive.lock").open("w")
        fcntl.flock(archive_lock, fcntl.LOCK_EX)
        transfer(HOST + ":" + xavier_dir + "/", str(local) + "/")
        for line in checksums.splitlines():
            expected, filename = line.split("  ", 1)
            path = local / filename
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != expected:
                raise RuntimeError("Archive checksum mismatch: " + str(path))
        (local / ("xavier-" + "-".join(group) + ".sha256")).write_text(checksums + "\n")
        remote(HOST, "if test -d {0}; then rm -r -- {0}; fi".format(shlex.quote(xavier_dir)))
        receipt = (
            json.dumps(
                {"device": args.device, "verified_files": len(checksums.splitlines()), "time": time.time()}, indent=2
            )
            + "\n"
        )
        for variant in variants:
            (local / variant / "archived.json").write_text(receipt)
        if all(
            (local / variant / "archived.json").exists() for group in PROTOCOL["variant_priority"] for variant in group
        ):
            (local / "archived.json").write_text(receipt)
        subprocess.run(["python3", str(ROOT / "exp5/summarize.py"), str(output)], check=True)
        archive_lock.close()
        print("Archived and removed Xavier copy of " + model, flush=True)
    remote(HOST, "touch " + shlex.quote(XAVIER + "/exp5/runs/" + args.run + "/telemetry.stop"))
    with (output / "archive.lock").open("w") as archive_lock:
        fcntl.flock(archive_lock, fcntl.LOCK_EX)
        subprocess.run(["python3", str(ROOT / "exp5/summarize.py"), str(output)], check=True)
    (output / ("complete-" + args.device + ".json")).write_text(
        json.dumps({"models": len(cases), "time": time.time()}, indent=2) + "\n"
    )
    if all((output / ("complete-" + name + ".json")).exists() for name in DEPLOYMENTS):
        (output / "complete.json").write_text(
            json.dumps({"devices": list(DEPLOYMENTS), "time": time.time()}, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()

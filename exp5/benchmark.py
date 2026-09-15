"""Build or evaluate one model variant on Xavier without replacing the native validation loop."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import resource
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("variant", choices=("onnx", "fp32", "fp16", "ptq", "qat"))
    parser.add_argument("stage", choices=("build", "evaluate", "profile", "idle"))
    parser.add_argument("--images", type=int, default=4579)
    parser.add_argument("--passes", type=int, default=3)
    args = parser.parse_args()
    if "Xavier" not in Path("/proc/device-tree/model").read_text():
        raise RuntimeError("Deployment measurements must run on Xavier")
    directory = args.directory.resolve()
    artifact = directory / ("fp32.onnx" if args.variant == "onnx" else args.variant + ".engine")
    output = directory / args.variant
    output.mkdir(exist_ok=True)
    if args.stage == "idle":
        started = time.monotonic()
        time.sleep(30)
        (output / "idle.json").write_text(
            json.dumps({"start_monotonic": started, "end_monotonic": time.monotonic()}, indent=2) + "\n"
        )
        return
    if args.stage == "profile":
        import cv2
        import numpy as np
        import onnxruntime as ort

        from ultralytics.data.augment import LetterBox

        options = ort.SessionOptions()
        options.enable_profiling = True
        options.profile_file_prefix = str(output / "ort-placement")
        session = ort.InferenceSession(
            str(artifact), options, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        first_image = next(
            p
            for p in sorted((ROOT / "exp5/inputs/test/images").iterdir())
            if p.suffix.lower() in {".jpg", ".png", ".jpeg"}
        )
        sample = LetterBox((640, 640), auto=False)(image=cv2.imread(str(first_image)))
        sample = np.ascontiguousarray(sample[:, :, ::-1].transpose(2, 0, 1))[None].astype(np.float32) / 255
        session.run(None, {"images": sample})
        profile = json.loads(Path(session.end_profiling()).read_text())
        placement = [
            {"node": event["name"], "provider": event["args"]["provider"]}
            for event in profile
            if event.get("args", {}).get("provider")
        ]
        if not any(node["provider"] == "CUDAExecutionProvider" for node in placement):
            raise RuntimeError("The ONNX graph did not execute any CUDA kernels")
        (output / "placement.json").write_text(json.dumps(placement, indent=2) + "\n")
        return
    if args.stage == "build":
        source = directory / ((args.variant if args.variant in {"ptq", "qat"} else "fp32") + ".onnx")
        command = [
            "/usr/src/tensorrt/bin/trtexec",
            "--onnx=" + str(source),
            "--saveEngine=" + str(artifact),
            "--memPoolSize=workspace:1024",
            "--profilingVerbosity=detailed",
            "--buildOnly",
            "--noTF32",
            "--exportLayerInfo=" + str(output / "layers.json"),
            "--timingCacheFile=" + str(directory.parent / "timing.cache"),
        ]
        if args.variant == "fp16":
            command.append("--fp16")
        elif args.variant in {"ptq", "qat"}:
            command.append("--int8")
        started = time.monotonic()
        result = subprocess.run(command)
        formats = Counter()
        if result.returncode == 0:
            shutil.copyfile(directory.parent / "timing.cache", output / "timing.cache")
            for layer in json.loads((output / "layers.json").read_text())["Layers"]:
                formats.update(tensor["Format/Datatype"] for tensor in layer.get("Outputs", []))
        (output / "build.json").write_text(
            json.dumps(
                {
                    "command": command,
                    "start_monotonic": started,
                    "end_monotonic": time.monotonic(),
                    "returncode": result.returncode,
                    "layer_output_tensor_formats": dict(formats),
                },
                indent=2,
            )
            + "\n"
        )
        raise SystemExit(result.returncode)

    import torch

    from ultralytics.data import build_dataloader
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.utils import YAML

    class XavierValidator(DetectionValidator):
        def get_dataloader(self, dataset_path, batch_size):
            dataset = self.build_dataset(dataset_path)
            if args.images == 4579 and len(dataset) != 4579:
                raise ValueError("The complete test dataset must contain 4,579 images")
            if args.images != 4579:
                subset = torch.utils.data.Subset(dataset, range(args.images))
                subset.collate_fn = dataset.collate_fn
                dataset = subset
            return build_dataloader(dataset, batch=1, workers=2, shuffle=False, rank=-1)

        def init_metrics(self, model):
            super().init_metrics(model)
            if args.variant == "onnx":
                providers = model.backend.session.get_providers()
                if "CUDAExecutionProvider" not in providers:
                    raise RuntimeError("The ONNX GPU benchmark requires CUDAExecutionProvider")
                (output / "providers.json").write_text(json.dumps(providers, indent=2) + "\n")
            batches = itertools.chain.from_iterable(itertools.repeat(self.dataloader))
            for batch in itertools.islice(batches, 100):
                batch = self.preprocess(batch)
                self.postprocess(model(batch["img"]))
            torch.cuda.synchronize()
            self.started = time.monotonic()

    test = ROOT / "exp5/inputs/test"
    data_file = ROOT / "exp5/inputs/test/data.yaml"
    YAML.save(
        data_file, {"path": str(test), "train": "images", "val": "images", "test": "images", "names": {0: "weed"}}
    )
    reference = json.loads((directory / "reference.json").read_text())
    for trial in range(args.passes):
        validator = XavierValidator(
            args={
                "model": str(artifact),
                "data": str(data_file),
                "split": "test",
                "imgsz": 640,
                "batch": 1,
                "device": 0,
                "workers": 2,
                "rect": False,
                "nms": None,
                "iou": 0.5,
                "conf": 0.001,
                "max_det": 300,
                "plots": False,
                "save_json": False,
                "project": str(output),
                "name": "pass{}".format(trial + 1),
                "exist_ok": True,
            }
        )

        def batch_done(v):
            v.finished = time.monotonic()

        validator.add_callback("on_val_batch_end", batch_done)
        metrics = validator()
        speed = validator.speed
        fps = 1000 / speed["inference"]
        result = {
            "model": reference["model"],
            "device": json.loads((directory / "device.json").read_text()),
            "variant": args.variant,
            "pass": trial + 1,
            "test_images": args.images,
            "metrics": metrics,
            "speed_ms_per_image": speed,
            "inference_fps": fps,
            "pipeline_fps": 1000 / sum(speed[k] for k in ("preprocess", "inference", "postprocess")),
            "start_monotonic": validator.started,
            "end_monotonic": validator.finished,
            "pass_wall_seconds": validator.finished - validator.started,
            "reference_gflops_per_image": reference["reference_gflops_per_image"],
            "effective_gops_per_second": (reference["reference_gflops_per_image"] * fps)
            if reference["reference_gflops_per_image"]
            else None,
            "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            "nms_iou": 0.5,
            "throughput_basis": "reference model operations times synchronized inference FPS; not hardware counters",
        }
        (output / "pass{}.json".format(trial + 1)).write_text(json.dumps(result, indent=2) + "\n")
        del validator
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

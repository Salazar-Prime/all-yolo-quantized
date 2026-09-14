"""Prepare one Exp2 model on Rainbow using the repository's quantization and training owners."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
import os
import shutil
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("run")
    parser.add_argument("stage", choices=("fp32", "ptq", "qat"))
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    if socket.gethostname() != "digital-ag":
        raise RuntimeError("Model preparation must run on Rainbow")

    os.environ["CUDA_HOME"] = "/usr/local/cuda-11.8"
    os.environ["PATH"] = str(Path(sys.executable).parent) + ":/usr/local/cuda-11.8/bin:" + os.environ["PATH"]
    os.environ["TORCH_EXTENSIONS_DIR"] = str(ROOT / "exp5/.cache/torch_extensions")
    os.environ["MAX_JOBS"] = "4"

    import onnx
    import torch

    torch.cuda.init()
    gpu = torch.cuda.get_device_properties(0)
    if "GPU-" + str(gpu.uuid) != os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("Select one physical Rainbow GPU by UUID before starting preparation")
    device = torch.device("cuda:0")
    print("Preparing on {} (GPU-{})".format(gpu.name, gpu.uuid), flush=True)

    from ultralytics import YOLO, settings
    from ultralytics.cfg import get_cfg
    from ultralytics.data import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils import YAML
    from ultralytics.utils.torch_utils import get_flops, prepare_qat, qat_state, strip_qat

    settings.update({"wandb": False, "tensorboard": False, "sync": False})
    case = next(r for r in csv.DictReader((ROOT / "exp5/models.csv").open()) if r["model"] == args.model)
    source = ROOT / case["checkpoint"]
    if hashlib.sha256(source.read_bytes()).hexdigest() != case["sha256"]:
        raise ValueError("Source checkpoint hash differs from exp3")
    output = ROOT / "exp5/runs" / args.run / args.model
    output.mkdir(parents=True, exist_ok=True)
    (output / (args.stage + "-environment.json")).write_text(
        json.dumps(
            {
                "gpu_name": gpu.name,
                "gpu_uuid": str(gpu.uuid),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "packages": {
                    name: importlib.metadata.version(name) for name in ("nvidia-modelopt", "onnx", "ultralytics-thop")
                },
                "argv": sys.argv,
            },
            indent=2,
        )
        + "\n"
    )
    data = YAML.load(ROOT / "exp3/prepared/full/data.yaml")
    data["train"] = data["val"]
    data_file = output / "data.yaml"
    YAML.save(data_file, data)
    data = check_det_dataset(data_file, autodownload=False)
    batch = 2 if int(case["checkpoint_bytes"]) > 100_000_000 else 4 if int(case["checkpoint_bytes"]) > 50_000_000 else 8
    config = get_cfg(overrides={"imgsz": 640, "batch": batch, "workers": 4, "task": "detect", "rect": False})

    if args.stage == "fp32":
        checkpoint = output / "fp32.pt"
        shutil.copyfile(source, checkpoint)
        model = YOLO(checkpoint)
        model.model.set_head_attr(end2end=False)
        model.model.to(device).float().eval()
        flops = get_flops(model.model, imgsz=640)
        if flops <= 0:
            raise RuntimeError("The reference FLOPs profiler failed")
        from thop.profile import register_hooks
        from ultralytics.nn.modules.block import AAttn, Attention

        uninstrumented = sorted(
            {
                type(module).__name__
                for module in model.model.modules()
                if type(module) not in register_hooks and not isinstance(module, (AAttn, Attention))
            }
        )
        (output / "reference.json").write_text(
            json.dumps(
                {
                    **case,
                    "reference_gflops_per_image": flops,
                    "profiler": "ultralytics-thop; MAC=2 ops",
                    "modules_without_direct_counter": uninstrumented,
                    "coverage": "Child modules are counted; uncovered functional arithmetic, NMS, and runtime fusion are not hardware instruction counts",
                },
                indent=2,
            )
            + "\n"
        )
    elif args.stage == "ptq":
        model = YOLO(source)
        model.model.to(device).float().eval()
        dataset = build_yolo_dataset(config, data["val"], batch, data, mode="val", rect=False, stride=32)
        if len(dataset) != 1832:
            raise ValueError("Calibration must contain all 1,832 validation images")
        loader = build_dataloader(dataset, batch=batch, workers=4, shuffle=False, rank=-1)
        prepare_qat(model.model, loader, lambda b: b["img"].to(device).float() / 255, batches=len(loader))
        import modelopt.torch.quantization as mtq

        # Exp4 excludes this attention convolution, which TensorRT 8.5 cannot build with INT8 Q/DQ on Xavier.
        mtq.disable_quantizer(model.model, "*attn.pe.conv*")
        state = qat_state(model.model)
        saved = copy.deepcopy(model.model).cpu().float()
        strip_qat(saved)
        checkpoint = output / "ptq.pt"
        torch.save(
            {"model": saved, "modelopt": state, "train_args": {"task": "detect", "imgsz": 640, "nms": None}}, checkpoint
        )
        del model, saved, loader, dataset
        torch.cuda.empty_cache()
        model = YOLO(checkpoint)
    else:

        class CalibratedTrainer(DetectionTrainer):
            def get_model(self, cfg=None, weights=None, verbose=True):
                return YOLO(output / "ptq.pt").model

        model = YOLO(source)
        model.train(
            trainer=CalibratedTrainer,
            data=str(data_file),
            epochs=args.epochs,
            batch=batch,
            imgsz=640,
            device=0,
            workers=4,
            project=str(output),
            name="qat",
            exist_ok=False,
            quantize=None,
            optimizer="AdamW",
            lr0=0.0001,
            lrf=0.1,
            cos_lr=True,
            warmup_epochs=1,
            amp=False,
            seed=42,
            deterministic=True,
            mosaic=0.0,
            mixup=0.0,
            copy_paste=0.0,
            plots=False,
            cache=False,
            patience=0,
            nms=None,
            iou=0.5,
            max_det=300,
        )
        model = YOLO(model.trainer.last)

    exported = Path(model.export(format="onnx", imgsz=640, batch=1, device=0, nms=None, opset=13, simplify=False))
    target = output / (args.stage + ".onnx")
    if exported != target:
        shutil.move(str(exported), target)
    graph = onnx.load(target)
    onnx.checker.check_model(graph)
    quantizers = sum(n.op_type == "QuantizeLinear" for n in graph.graph.node)
    if bool(quantizers) != (args.stage != "fp32"):
        raise ValueError("ONNX quantization nodes do not match the requested variant")
    (output / (args.stage + ".json")).write_text(
        json.dumps(
            {
                "quantize_linear_nodes": quantizers,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "opset": 13,
                "calibration_images": 1832 if quantizers else 0,
                "preparation_gpu": {"name": gpu.name, "uuid": str(gpu.uuid)},
                "qat_epochs": args.epochs if args.stage == "qat" else 0,
                "float_exclusions": ["detection head", "*attn.pe.conv*"] if quantizers else [],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

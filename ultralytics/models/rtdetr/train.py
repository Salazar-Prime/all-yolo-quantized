# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import copy

from ultralytics.cfg import DEFAULT_CFG
from ultralytics.data.utils import get_split_fraction
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import RTDETRDetectionModel, YOLODETRDetectionModel
from ultralytics.utils import LOGGER, RANK, colorstr
from ultralytics.utils.torch_utils import unwrap_model

from .val import RTDETRDataset, RTDETRValidator


class RTDETRTrainer(DetectionTrainer):
    """Trainer class for the RT-DETR model developed by Baidu for real-time object detection.

    This class extends the DetectionTrainer class for YOLO to adapt to the specific features and architecture of
    RT-DETR. The model leverages Vision Transformers and has capabilities like IoU-aware query selection and adaptable
    inference speed.

    Attributes:
        loss_names (tuple): Names of the loss components, derived from the loss dict returned by the criterion.
        data (dict): Dataset configuration containing class count and other parameters.
        args (dict): Training arguments and hyperparameters.
        save_dir (Path): Directory to save training results.
        test_loader (DataLoader): DataLoader for validation/testing data.

    Methods:
        get_model: Initialize and return an RT-DETR model for object detection tasks.
        build_dataset: Build and return an RT-DETR dataset for training or validation.
        get_validator: Return a DetectionValidator suitable for RT-DETR model validation.

    Examples:
        >>> from ultralytics.models.rtdetr.train import RTDETRTrainer
        >>> args = dict(model="rtdetr-l.yaml", data="coco8.yaml", imgsz=640, epochs=3)
        >>> trainer = RTDETRTrainer(overrides=args)
        >>> trainer.train()

    Notes:
        - F.grid_sample used in RT-DETR does not support the `deterministic=True` argument.
        - AMP training can lead to NaN outputs and may produce errors during bipartite graph matching.
    """

    def get_model(self, cfg: dict | None = None, weights: str | None = None, verbose: bool = True):
        """Initialize and return an RT-DETR model for object detection tasks.

        Args:
            cfg (dict, optional): Model configuration.
            weights (str, optional): Path to pre-trained model weights.
            verbose (bool): Verbose logging if True.

        Returns:
            (RTDETRDetectionModel): Initialized model.
        """
        model = self.set_model_names_for_load(
            RTDETRDetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        )
        if weights:
            model.load(weights)
        return model

    def build_dataset(self, img_path: str, mode: str = "val", batch: int | None = None):
        """Build and return an RT-DETR dataset for training or validation.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): Dataset mode, either 'train' or 'val'.
            batch (int, optional): Batch size for rectangle training.

        Returns:
            (RTDETRDataset): Dataset object for the specific mode.
        """
        return RTDETRDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=False,
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            prefix=colorstr(f"{mode}: "),
            classes=self.args.classes,
            data=self.data,
            fraction=1.0 if self.data.get("complete") else get_split_fraction(self.args.fraction, mode),
        )

    def get_validator(self):
        """Return an RTDETRValidator suitable for RT-DETR model validation."""
        return RTDETRValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))


class DEIMTrainer(RTDETRTrainer):
    """RT-DETR trainer for DeimDecoder models.

    ``backbone_lr_ratio`` defaults to 0.1 and discounts the backbone param groups' LR in ``build_optimizer``.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize the DEIM trainer with a 0.1 backbone LR ratio and no separate bias warmup LR."""
        super().__init__(cfg, {"backbone_lr_ratio": 0.1, **(overrides or {}), "warmup_bias_lr": 0.0}, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Build YOLODETRDetectionModel and load weights; cls-head rows remap by class name inside model.load().

        Args:
            cfg (str | dict, optional): Model configuration.
            weights (str | Path, optional): Pretrained weights to load.
            verbose (bool): Log the model summary.

        Returns:
            (YOLODETRDetectionModel): Model ready for training.
        """
        model = self.set_model_names_for_load(
            YOLODETRDetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        )
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Return an RTDETRValidator with loss_names extended for the DEIM head.

        Returns:
            (RTDETRValidator): Validator whose loss names match the head in use.
        """
        loss_names = ["giou_loss", "cls_loss", "l1_loss"]
        head_name = type(unwrap_model(self.model).model[-1]).__name__
        if head_name == "DeimDecoder":
            loss_names += ["fgl_loss", "ddf_loss"]
        self.loss_names = tuple(loss_names)
        return RTDETRValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """Resolve 'auto' to the DEIM AdamW defaults, then group parameters with the shared BaseTrainer builder.

        Args:
            model (nn.Module): Model whose parameters are grouped.
            name (str): Optimizer name; auto resolves to AdamW with the DEIM learning rate defaults.
            lr (float): Learning rate for the head groups; the backbone groups scale it by backbone_lr_ratio.
            momentum (float): Momentum or beta1, depending on the optimizer.
            decay (float): Weight decay applied to the weight groups only.
            iterations (float): Total training iterations, unused here since auto never consults it.

        Returns:
            (torch.optim.Optimizer): Optimizer whose backbone groups run at lr * backbone_lr_ratio.
        """
        if str(name).lower() == "auto":
            name, lr, momentum = "AdamW", 5e-4, 0.9
            self.args.warmup_momentum = momentum
            LOGGER.info(
                f"{colorstr('optimizer:')} 'optimizer=auto' found, ignoring 'lr0={self.args.lr0}' and "
                f"'momentum={self.args.momentum}' and using DEIM defaults '{name}', 'lr0={lr}', 'momentum={momentum}'..."
            )
        return super().build_optimizer(model, name, lr, momentum, decay, iterations)

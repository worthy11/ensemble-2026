from __future__ import annotations

import os
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ecg_common import list_images


def load_rgb_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    return (mask > 127).astype(np.float32)


class ECGSegmentationDataset(Dataset):
    def __init__(self, image_dir: Path, mask_dir: Path | None, image_size: int, train: bool) -> None:
        self.image_paths = list_images(image_dir)
        self.mask_dir = mask_dir
        self.transform = build_transforms(image_size=image_size, train=train)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        image_path = self.image_paths[index]
        image = load_rgb_image(image_path)

        if self.mask_dir is None:
            transformed = self.transform(image=image)
            return {
                "image": transformed["image"],
                "image_name": image_path.name,
            }

        mask_path = self.mask_dir / image_path.name
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found for {image_path.name}: {mask_path}")

        mask = load_mask(mask_path)
        transformed = self.transform(image=image, mask=mask)
        return {
            "image": transformed["image"],
            "mask": transformed["mask"].unsqueeze(0),
            "image_name": image_path.name,
        }


def build_transforms(image_size: int, train: bool) -> A.Compose:
    resize = [A.Resize(height=image_size, width=image_size)]
    normalize = [A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]

    if not train:
        return A.Compose(resize + normalize)

    augmentations = [
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.03,
            scale_limit=0.08,
            rotate_limit=4,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.7,
        ),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.2),
        A.Perspective(scale=(0.02, 0.05), p=0.2),
    ]
    return A.Compose(resize + augmentations + normalize)


def build_model(encoder_name: str = "resnet50") -> nn.Module:
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    )


def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    probabilities = probabilities.flatten(1)
    targets = targets.flatten(1)

    intersection = (probabilities * targets).sum(dim=1)
    union = probabilities.sum(dim=1) + targets.sum(dim=1)
    dice_score = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice_score.mean()


def segmentation_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = dice_loss_from_logits(logits, targets)
    return dice + 0.5 * bce


@torch.no_grad()
def batch_iou(logits: torch.Tensor, targets: torch.Tensor, threshold: float) -> torch.Tensor:
    predictions = (torch.sigmoid(logits) >= threshold).float()
    predictions = predictions.flatten(1)
    targets = targets.flatten(1)
    intersection = (predictions * targets).sum(dim=1)
    union = predictions.sum(dim=1) + targets.sum(dim=1) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean()


class ECGSegmentationLightningModule(pl.LightningModule):
    def __init__(self, lr: float, threshold: float, image_size: int,
                 encoder_name: str = "resnet50", epochs: int = 25) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = build_model(encoder_name=encoder_name)
        self.lr = lr
        self.threshold = threshold
        self.image_size = image_size
        self.total_epochs = epochs

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)

    def _shared_step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        images = batch["image"]
        masks = batch["mask"]
        logits = self(images)
        loss = segmentation_loss(logits, masks)
        iou = batch_iou(logits, masks, self.threshold)

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=images.size(0))
        self.log(f"{stage}_iou", iou, prog_bar=True, on_step=False, on_epoch=True, batch_size=images.size(0))
        return loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="val")

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.total_epochs, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


def _extract_model_state_dict(checkpoint: dict) -> dict[str, torch.Tensor]:
    if "model_state" in checkpoint:
        return checkpoint["model_state"]

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        model_prefix = "model."
        model_keys = [key for key in state_dict if key.startswith(model_prefix)]
        if model_keys:
            return {key[len(model_prefix):]: value for key, value in state_dict.items() if key.startswith(model_prefix)}
        return state_dict

    raise KeyError("Checkpoint must contain 'model_state' or 'state_dict'.")


def _read_checkpoint_hparam(checkpoint: dict, key: str, default):
    if key in checkpoint:
        return checkpoint[key]
    hparams = checkpoint.get("hyper_parameters", {})
    return hparams.get(key, default)


def train_model(args) -> None:
    if torch.cuda.is_available():
        # Better Tensor Core utilization on modern NVIDIA GPUs.
        torch.set_float32_matmul_precision("high")

    requested_workers = int(args.num_workers)
    if requested_workers > 0:
        num_workers = requested_workers
    else:
        cpu_count = os.cpu_count() or 4
        num_workers = min(16, max(2, cpu_count - 1))

    train_dataset = ECGSegmentationDataset(
        image_dir=args.train_images,
        mask_dir=args.train_masks,
        image_size=args.image_size,
        train=True,
    )
    val_dataset = ECGSegmentationDataset(
        image_dir=args.val_images,
        mask_dir=args.val_masks,
        image_size=args.image_size,
        train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    encoder_name = getattr(args, 'encoder_name', 'resnet50')
    lightning_module = ECGSegmentationLightningModule(
        lr=args.lr,
        threshold=args.threshold,
        image_size=args.image_size,
        encoder_name=encoder_name,
        epochs=args.epochs,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(args.output.parent),
        filename=f"{args.output.stem}-lightning-best",
        monitor="val_iou",
        mode="max",
        save_top_k=1,
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint_callback],
        logger=False,
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        benchmark=torch.cuda.is_available(),
        log_every_n_steps=25,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    trainer.fit(lightning_module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    if checkpoint_callback.best_model_path:
        best_checkpoint = torch.load(checkpoint_callback.best_model_path, map_location="cpu")
        model_state = _extract_model_state_dict(best_checkpoint)
        torch.save(
            {
                "model_state": model_state,
                "image_size": args.image_size,
                "threshold": args.threshold,
                "source_lightning_checkpoint": checkpoint_callback.best_model_path,
            },
            args.output,
        )
        print(f"saved best checkpoint to {args.output}")
    else:
        # Fallback for edge cases where callback path is unavailable.
        torch.save(
            {
                "model_state": lightning_module.model.state_dict(),
                "image_size": args.image_size,
                "threshold": args.threshold,
            },
            args.output,
        )
        print(f"saved final checkpoint to {args.output}")


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, int, float]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state = _extract_model_state_dict(checkpoint)

    # Try resnet50 first, fall back to resnet34 for older checkpoints.
    for enc in ("resnet50", "resnet34"):
        try:
            model = build_model(encoder_name=enc).to(device)
            model.load_state_dict(model_state)
            break
        except RuntimeError:
            continue
    else:
        raise RuntimeError("Could not load checkpoint with resnet50 or resnet34 encoder.")
    model.eval()

    image_size = int(_read_checkpoint_hparam(checkpoint, "image_size", 512))
    threshold = float(_read_checkpoint_hparam(checkpoint, "threshold", 0.5))
    return model, image_size, threshold


@torch.no_grad()
def predict_mask_array(
    model: nn.Module,
    image_rgb: np.ndarray,
    image_size: int,
    threshold: float,
    device: torch.device,
) -> np.ndarray:
    original_height, original_width = image_rgb.shape[:2]
    transform = build_transforms(image_size=image_size, train=False)
    tensor = transform(image=image_rgb)["image"].unsqueeze(0).to(device)
    logits = model(tensor)
    probability = torch.sigmoid(logits)[0, 0].cpu().numpy()
    probability = cv2.resize(probability, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
    return (probability >= threshold).astype(np.uint8) * 255


def predict_masks(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint_image_size, checkpoint_threshold = load_checkpoint(args.checkpoint, device)
    image_size = args.image_size or checkpoint_image_size
    threshold = args.threshold if args.threshold is not None else checkpoint_threshold

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for image_path in tqdm(list_images(args.input_dir), desc="predict-masks"):
        image_rgb = load_rgb_image(image_path)
        mask = predict_mask_array(model, image_rgb, image_size=image_size, threshold=threshold, device=device)
        cv2.imwrite(str(args.output_dir / image_path.name), mask)
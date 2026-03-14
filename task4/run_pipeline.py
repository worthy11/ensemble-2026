from __future__ import annotations

import argparse
import os
from pathlib import Path

from ecg_data import prepare_dataset
from ecg_digitization import run_pipeline
from ecg_segmentation import train_model


def parse_args() -> argparse.Namespace:
    scratch = os.environ.get("SCRATCH")
    if scratch:
        default_data_root = Path(scratch) / "tasks_data" / "task4"
    else:
        default_data_root = Path(".")

    parser = argparse.ArgumentParser(
        description=(
            "Run the full ECG digitization pipeline end-to-end: "
            "prepare data, train segmentation, and generate final 1D NPZ output."
        )
    )
    # Data preparation
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=default_data_root / "train",
        help="Folder with train images and JSON metadata",
    )
    parser.add_argument(
        "--prepared-dir",
        type=Path,
        default=Path("data/segmentation_full"),
        help="Output folder for prepared train/val images+masks",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    parser.add_argument("--mask-thickness", type=int, default=3, help="Trace thickness used for generated GT masks")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for quick experiments")

    # Training
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--image-size", type=int, default=512, help="Input image size for segmentation model")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--threshold", type=float, default=0.5, help="Mask threshold for prediction")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/unet_resnet34_full.pt"),
        help="Path to save trained segmentation checkpoint",
    )

    # Inference + digitization
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=default_data_root / "test",
        help="Input test ECG image folder",
    )
    parser.add_argument(
        "--pred-mask-dir",
        type=Path,
        default=Path("data/test_masks_full"),
        help="Where predicted masks are saved",
    )
    parser.add_argument(
        "--submission-output",
        type=Path,
        default=Path("data/out/ecg_example_submission.npz"),
        help="Output NPZ file with 1D signals",
    )
    parser.add_argument("--num-samples", type=int, default=1250, help="Samples per lead in final output")

    return parser.parse_args()


def run_prepare_stage(args: argparse.Namespace) -> None:
    print("[1/3] Preparing data (images + masks + split)...")
    summary = prepare_dataset(
        source_dir=args.source_dir,
        output_dir=args.prepared_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        thickness=args.mask_thickness,
        limit=args.limit,
    )
    print(
        f"Prepared dataset at {summary.output_dir} "
        f"(train={summary.train_count}, val={summary.val_count})"
    )


def run_train_stage(args: argparse.Namespace) -> None:
    print("[2/3] Training segmentation model...")
    train_args = argparse.Namespace(
        train_images=args.prepared_dir / "train" / "images",
        train_masks=args.prepared_dir / "train" / "masks",
        val_images=args.prepared_dir / "val" / "images",
        val_masks=args.prepared_dir / "val" / "masks",
        output=args.checkpoint,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        num_workers=args.num_workers,
        threshold=args.threshold,
    )
    train_model(train_args)
    print(f"Trained model checkpoint saved to {args.checkpoint}")


def run_inference_stage(args: argparse.Namespace) -> None:
    print("[3/3] Running segmentation + 1D digitization...")

    pipeline_args = argparse.Namespace(
        checkpoint=args.checkpoint,
        input_dir=args.test_dir,
        output=args.submission_output,
        num_samples=args.num_samples,
        image_size=args.image_size,
        threshold=args.threshold,
        mask_output_dir=args.pred_mask_dir,
    )

    run_pipeline(pipeline_args)

    print(f"Predicted masks saved to {args.pred_mask_dir}")
    print(f"Final NPZ submission saved to {args.submission_output}")


def main() -> None:
    args = parse_args()

    args.prepared_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.pred_mask_dir.mkdir(parents=True, exist_ok=True)
    args.submission_output.parent.mkdir(parents=True, exist_ok=True)

    run_prepare_stage(args)
    run_train_stage(args)
    run_inference_stage(args)

    print("Pipeline finished successfully.")


if __name__ == "__main__":
    main()

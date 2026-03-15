import argparse
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

ENDPOINT = "task4"
API_TOKEN = os.getenv("TEAM_TOKEN")
SERVER_URL = os.getenv("SERVER_URL")

# Use SCRATCH/tasks_data/task4 as data root if SCRATCH is set (for cluster)
SCRATCH = os.environ.get("SCRATCH")
if SCRATCH:
    DATA_ROOT = Path(SCRATCH) / "tasks_data" / "task4"
else:
    DATA_ROOT = Path(__file__).resolve().parent

# Ensure task4/ is on sys.path regardless of launch directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

TEST_DIR         = DATA_ROOT / "test"
MASK_OUTPUT_DIR  = DATA_ROOT / "data" / "test_masks"
NPZ_FILE         = DATA_ROOT / "data" / "out" / "submission.npz"
CHECKPOINT       = DATA_ROOT / "artifacts" / "unet_resnet34.pt"
NUM_SAMPLES      = 1250


def run_classical_pipeline_cmd() -> None:
    """Run the classical (non-ML) CV pipeline."""
    from ecg_classical import run_classical_pipeline  # noqa: PLC0415

    pipeline_args = argparse.Namespace(
        input_dir=TEST_DIR,
        output=NPZ_FILE,
        num_samples=NUM_SAMPLES,
        mask_output_dir=MASK_OUTPUT_DIR / "classical",
        test_limit=None,
    )
    (MASK_OUTPUT_DIR / "classical").mkdir(parents=True, exist_ok=True)
    NPZ_FILE.parent.mkdir(parents=True, exist_ok=True)
    run_classical_pipeline(pipeline_args)


def run_unet_pipeline_cmd(checkpoint: Path, image_size: int, threshold: float) -> None:
    """Run inference with a trained UNet checkpoint and save submission.npz."""
    from ecg_digitization import run_pipeline  # noqa: PLC0415

    unet_mask_dir = MASK_OUTPUT_DIR / "unet"
    unet_mask_dir.mkdir(parents=True, exist_ok=True)
    NPZ_FILE.parent.mkdir(parents=True, exist_ok=True)

    pipeline_args = argparse.Namespace(
        checkpoint=checkpoint,
        input_dir=TEST_DIR,
        output=NPZ_FILE,
        num_samples=NUM_SAMPLES,
        image_size=image_size,
        threshold=threshold,
        mask_output_dir=unet_mask_dir,
    )
    run_pipeline(pipeline_args)


def run_adaptive_pipeline_cmd() -> None:
    """Run adaptive Otsu + Viterbi pipeline (no trained model needed)."""
    from ecg_digitization import run_adaptive_pipeline  # noqa: PLC0415

    NPZ_FILE.parent.mkdir(parents=True, exist_ok=True)

    pipeline_args = argparse.Namespace(
        input_dir=TEST_DIR,
        output=NPZ_FILE,
        num_samples=NUM_SAMPLES,
    )
    run_adaptive_pipeline(pipeline_args)


def submit() -> None:
    if not API_TOKEN:
        raise ValueError("TEAM_TOKEN not provided. Define TEAM_TOKEN in .env")
    if not SERVER_URL:
        raise ValueError("SERVER_URL not defined. Define SERVER_URL in .env")

    headers = {"X-API-Token": API_TOKEN}

    with open(NPZ_FILE, "rb") as f:
        response = requests.post(
            f"{SERVER_URL}/{ENDPOINT}",
            files={"npz_file": f},
            headers=headers,
        )

    try:
        data = response.json()
    except Exception:
        data = response.text

    print("response:", response.status_code, data)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ECG digitisation pipeline and submit results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["classical", "unet", "adaptive"],
        default="classical",
        help="Pipeline: 'classical' (HSV), 'unet' (trained model), 'adaptive' (Otsu+Viterbi, no model).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT,
        help="Path to trained UNet checkpoint (.pt file). Used only with --mode unet.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Image size for UNet inference (uses checkpoint default if not set).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Mask binarisation threshold (uses checkpoint default if not set).",
    )
    parser.add_argument(
        "--skip-pipeline",
        action="store_true",
        help="Skip pipeline execution; submit the existing NPZ directly.",
    )
    args = parser.parse_args()

    if not args.skip_pipeline:
        if args.mode == "unet":
            if not args.checkpoint.exists():
                raise FileNotFoundError(
                    f"UNet checkpoint not found: {args.checkpoint}\n"
                    "Train the model first with run_pipeline.py, or pass --checkpoint <path>."
                )
            run_unet_pipeline_cmd(
                checkpoint=args.checkpoint,
                image_size=args.image_size or 512,
                threshold=args.threshold or 0.5,
            )
        elif args.mode == "adaptive":
            run_adaptive_pipeline_cmd()
        else:
            run_classical_pipeline_cmd()

    submit()


if __name__ == "__main__":
    main()

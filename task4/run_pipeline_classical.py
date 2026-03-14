from __future__ import annotations

import argparse
import os
from pathlib import Path

from ecg_classical import run_classical_pipeline


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    scratch = os.environ.get("SCRATCH")
    if scratch:
        default_data_root = Path(scratch) / "tasks_data" / "task4"
    else:
        default_data_root = repo_root

    parser = argparse.ArgumentParser(
        description="Run classical OpenCV ECG digitization pipeline (no deep learning)."
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=default_data_root / "test",
        help="Input test ECG image folder",
    )
    parser.add_argument(
        "--pred-mask-dir",
        type=Path,
        default=default_data_root / "data" / "test_masks_classical",
        help="Where predicted binary masks are saved",
    )
    parser.add_argument(
        "--submission-output",
        type=Path,
        default=default_data_root / "data" / "out" / "ecg_submission_classical.npz",
        help="Output NPZ file with 1D signals",
    )
    parser.add_argument("--num-samples", type=int, default=1250, help="Samples per lead in final output")
    parser.add_argument("--test-limit", type=int, default=20, help="Maximum number of test images to process")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.pred_mask_dir.mkdir(parents=True, exist_ok=True)
    args.submission_output.parent.mkdir(parents=True, exist_ok=True)

    pipeline_args = argparse.Namespace(
        input_dir=args.test_dir,
        output=args.submission_output,
        num_samples=args.num_samples,
        mask_output_dir=args.pred_mask_dir,
        test_limit=args.test_limit,
    )
    run_classical_pipeline(pipeline_args)


if __name__ == "__main__":
    main()

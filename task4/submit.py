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

REPO_ROOT = Path(__file__).resolve().parent
TEST_DIR = REPO_ROOT / "test"
MASK_OUTPUT_DIR = REPO_ROOT / "data" / "test_masks_classical"
NPZ_FILE = REPO_ROOT / "data" / "out" / "submission.npz"
NUM_SAMPLES = 1250 


def run_pipeline() -> None:
    sys.path.insert(0, str(REPO_ROOT / "task4"))
    from ecg_classical import run_classical_pipeline  # noqa: PLC0415

    pipeline_args = argparse.Namespace(
        input_dir=TEST_DIR,
        output=NPZ_FILE,
        num_samples=NUM_SAMPLES,
        mask_output_dir=MASK_OUTPUT_DIR,
        test_limit=None,  
    )
    MASK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    NPZ_FILE.parent.mkdir(parents=True, exist_ok=True)
    run_classical_pipeline(pipeline_args)


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
    parser = argparse.ArgumentParser(description="Run ECG pipeline and submit results.")
    parser.add_argument("--skip-pipeline", action="store_true", help="Skip pipeline; submit existing NPZ directly")
    args = parser.parse_args()

    if not args.skip_pipeline:
        run_pipeline()

    submit()


if __name__ == "__main__":
    main()

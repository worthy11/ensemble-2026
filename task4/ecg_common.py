from __future__ import annotations

from pathlib import Path

import numpy as np


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

STANDARD_LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]

GRID_LEAD_LAYOUT = [
    ["I", "aVR", "V1", "V4"],
    ["II", "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"],
]

LEAD_NAME_MAP = {
    "I": "I",
    "II": "II",
    "III": "III",
    "AVR": "AVR",
    "AVL": "AVL",
    "AVF": "AVF",
    "aVR": "AVR",
    "aVL": "AVL",
    "aVF": "AVF",
}


def canonical_lead_name(name: str) -> str:
    return LEAD_NAME_MAP.get(name, name.upper())


def list_images(directory: Path) -> list[Path]:
    image_paths = [path for path in sorted(directory.iterdir()) if path.suffix.lower() in IMAGE_SUFFIXES]
    if not image_paths:
        raise FileNotFoundError(f"No images found in {directory}")
    return image_paths


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32)
    signal = signal - float(np.median(signal))
    scale = float(np.percentile(np.abs(signal), 95))
    if scale > 1e-6:
        signal = signal / scale
    return signal.astype(np.float32)


def record_name_from_path(path: Path) -> str:
    return path.stem
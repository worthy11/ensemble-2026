from __future__ import annotations

from pathlib import Path

import numpy as np


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

STANDARD_LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# Standard 12-lead ECG grid layout: 3 rows × 4 columns.
# In image coordinates, Y=0 is at the top.
# The layout from top (Y=0) to bottom (Y=H) is:
# Row 0: I, aVR, V1, V4
# Row 1: II, aVL, V2, V5
# Row 2: III, aVF, V3, V6
GRID_LEAD_LAYOUT: list[list[str]] = [
    ["I",   "aVR", "V1", "V4"],
    ["II",  "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"],
]

# Sample offset for each column in the 4-column ECG layout.
COLUMN_SAMPLE_OFFSETS = [0, 1250, 2500, 3750]

# Number of samples shown per lead panel (2.5 s × 500 Hz).
LEAD_PANEL_SAMPLES = 1250

# Physical ECG constants (standard calibration).
ECG_PAPER_SPEED_MM_PER_S = 25.0   # mm per second
ECG_GAIN_MM_PER_MV = 10.0          # mm per millivolt (standard gain)
ECG_SAMPLING_RATE_HZ = 500         # samples per second

LEAD_NAME_MAP = {
    "I":   "I",
    "II":  "II",
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
    """Normalize signal to zero-median, 95th-percentile scale.

    NOTE: This removes physical amplitude information (mV units).
    Prefer the mV-calibrated extraction path; this function is kept
    as an optional utility for shape-only comparison.
    """
    signal = np.asarray(signal, dtype=np.float32)
    signal = signal - float(np.median(signal))
    scale = float(np.percentile(np.abs(signal), 95))
    if scale > 1e-6:
        signal = signal / scale
    return signal.astype(np.float32)


def record_name_from_path(path: Path) -> str:
    return path.stem
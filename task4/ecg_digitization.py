from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from ecg_common import (
    COLUMN_SAMPLE_OFFSETS,
    ECG_GAIN_MM_PER_MV,
    ECG_PAPER_SPEED_MM_PER_S,
    GRID_LEAD_LAYOUT,
    LEAD_PANEL_SAMPLES,
    STANDARD_LEADS,
    canonical_lead_name,
    record_name_from_path,
)
from ecg_segmentation import load_checkpoint, load_rgb_image, predict_mask_array


@dataclass
class Layout:
    column_bounds: list[tuple[int, int]]
    row_bounds: list[tuple[int, int]]


# ---------------------------------------------------------------------------
# Amplitude calibration
# ---------------------------------------------------------------------------

def estimate_pixels_per_mv(mask_width: int) -> float:
    """Estimate the pixel-to-millivolt scale from the image (mask) width.

    Standard ECG layout:
      - Paper speed = 25 mm/s  →  2.5 s × 25 mm/s = 62.5 mm per column
      - 4 columns               →  250 mm total useful width
      - Gain = 10 mm/mV         →  pixels_per_mv = (mask_width / 250) × 10

    This is a good first-order approximation for standard A4 / Letter scans.
    For highly degraded or non-standard images the estimate may be off, but
    it will at least preserve relative amplitude proportions.
    """
    seconds_per_column = LEAD_PANEL_SAMPLES / 500  # 2.5 s
    mm_per_column = ECG_PAPER_SPEED_MM_PER_S * seconds_per_column  # 62.5 mm
    total_mm = mm_per_column * len(GRID_LEAD_LAYOUT[0])            # 250 mm (4 cols)
    pixels_per_mm = mask_width / total_mm
    return pixels_per_mm * ECG_GAIN_MM_PER_MV  # px/mV


# ---------------------------------------------------------------------------
# Mask cleaning & layout detection
# ---------------------------------------------------------------------------

def clean_mask(mask: np.ndarray, min_component_area: int = 64) -> np.ndarray:
    binary = (mask > 127).astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    for component_id in range(1, component_count):
        area = stats[component_id, cv2.CC_STAT_AREA]
        if area >= min_component_area:
            cleaned[labels == component_id] = 255
    kernel = np.ones((3, 3), dtype=np.uint8)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    return cleaned


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(values, kernel, mode="same")


def _projection_centers(projection: np.ndarray, groups: int) -> list[int]:
    projection = projection.astype(np.float32)
    smoothed = moving_average(projection, max(9, len(projection) // 80))
    centers: list[int] = []
    rough_edges = np.linspace(0, len(smoothed), groups + 1, dtype=int)
    for start, end in zip(rough_edges[:-1], rough_edges[1:]):
        segment = smoothed[start:end]
        if segment.size == 0:
            centers.append(start)
            continue
        centers.append(start + int(np.argmax(segment)))
    return centers


def _bounds_from_centers(centers: list[int], limit: int) -> list[tuple[int, int]]:
    boundaries = [0]
    for left, right in zip(centers[:-1], centers[1:]):
        boundaries.append((left + right) // 2)
    boundaries.append(limit)
    return [(int(start), int(end)) for start, end in zip(boundaries[:-1], boundaries[1:])]


def estimate_layout(mask: np.ndarray) -> Layout:
    binary = (mask > 127).astype(np.uint8)
    x_projection = binary.sum(axis=0)
    y_projection = binary.sum(axis=1)
    n_cols = len(GRID_LEAD_LAYOUT[0])   # 4
    n_rows = len(GRID_LEAD_LAYOUT)      # 3
    column_centers = _projection_centers(x_projection, groups=n_cols)
    row_centers = _projection_centers(y_projection, groups=n_rows)
    return Layout(
        column_bounds=_bounds_from_centers(column_centers, mask.shape[1]),
        row_bounds=_bounds_from_centers(row_centers, mask.shape[0]),
    )


# ---------------------------------------------------------------------------
# Signal extraction with mV calibration
# ---------------------------------------------------------------------------

def extract_signal_from_region(
    region_mask: np.ndarray,
    target_length: int,
    pixels_per_mv: float,
) -> np.ndarray:
    """Extract a 1-D ECG signal from a binary mask region.

    Args:
        region_mask:  Mask for a single lead panel (uint8, 0/255).
        target_length: Number of output samples.
        pixels_per_mv: Calibration — how many pixel rows equal 1 mV.

    Returns:
        float32 array of length `target_length`, values in millivolts.
        Zero corresponds to the vertical centre of the region (baseline).
    """
    binary = (region_mask > 127).astype(np.uint8)
    height, width = binary.shape

    if binary.sum() == 0:
        return np.zeros(target_length, dtype=np.float32)

    # For each x column, take the median y of lit pixels.
    trace = np.full(width, np.nan, dtype=np.float32)
    for x in range(width):
        ys = np.flatnonzero(binary[:, x])
        if ys.size:
            trace[x] = float(np.median(ys))

    # Interpolate over gaps.
    valid = np.flatnonzero(np.isfinite(trace))
    if valid.size == 0:
        return np.zeros(target_length, dtype=np.float32)
    if valid.size == 1:
        trace[:] = trace[valid[0]]
    else:
        missing = np.flatnonzero(~np.isfinite(trace))
        trace[missing] = np.interp(missing, valid, trace[valid])

    # Light smoothing.
    trace = moving_average(trace, max(5, trace.size // 100))

    # Convert pixel y → millivolts.
    # The vertical centre of the region is treated as the 0 mV baseline.
    # The y-axis is inverted (increasing y = downward in image = negative amplitude).
    baseline_px = height / 2.0
    signal_mv = -(trace - baseline_px) / pixels_per_mv

    # Resample to target_length via linear interpolation.
    src_x = np.linspace(0.0, 1.0, num=signal_mv.size, dtype=np.float32)
    tgt_x = np.linspace(0.0, 1.0, num=target_length, dtype=np.float32)
    return np.interp(tgt_x, src_x, signal_mv).astype(np.float32)


# ---------------------------------------------------------------------------
# Top-level digitisation functions
# ---------------------------------------------------------------------------

def digitize_mask(mask: np.ndarray, num_samples: int) -> dict[str, np.ndarray]:
    """Convert a predicted segmentation mask to a dict of lead → signal (mV).

    Each lead panel is assumed to show `num_samples` samples (e.g. 1250 at 500 Hz).
    The amplitude is calibrated using the image width and standard ECG constants
    (25 mm/s paper speed, 10 mm/mV gain).
    """
    cleaned = clean_mask(mask)
    layout = estimate_layout(cleaned)
    pixels_per_mv = estimate_pixels_per_mv(mask.shape[1])

    signals: dict[str, np.ndarray] = {}

    for column_index, lead_names in enumerate(GRID_LEAD_LAYOUT):
        x0, x1 = layout.column_bounds[column_index]
        width_margin = max(4, (x1 - x0) // 12)
        x0 = min(x1, x0 + width_margin)
        x1 = max(x0 + 1, x1 - width_margin)

        for row_index, lead_name in enumerate(lead_names):
            y0, y1 = layout.row_bounds[row_index]
            height_margin = max(4, (y1 - y0) // 12)
            y0 = min(y1, y0 + height_margin)
            y1 = max(y0 + 1, y1 - height_margin)

            region = cleaned[y0:y1, x0:x1]
            signals[canonical_lead_name(lead_name)] = extract_signal_from_region(
                region,
                target_length=num_samples,
                pixels_per_mv=pixels_per_mv,
            )

    for lead_name in STANDARD_LEADS:
        signals.setdefault(lead_name, np.zeros(num_samples, dtype=np.float32))

    return {lead_name: signals[lead_name] for lead_name in STANDARD_LEADS}


def digitize_mask_file(mask_path: Path, num_samples: int) -> dict[str, np.ndarray]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")
    return digitize_mask(mask, num_samples=num_samples)


def save_submission(records: dict[str, dict[str, np.ndarray]], output_path: Path) -> None:
    flat: dict[str, np.ndarray] = {}
    for record_name, lead_signals in records.items():
        for lead_name in STANDARD_LEADS:
            flat[f"{record_name}_{lead_name}"] = lead_signals[lead_name].astype(np.float16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **flat)


def digitize_masks(args) -> None:
    records: dict[str, dict[str, np.ndarray]] = {}
    mask_paths = sorted(
        path for path in args.mask_dir.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    )
    if not mask_paths:
        raise FileNotFoundError(f"No mask images found in {args.mask_dir}")

    for mask_path in tqdm(mask_paths, desc="digitize-masks"):
        records[record_name_from_path(mask_path)] = digitize_mask_file(
            mask_path, num_samples=args.num_samples
        )

    save_submission(records, args.output)
    print(f"saved digitized signals to {args.output}")


def run_pipeline(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint_image_size, checkpoint_threshold = load_checkpoint(args.checkpoint, device)
    image_size = args.image_size or checkpoint_image_size
    threshold = args.threshold if args.threshold is not None else checkpoint_threshold

    records: dict[str, dict[str, np.ndarray]] = {}
    if args.mask_output_dir is not None:
        args.mask_output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        path for path in args.input_dir.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.input_dir}")

    for image_path in tqdm(image_paths, desc="pipeline"):
        image_rgb = load_rgb_image(image_path)
        mask = predict_mask_array(model, image_rgb, image_size=image_size, threshold=threshold, device=device)
        if args.mask_output_dir is not None:
            cv2.imwrite(str(args.mask_output_dir / image_path.name), mask)
        records[record_name_from_path(image_path)] = digitize_mask(mask, num_samples=args.num_samples)

    save_submission(records, args.output)
    print(f"saved submission archive to {args.output}")
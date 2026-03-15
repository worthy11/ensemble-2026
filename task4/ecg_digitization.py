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
# Adaptive Otsu thresholding with hedging factor (paper approach)
# ---------------------------------------------------------------------------

def _has_grid_lines(binary: np.ndarray, min_line_length_ratio: float = 0.3) -> bool:
    h, w = binary.shape
    horiz_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(20, int(w * min_line_length_ratio)), 1)
    )
    horiz = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel)
    if horiz.sum() > 0:
        return True
    vert_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(20, int(h * min_line_length_ratio)))
    )
    vert = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vert_kernel)
    return vert.sum() > 0


def adaptive_otsu_threshold(
    image_rgb: np.ndarray,
    hedge_start: float = 1.0,
    hedge_step: float = 0.05,
    hedge_min: float = 0.6,
) -> np.ndarray:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray_inv = 255 - gray
    otsu_thresh, _ = cv2.threshold(gray_inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    hedge = hedge_start
    best_binary = None

    while hedge >= hedge_min:
        threshold = otsu_thresh * hedge
        _, binary = cv2.threshold(gray_inv, threshold, 255, cv2.THRESH_BINARY)
        binary = binary.astype(np.uint8)
        if not _has_grid_lines(binary):
            best_binary = binary
            break
        best_binary = binary
        hedge -= hedge_step

    if best_binary is None:
        _, best_binary = cv2.threshold(gray_inv, otsu_thresh * hedge_min, 255, cv2.THRESH_BINARY)
        best_binary = best_binary.astype(np.uint8)

    kernel = np.ones((3, 3), dtype=np.uint8)
    best_binary = cv2.morphologyEx(best_binary, cv2.MORPH_OPEN, kernel)
    best_binary = cv2.morphologyEx(best_binary, cv2.MORPH_CLOSE, kernel)
    return best_binary


# ---------------------------------------------------------------------------
# Grid-based amplitude calibration via FFT
# ---------------------------------------------------------------------------

def _detect_grid_spacing_fft(projection: np.ndarray) -> float | None:
    """Detect grid spacing from a 1-D projection using FFT peak detection.

    Returns the spacing in pixels, or None if detection fails.
    """
    proj = projection.astype(np.float64)
    proj -= proj.mean()
    if proj.std() < 1e-6:
        return None

    fft = np.abs(np.fft.rfft(proj))
    fft[0] = 0  # remove DC
    # Ignore very low frequencies (spacing > 1/3 of total length).
    min_freq_idx = max(3, len(fft) // (len(proj) // 3 + 1))
    # Ignore very high frequencies (spacing < 5 pixels).
    max_freq_idx = min(len(fft) - 1, len(proj) // 5)

    if min_freq_idx >= max_freq_idx:
        return None

    search_region = fft[min_freq_idx:max_freq_idx]
    peak_idx = int(np.argmax(search_region)) + min_freq_idx

    if peak_idx == 0:
        return None

    spacing = len(proj) / peak_idx
    return spacing


def estimate_pixels_per_mv(image_width: int, image_height: int,
                           image_gray: np.ndarray | None = None) -> float:
    """Estimate pixels/mV using FFT grid detection, with width-based fallback.

    If a grayscale image is provided, tries to detect the actual grid spacing
    using FFT on the vertical projection. Falls back to the geometric estimate.
    """
    if image_gray is not None:
        # Vertical projection → detect horizontal grid spacing (amplitude axis).
        v_proj = image_gray.mean(axis=1).astype(np.float64)
        spacing = _detect_grid_spacing_fft(v_proj)
        # Typical ECG image is ~1500-2000px high. 1mm grid is ~8-15px.
        if spacing is not None and 5 < spacing < 30:
            pixels_per_mm = spacing  # spacing IS one grid square (1mm)
            pixels_per_mv = pixels_per_mm * ECG_GAIN_MM_PER_MV  # × 10
            return pixels_per_mv

    # Fallback: geometric estimate from image width.
    seconds_per_column = LEAD_PANEL_SAMPLES / 500.0
    mm_per_column = ECG_PAPER_SPEED_MM_PER_S * seconds_per_column
    total_mm_width = mm_per_column * len(GRID_LEAD_LAYOUT[0])
    pixels_per_mm = image_width / total_mm_width
    return pixels_per_mm * ECG_GAIN_MM_PER_MV


# ---------------------------------------------------------------------------
# Viterbi-based signal path extraction
# ---------------------------------------------------------------------------

def _find_segment_centers(column: np.ndarray) -> list[float]:
    centers: list[float] = []
    in_segment = False
    start = 0
    for i, val in enumerate(column):
        if val > 0 and not in_segment:
            in_segment = True
            start = i
        elif val == 0 and in_segment:
            in_segment = False
            centers.append((start + i - 1) / 2.0)
    if in_segment:
        centers.append((start + len(column) - 1) / 2.0)
    return centers


def viterbi_trace(
    binary_region: np.ndarray,
    alpha: float = 0.5,
    max_jump: float | None = None,
) -> np.ndarray:
    """Extract optimal signal path using Viterbi algorithm.

    For each column x, find centers of contiguous signal-pixel segments.
    Edge cost = alpha * distance + (1-alpha) * |slope_change|.
    """
    height, width = binary_region.shape
    if max_jump is None:
        max_jump = height / 3.0

    # Build nodes per column.
    all_nodes: list[list[float]] = []
    for x in range(width):
        centers = _find_segment_centers(binary_region[:, x])
        all_nodes.append(centers)

    if all(len(n) == 0 for n in all_nodes):
        return np.full(width, height / 2.0, dtype=np.float32)

    # Forward pass.
    INF = 1e18
    cost: list[list[float]] = [[] for _ in range(width)]
    backptr: list[list[int]] = [[] for _ in range(width)]
    prev_slope: list[list[float]] = [[] for _ in range(width)]

    # Find and initialise first non-empty column.
    first_x = -1
    for x in range(width):
        if all_nodes[x]:
            cost[x] = [0.0] * len(all_nodes[x])
            backptr[x] = [-1] * len(all_nodes[x])
            prev_slope[x] = [0.0] * len(all_nodes[x])
            first_x = x
            break

    if first_x == -1:
        return np.full(width, height / 2.0, dtype=np.float32)

    last_valid_x = first_x
    for x in range(first_x + 1, width):
        if not all_nodes[x]:
            continue

        nodes_prev = all_nodes[last_valid_x]
        costs_prev = cost[last_valid_x]
        slopes_prev = prev_slope[last_valid_x]
        dx = float(x - last_valid_x)

        c_x: list[float] = []
        bp_x: list[int] = []
        sl_x: list[float] = []

        for yj in all_nodes[x]:
            best_cost = INF
            best_k = 0
            best_slope = 0.0

            for k, yk in enumerate(nodes_prev):
                dist = abs(yj - yk)
                if dist > max_jump:
                    continue
                new_slope = (yj - yk) / dx
                slope_change = abs(new_slope - slopes_prev[k])
                transition = alpha * dist + (1.0 - alpha) * slope_change
                total = costs_prev[k] + transition

                if total < best_cost:
                    best_cost = total
                    best_k = k
                    best_slope = new_slope

            if best_cost >= INF:
                best_cost = 0.0
                best_slope = 0.0
                best_k = -1

            c_x.append(best_cost)
            bp_x.append(best_k)
            sl_x.append(best_slope)

        cost[x] = c_x
        backptr[x] = bp_x
        prev_slope[x] = sl_x
        last_valid_x = x

    # Backward pass.
    trace = np.full(width, np.nan, dtype=np.float32)
    valid_cols = [x for x in range(width) if cost[x]]

    if not valid_cols:
        return np.full(width, height / 2.0, dtype=np.float32)

    last = valid_cols[-1]
    best_idx = int(np.argmin(cost[last]))
    trace[last] = all_nodes[last][best_idx]

    current_idx = best_idx
    for i in range(len(valid_cols) - 1, 0, -1):
        x = valid_cols[i]
        px = valid_cols[i - 1]
        parent_idx = backptr[x][current_idx] if current_idx < len(backptr[x]) else -1
        if 0 <= parent_idx < len(all_nodes[px]):
            trace[px] = all_nodes[px][parent_idx]
            current_idx = parent_idx
        elif all_nodes[px]:
            current_idx = len(all_nodes[px]) // 2
            trace[px] = all_nodes[px][current_idx]

    # Interpolate NaN gaps.
    valid = np.flatnonzero(np.isfinite(trace))
    if valid.size == 0:
        return np.full(width, height / 2.0, dtype=np.float32)
    if valid.size < width:
        missing = np.flatnonzero(~np.isfinite(trace))
        trace[missing] = np.interp(missing, valid, trace[valid])

    return trace


# ---------------------------------------------------------------------------
# Mask cleaning & layout detection
# ---------------------------------------------------------------------------

def clean_mask(mask: np.ndarray, min_component_area: int = 64) -> np.ndarray:
    binary = (mask > 127).astype(np.uint8)
    n_comp, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    for cid in range(1, n_comp):
        if stats[cid, cv2.CC_STAT_AREA] >= min_component_area:
            cleaned[labels == cid] = 255
    kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    k = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(values, k, mode="same")


def _projection_centers(projection: np.ndarray, groups: int) -> list[int]:
    smoothed = moving_average(projection.astype(np.float32), max(9, len(projection) // 80))
    centers: list[int] = []
    edges = np.linspace(0, len(smoothed), groups + 1, dtype=int)
    for s, e in zip(edges[:-1], edges[1:]):
        seg = smoothed[s:e]
        centers.append(s + int(np.argmax(seg)) if seg.size else s)
    return centers

def _bounds_from_centers(centers: list[int], limit: int) -> list[tuple[int, int]]:
    if len(centers) < 2:
        return [(0, limit)]
    b = [0]
    for l, r in zip(centers[:-1], centers[1:]):
        b.append((l + r) // 2)
    b.append(limit)
    return [(int(s), int(e)) for s, e in zip(b[:-1], b[1:])]

def estimate_layout(mask: np.ndarray) -> Layout:
    # 12-lead ECGs usually have 4 columns and 4 rows (3 standard rows + 1 rhythm strip at bottom).
    # We must split the Y-axis into 4 rows to correctly isolate the top 3 rows.
    binary = (mask > 127).astype(np.uint8)
    x_proj = binary.sum(axis=0)
    y_proj = binary.sum(axis=1)
    
    n_cols = len(GRID_LEAD_LAYOUT[0])  # 4
    n_rows_actual = len(GRID_LEAD_LAYOUT) + 1  # 3 standard rows + 1 rhythm strip = 4 rows
    
    col_bounds = _bounds_from_centers(_projection_centers(x_proj, n_cols), mask.shape[1])
    row_bounds = _bounds_from_centers(_projection_centers(y_proj, n_rows_actual), mask.shape[0])
    
    # We only care about the first 3 rows for the 12 standard leads.
    row_bounds = row_bounds[:len(GRID_LEAD_LAYOUT)]
    
    return Layout(column_bounds=col_bounds, row_bounds=row_bounds)


# ---------------------------------------------------------------------------
# Signal extraction with Viterbi + baseline detection + mV calibration
# ---------------------------------------------------------------------------

    pass

def _trim_calibration_pulse(trace: np.ndarray, threshold_factor: float = 3.0) -> np.ndarray:
    """Detect and replace the calibration pulse at the start of the trace.

    The calibration pulse is a tall rectangular spike in the first ~10% of the signal.
    Replace it with the baseline value to avoid corrupting the extracted signal.
    """
    n = len(trace)
    search_end = max(10, n // 10)  # first 10% of signal

    baseline = float(np.median(trace))
    mad = float(np.median(np.abs(trace - baseline)))
    if mad < 1e-6:
        return trace

    result = trace.copy()
    pulse_thresh = threshold_factor * mad
    for i in range(search_end):
        if abs(result[i] - baseline) > pulse_thresh:
            result[i] = baseline
        else:
            break  # stop at first non-pulse sample

    return result


def extract_signal_from_region(
    region_mask: np.ndarray,
    target_length: int,
    pixels_per_mv: float,
    use_viterbi: bool = True,
) -> np.ndarray:
    """Extract a 1-D ECG signal from a binary mask region.

    Uses Viterbi for optimal path, baseline detection for DC offset,
    calibration pulse trimming, and pixel→mV conversion.
    """
    binary = (region_mask > 127).astype(np.uint8)
    height, width = binary.shape

    if binary.sum() == 0:
        return np.zeros(target_length, dtype=np.float32)

    # --- Path extraction ---
    if use_viterbi and width > 20:
        trace = viterbi_trace(binary * 255)
    else:
        # Fallback: per-column median.
        trace = np.full(width, np.nan, dtype=np.float32)
        for x in range(width):
            ys = np.flatnonzero(binary[:, x])
            if ys.size:
                trace[x] = float(np.median(ys))
        valid = np.flatnonzero(np.isfinite(trace))
        if valid.size == 0:
            return np.zeros(target_length, dtype=np.float32)
        if valid.size < width:
            missing = np.flatnonzero(~np.isfinite(trace))
            trace[missing] = np.interp(missing, valid, trace[valid])

    # --- Smoothing ---
    trace = moving_average(trace, max(3, trace.size // 200))

    # --- Baseline detection: ISO-electric line is usually the median of the trace itself ---
    valid_trace = trace[np.isfinite(trace)]
    baseline_px = float(np.median(valid_trace)) if valid_trace.size > 0 else height / 2.0

    # --- Convert pixel y → millivolts ---
    # In images, y=0 is top, y=max is bottom.
    # So if trace > baseline_px, it is physically LOWER on the paper.
    # Therefore real voltage = baseline_px - trace
    signal_mv = (baseline_px - trace) / pixels_per_mv

    # --- Trim calibration pulse ---
    signal_mv = _trim_calibration_pulse(signal_mv)

    # --- Resample to target_length ---
    src_x = np.linspace(0.0, 1.0, num=signal_mv.size, dtype=np.float32)
    tgt_x = np.linspace(0.0, 1.0, num=target_length, dtype=np.float32)
    return np.interp(tgt_x, src_x, signal_mv).astype(np.float32)


# ---------------------------------------------------------------------------
# Top-level digitisation
# ---------------------------------------------------------------------------

def _get_gray_for_calibration(image_source: np.ndarray | None) -> np.ndarray | None:
    """Convert image to grayscale if available, for FFT grid detection."""
    if image_source is None:
        return None
    if len(image_source.shape) == 2:
        return image_source
    return cv2.cvtColor(image_source, cv2.COLOR_RGB2GRAY)


def digitize_mask(mask: np.ndarray, num_samples: int,
                  image_rgb: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Convert a predicted segmentation mask to a dict of lead → signal (mV)."""
    cleaned = clean_mask(mask)
    layout = estimate_layout(cleaned)

    gray = _get_gray_for_calibration(image_rgb)
    pixels_per_mv = estimate_pixels_per_mv(mask.shape[1], mask.shape[0], gray)

    signals: dict[str, np.ndarray] = {}

    for row_index, lead_names in enumerate(GRID_LEAD_LAYOUT):
        y0, y1 = layout.row_bounds[row_index]
        # Minimal padding
        hm = max(1, (y1 - y0) // 20)
        y0, y1 = max(0, y0 + hm), min(mask.shape[0], y1 - hm)

        for col_index, lead_name in enumerate(lead_names):
            x0, x1 = layout.column_bounds[col_index]
            wm = max(1, (x1 - x0) // 20)
            x0, x1 = max(0, x0 + wm), min(mask.shape[1], x1 - wm)

            region = cleaned[y0:y1, x0:x1]
            signals[canonical_lead_name(lead_name)] = extract_signal_from_region(
                region, target_length=num_samples, pixels_per_mv=pixels_per_mv,
            )

    for lead_name in STANDARD_LEADS:
        signals.setdefault(lead_name, np.zeros(num_samples, dtype=np.float32))

    return {ln: signals[ln] for ln in STANDARD_LEADS}


def digitize_image_adaptive(
    image_rgb: np.ndarray, num_samples: int,
) -> dict[str, np.ndarray]:
    """Full adaptive pipeline: Otsu threshold → Viterbi → mV. No model needed."""
    binary = adaptive_otsu_threshold(image_rgb)
    cleaned = clean_mask(binary, min_component_area=max(10, binary.size // 80000))
    layout = estimate_layout(cleaned)

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    pixels_per_mv = estimate_pixels_per_mv(image_rgb.shape[1], image_rgb.shape[0], gray)

    signals: dict[str, np.ndarray] = {}

    for row_index, lead_names in enumerate(GRID_LEAD_LAYOUT):
        y0, y1 = layout.row_bounds[row_index]
        # Minimal padding
        hm = max(1, (y1 - y0) // 20)
        y0, y1 = max(0, y0 + hm), min(mask.shape[0], y1 - hm)

        for col_index, lead_name in enumerate(lead_names):
            x0, x1 = layout.column_bounds[col_index]
            wm = max(1, (x1 - x0) // 20)
            x0, x1 = max(0, x0 + wm), min(mask.shape[1], x1 - wm)

            region = cleaned[y0:y1, x0:x1]
            signals[canonical_lead_name(lead_name)] = extract_signal_from_region(
                region, target_length=num_samples, pixels_per_mv=pixels_per_mv,
                use_viterbi=True,
            )

    for lead_name in STANDARD_LEADS:
        signals.setdefault(lead_name, np.zeros(num_samples, dtype=np.float32))

    return {ln: signals[ln] for ln in STANDARD_LEADS}


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
        p for p in args.mask_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
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
    model, ckpt_img_size, ckpt_threshold = load_checkpoint(args.checkpoint, device)
    image_size = args.image_size or ckpt_img_size
    threshold = args.threshold if args.threshold is not None else ckpt_threshold

    records: dict[str, dict[str, np.ndarray]] = {}
    if args.mask_output_dir is not None:
        args.mask_output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in args.input_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.input_dir}")

    for image_path in tqdm(image_paths, desc="pipeline"):
        image_rgb = load_rgb_image(image_path)
        mask = predict_mask_array(model, image_rgb, image_size=image_size,
                                  threshold=threshold, device=device)
        if args.mask_output_dir is not None:
            cv2.imwrite(str(args.mask_output_dir / image_path.name), mask)
        # Pass image_rgb for FFT-based amplitude calibration.
        records[record_name_from_path(image_path)] = digitize_mask(
            mask, num_samples=args.num_samples, image_rgb=image_rgb,
        )

    save_submission(records, args.output)
    print(f"saved submission archive to {args.output}")


def run_adaptive_pipeline(args) -> None:
    """Run full adaptive Otsu + Viterbi pipeline (no trained model needed)."""
    records: dict[str, dict[str, np.ndarray]] = {}
    image_paths = sorted(
        p for p in args.input_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.input_dir}")

    for image_path in tqdm(image_paths, desc="adaptive-pipeline"):
        image_rgb = load_rgb_image(image_path)
        records[record_name_from_path(image_path)] = digitize_image_adaptive(
            image_rgb, num_samples=args.num_samples,
        )

    save_submission(records, args.output)
    print(f"saved submission archive to {args.output}")
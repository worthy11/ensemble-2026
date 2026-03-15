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
    """Detect if the binary image still contains grid lines.

    Grid lines are long horizontal/vertical structures that span a significant
    fraction of the image width/height.
    """
    h, w = binary.shape
    # Check for horizontal lines
    horiz_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(20, int(w * min_line_length_ratio)), 1)
    )
    horiz = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel)
    if horiz.sum() > 0:
        return True
    # Check for vertical lines
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
    """Adaptive thresholding: Otsu + iterative hedging factor to remove grid.

    Steps:
      1. Convert to grayscale, compute Otsu threshold.
      2. Start with hedging_factor = 1.0.
      3. Binarise with threshold = otsu_thresh × hedging_factor.
      4. Reduce hedging_factor by 5% and repeat while grid lines are detected.
      5. Stop when grid lines disappear or hedging_factor < 0.6.

    Returns a uint8 binary mask (0/255) of the ECG signal.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    # Invert: ECG traces are dark lines on light background.
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

    # Light denoising: remove small components.
    kernel = np.ones((3, 3), dtype=np.uint8)
    best_binary = cv2.morphologyEx(best_binary, cv2.MORPH_OPEN, kernel)
    best_binary = cv2.morphologyEx(best_binary, cv2.MORPH_CLOSE, kernel)

    return best_binary


# ---------------------------------------------------------------------------
# Viterbi-based signal path extraction
# ---------------------------------------------------------------------------

def _find_segment_centers(column: np.ndarray) -> list[float]:
    """Find centres of contiguous True-pixel runs in a binary column."""
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
    Build a graph where nodes = segment centers, edges connect adjacent columns.
    Edge cost = alpha * euclidean_distance + (1-alpha) * |slope_change|.
    Find the minimum-cost path through the entire image width.

    Args:
        binary_region: uint8 mask (0/255), height × width.
        alpha: weight balancing distance vs. slope change (0.5 = equal).
        max_jump: maximum pixel distance allowed between adjacent nodes.
                  None = height/4 (reasonable default).

    Returns:
        float32 array of length = width, with the y-coordinate of the trace
        for each column.  NaN-free (gaps are interpolated).
    """
    height, width = binary_region.shape
    if max_jump is None:
        max_jump = height / 4.0

    # Build nodes per column.
    all_nodes: list[list[float]] = []
    for x in range(width):
        centers = _find_segment_centers(binary_region[:, x])
        all_nodes.append(centers)

    # Fallback: if no nodes found, return flat baseline.
    if all(len(nodes) == 0 for nodes in all_nodes):
        return np.full(width, height / 2.0, dtype=np.float32)

    # Forward pass: compute cheapest cost to reach each node.
    INF = 1e18
    # cost[x] = list of costs for each node at column x
    cost: list[list[float]] = []
    backptr: list[list[int]] = []  # backpointer to previous node index
    prev_slope: list[list[float]] = []  # slope arriving at each node

    # Initialise first non-empty column.
    first_x = -1
    for x in range(width):
        if all_nodes[x]:
            cost.append([0.0] * len(all_nodes[x]))
            backptr.append([-1] * len(all_nodes[x]))
            prev_slope.append([0.0] * len(all_nodes[x]))
            first_x = x
            break
        else:
            cost.append([])
            backptr.append([])
            prev_slope.append([])

    if first_x == -1:
        return np.full(width, height / 2.0, dtype=np.float32)

    # Process remaining columns.
    last_valid_x = first_x
    for x in range(first_x + 1, width):
        nodes_x = all_nodes[x]
        if not nodes_x:
            cost.append([])
            backptr.append([])
            prev_slope.append([])
            continue

        nodes_prev = all_nodes[last_valid_x]
        costs_prev = cost[last_valid_x]
        slopes_prev = prev_slope[last_valid_x]
        dx = float(x - last_valid_x)

        c_x: list[float] = []
        bp_x: list[int] = []
        sl_x: list[float] = []

        for j, yj in enumerate(nodes_x):
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
                # No valid predecessor — start fresh from this node.
                best_cost = 0.0
                best_slope = 0.0
                best_k = -1

            c_x.append(best_cost)
            bp_x.append(best_k)
            sl_x.append(best_slope)

        cost.append(c_x)
        backptr.append(bp_x)
        prev_slope.append(sl_x)
        last_valid_x = x

    # Backward pass: trace the optimal path.
    trace = np.full(width, np.nan, dtype=np.float32)

    # Find best terminal node.
    best_end_cost = INF
    best_end_idx = 0
    for x in range(width - 1, -1, -1):
        if cost[x]:
            for j, c in enumerate(cost[x]):
                if c < best_end_cost:
                    best_end_cost = c
                    best_end_idx = j
            # Trace back from this column.
            trace[x] = all_nodes[x][best_end_idx]
            current_idx = best_end_idx

            prev_x = x
            for bx in range(x - 1, -1, -1):
                if not backptr[bx + 1 if bx + 1 <= prev_x else prev_x]:
                    continue
                if bx + 1 <= prev_x and backptr[prev_x] and current_idx < len(backptr[prev_x]):
                    parent = backptr[prev_x][current_idx]
                    # Walk back to the actual previous valid column.
                    # Find prev_x's predecessor.
                    pass

            break

    # Simpler backward trace: walk from last valid column to first.
    # Rebuild: find all valid columns in order.
    valid_cols = [x for x in range(width) if cost[x]]

    if not valid_cols:
        return np.full(width, height / 2.0, dtype=np.float32)

    # Start from the last valid column.
    last = valid_cols[-1]
    best_idx = int(np.argmin(cost[last]))
    trace[last] = all_nodes[last][best_idx]

    current_idx = best_idx
    for i in range(len(valid_cols) - 1, 0, -1):
        x = valid_cols[i]
        px = valid_cols[i - 1]
        parent_idx = backptr[x][current_idx] if current_idx < len(backptr[x]) else -1
        if parent_idx >= 0 and parent_idx < len(all_nodes[px]):
            trace[px] = all_nodes[px][parent_idx]
            current_idx = parent_idx
        else:
            # Broken chain — use median fallback for this column.
            if all_nodes[px]:
                current_idx = len(all_nodes[px]) // 2
                trace[px] = all_nodes[px][current_idx]

    # Interpolate any remaining NaN gaps.
    valid = np.flatnonzero(np.isfinite(trace))
    if valid.size == 0:
        return np.full(width, height / 2.0, dtype=np.float32)
    if valid.size < width:
        missing = np.flatnonzero(~np.isfinite(trace))
        trace[missing] = np.interp(missing, valid, trace[valid])

    return trace


# ---------------------------------------------------------------------------
# Amplitude calibration
# ---------------------------------------------------------------------------

def estimate_pixels_per_mv(mask_width: int) -> float:
    """Estimate the pixel-to-millivolt scale from the image (mask) width.

    Standard ECG layout:
      - Paper speed = 25 mm/s  →  2.5 s × 25 mm/s = 62.5 mm per column
      - 4 columns               →  250 mm total useful width
      - Gain = 10 mm/mV         →  pixels_per_mv = (mask_width / 250) × 10
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
# Signal extraction with Viterbi + mV calibration
# ---------------------------------------------------------------------------

def extract_signal_from_region(
    region_mask: np.ndarray,
    target_length: int,
    pixels_per_mv: float,
    use_viterbi: bool = True,
) -> np.ndarray:
    """Extract a 1-D ECG signal from a binary mask region.

    Uses Viterbi path tracing (if enabled) for a smooth, optimal path,
    then converts pixel coordinates to millivolts using the calibration.

    Args:
        region_mask:  Mask for a single lead panel (uint8, 0/255).
        target_length: Number of output samples.
        pixels_per_mv: Calibration — how many pixel rows equal 1 mV.
        use_viterbi: If True, use Viterbi path tracing (recommended).

    Returns:
        float32 array of length `target_length`, values in millivolts.
    """
    binary = (region_mask > 127).astype(np.uint8)
    height, width = binary.shape

    if binary.sum() == 0:
        return np.zeros(target_length, dtype=np.float32)

    if use_viterbi:
        trace = viterbi_trace(binary * 255)
    else:
        # Fallback: per-column median (original method).
        trace = np.full(width, np.nan, dtype=np.float32)
        for x in range(width):
            ys = np.flatnonzero(binary[:, x])
            if ys.size:
                trace[x] = float(np.median(ys))

        valid = np.flatnonzero(np.isfinite(trace))
        if valid.size == 0:
            return np.zeros(target_length, dtype=np.float32)
        if valid.size == 1:
            trace[:] = trace[valid[0]]
        else:
            missing = np.flatnonzero(~np.isfinite(trace))
            trace[missing] = np.interp(missing, valid, trace[valid])

    # Light smoothing.
    trace = moving_average(trace, max(3, trace.size // 200))

    # Convert pixel y → millivolts.
    baseline_px = height / 2.0
    signal_mv = -(trace - baseline_px) / pixels_per_mv

    # Resample to target_length.
    src_x = np.linspace(0.0, 1.0, num=signal_mv.size, dtype=np.float32)
    tgt_x = np.linspace(0.0, 1.0, num=target_length, dtype=np.float32)
    return np.interp(tgt_x, src_x, signal_mv).astype(np.float32)


# ---------------------------------------------------------------------------
# Top-level digitisation functions
# ---------------------------------------------------------------------------

def digitize_mask(mask: np.ndarray, num_samples: int) -> dict[str, np.ndarray]:
    """Convert a predicted segmentation mask to a dict of lead → signal (mV)."""
    cleaned = clean_mask(mask)
    layout = estimate_layout(cleaned)
    pixels_per_mv = estimate_pixels_per_mv(mask.shape[1])

    signals: dict[str, np.ndarray] = {}

    for row_index, lead_names in enumerate(GRID_LEAD_LAYOUT):
        y0, y1 = layout.row_bounds[row_index]
        height_margin = max(4, (y1 - y0) // 12)
        y0 = min(y1, y0 + height_margin)
        y1 = max(y0 + 1, y1 - height_margin)

        for column_index, lead_name in enumerate(lead_names):
            x0, x1 = layout.column_bounds[column_index]
            width_margin = max(4, (x1 - x0) // 12)
            x0 = min(x1, x0 + width_margin)
            x1 = max(x0 + 1, x1 - width_margin)

            region = cleaned[y0:y1, x0:x1]
            signals[canonical_lead_name(lead_name)] = extract_signal_from_region(
                region,
                target_length=num_samples,
                pixels_per_mv=pixels_per_mv,
            )

    for lead_name in STANDARD_LEADS:
        signals.setdefault(lead_name, np.zeros(num_samples, dtype=np.float32))

    return {lead_name: signals[lead_name] for lead_name in STANDARD_LEADS}


def digitize_image_adaptive(
    image_rgb: np.ndarray,
    num_samples: int,
) -> dict[str, np.ndarray]:
    """Full adaptive pipeline: Otsu threshold → clean → layout → Viterbi → mV.

    This does NOT need a trained model — it uses classical CV only.
    """
    binary = adaptive_otsu_threshold(image_rgb)
    cleaned = clean_mask(binary, min_component_area=max(10, binary.size // 80000))
    layout = estimate_layout(cleaned)
    pixels_per_mv = estimate_pixels_per_mv(image_rgb.shape[1])

    signals: dict[str, np.ndarray] = {}

    for row_index, lead_names in enumerate(GRID_LEAD_LAYOUT):
        y0, y1 = layout.row_bounds[row_index]
        height_margin = max(4, (y1 - y0) // 12)
        y0 = min(y1, y0 + height_margin)
        y1 = max(y0 + 1, y1 - height_margin)

        for column_index, lead_name in enumerate(lead_names):
            x0, x1 = layout.column_bounds[column_index]
            width_margin = max(4, (x1 - x0) // 12)
            x0 = min(x1, x0 + width_margin)
            x1 = max(x0 + 1, x1 - width_margin)

            region = cleaned[y0:y1, x0:x1]
            signals[canonical_lead_name(lead_name)] = extract_signal_from_region(
                region,
                target_length=num_samples,
                pixels_per_mv=pixels_per_mv,
                use_viterbi=True,
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


def run_adaptive_pipeline(args) -> None:
    """Run full adaptive Otsu + Viterbi pipeline (no trained model needed)."""
    records: dict[str, dict[str, np.ndarray]] = {}

    image_paths = sorted(
        path for path in args.input_dir.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
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
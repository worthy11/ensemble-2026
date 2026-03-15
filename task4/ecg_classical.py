from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from ecg_common import record_name_from_path
from ecg_digitization import clean_mask, digitize_mask, save_submission
from ecg_segmentation import load_rgb_image


def build_classical_trace_mask(image_rgb):
    """Extract ECG trace mask directly from image using classical CV operations.

    Pipeline:
      1. HSV: detect colored grid pixels (pink/blue/red dots and lines).
      2. Percentile-based brightness threshold within the paper area (adapts to
         each image without inpainting, which caused large blob artifacts).
      3. Morphological grid-line removal (long horizontal/vertical structures).
      4. Horizontal closing to bridge short breaks along the time axis.
      5. Component cleanup — remove tiny isolated blobs.
    """
    h, w = image_rgb.shape[:2]

    # ── Step 1: HSV grid detection ──────────────────────────────────────────
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    hh, ss, vv = cv2.split(hsv)

    # Colored grid: saturated (not grey) AND not very dark.
    # Typical pink/red/blue ECG grid is S > 30, V > 80.
    grid_colored = (ss > 30) & (vv > 80)

    # ── Step 2: Paper area and percentile threshold ──────────────────────────
    # Paper = not the black border (border has V < ~50).
    paper = vv > 50

    # Within the paper, the trace is the darkest ink.
    # Use the 20th percentile of paper-pixel brightness as threshold.
    # This adapts per-image to varying scan contrast without needing inpainting.
    paper_vals = vv[paper]
    if paper_vals.size > 0:
        pct20 = int(np.percentile(paper_vals, 20))
        # Clamp: never go above 200 (avoid treating grey background as trace)
        #        never go below 60  (avoid losing trace on very dark scans)
        trace_vthresh = max(60, min(200, pct20))
    else:
        trace_vthresh = 120

    # Trace candidates: dark ink on paper, not colored grid.
    trace_candidates = (vv < trace_vthresh) & paper & ~grid_colored
    trace_mask = trace_candidates.astype(np.uint8) * 255

    # ── Step 3: Remove long straight grid lines ──────────────────────────────
    # Genuine trace segments are rarely wider than the lead strip is tall,
    # but grid lines span the full width/height.
    horiz_len = max(20, w // 60)
    vert_len  = max(20, h // 60)
    horiz_lines = cv2.morphologyEx(
        trace_mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_len, 1)))
    vert_lines = cv2.morphologyEx(
        trace_mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_len)))
    grid_lines = cv2.bitwise_or(horiz_lines, vert_lines)
    trace_mask = cv2.subtract(trace_mask, grid_lines)

    # ── Step 4: Bridge gaps along the time axis ──────────────────────────────
    # ECG traces run horizontally; close gaps up to ~1 % of image width.
    gap_h = max(7, w // 100)
    trace_mask = cv2.morphologyEx(
        trace_mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (gap_h, 1)))
    # Small circular close for tiny vertical discontinuities.
    trace_mask = cv2.morphologyEx(
        trace_mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    # ── Step 5: Remove tiny isolated noise blobs ─────────────────────────────
    min_area = max(10, (w * h) // 80000)
    trace_mask = clean_mask(trace_mask, min_component_area=min_area)

    return trace_mask


def run_classical_pipeline(args) -> None:
    records = {}
    if args.mask_output_dir is not None:
        args.mask_output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(path for path in args.input_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.input_dir}")

    if args.test_limit is not None and args.test_limit > 0:
        image_paths = image_paths[: args.test_limit]

    for image_path in tqdm(image_paths, desc="classical-pipeline"):
        image_rgb = load_rgb_image(image_path)
        mask = build_classical_trace_mask(image_rgb)
        if args.mask_output_dir is not None:
            cv2.imwrite(str(args.mask_output_dir / image_path.name), mask)
        records[record_name_from_path(image_path)] = digitize_mask(mask, num_samples=args.num_samples, image_rgb=image_rgb)

    save_submission(records, args.output)
    print(f"saved submission archive to {args.output}")

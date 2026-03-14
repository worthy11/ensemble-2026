"""Quick test to compare classical mask approaches."""
import argparse
import sys
sys.path.insert(0, "task4")

import cv2
import numpy as np
from ecg_classical import build_classical_trace_mask


def hsv_trace_mask(image_rgb):
    """HSV-based trace extraction via adaptive thresholding + color filter."""
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # Step 1: Adaptive threshold on grayscale.
    # Detects pixels significantly darker than their local neighborhood.
    # blockSize scales with image size; C controls sensitivity.
    long_edge = max(gray.shape[0], gray.shape[1])
    block_size = long_edge // 40 | 1  # ~2.5% of image, must be odd
    if block_size % 2 == 0:
        block_size += 1
    block_size = max(block_size, 15)

    # THRESH_BINARY_INV: output 255 where gray < (local_mean - C)
    trace_mask = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, 12
    )

    # Step 2: HSV color filter — remove bright, saturated grid remnants.
    # The grid has high saturation AND high value; the trace is darker.
    median_s = float(np.median(s))
    if median_s > 20:
        # Keep only pixels that are NOT (bright and saturated)
        colored_bright = ((s > 40) & (v > 120)).astype(np.uint8) * 255
        trace_mask = cv2.subtract(trace_mask, colored_bright)

    # Step 3: Remove border artifacts.
    # Black borders around the ECG paper create false positives at edges.
    # Detect the border by looking at very dark regions.
    border_mask = (v < 50).astype(np.uint8) * 255
    border_dilated = cv2.dilate(border_mask, np.ones((5, 5), np.uint8))
    trace_mask = cv2.subtract(trace_mask, border_dilated)

    # Step 4: Morphological cleanup.
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    trace_mask = cv2.morphologyEx(trace_mask, cv2.MORPH_CLOSE, close_kernel)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(trace_mask, connectivity=8)
    cleaned = np.zeros_like(trace_mask)
    for label_id in range(1, n_labels):
        if stats[label_id, cv2.CC_STAT_AREA] >= 50:
            cleaned[labels == label_id] = 255

    return cleaned


def main():
    import os
    parser = argparse.ArgumentParser(description="Compare current classical mask vs HSV variant.")
    parser.add_argument("--start", type=int, default=1, help="Starting image index, e.g. 1 for ecg_train_0001.png")
    parser.add_argument("--limit", type=int, default=20, help="Number of consecutive images to test")
    args = parser.parse_args()

    os.makedirs("data/mask_comparison", exist_ok=True)

    end = args.start + args.limit
    for i in range(args.start, end):
        path = f"train/ecg_train_{i:04d}.png"
        if not os.path.exists(path):
            continue
        img_bgr = cv2.imread(path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        curr = build_classical_trace_mask(img_rgb)
        new = hsv_trace_mask(img_rgb)

        curr_px = (curr > 127).sum()
        new_px = (new > 127).sum()
        total = img_rgb.shape[0] * img_rgb.shape[1]
        print(f"img {i:04d}: current={curr_px} ({100*curr_px/total:.1f}%) | "
              f"hsv={new_px} ({100*new_px/total:.1f}%)", flush=True)

        cv2.imwrite(f"data/mask_comparison/{i:04d}_current.png", curr)
        cv2.imwrite(f"data/mask_comparison/{i:04d}_hsv.png", new)

    print("Masks saved to data/mask_comparison/")


if __name__ == "__main__":
    main()

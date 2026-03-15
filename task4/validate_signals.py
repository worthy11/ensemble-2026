"""Validation script: compare extracted signals against WFDB ground truth.

Usage (from task4/):
    python validate_signals.py \
        --source-dir ../ecg_dataset/train \
        --checkpoint ../artifacts/unet_resnet50.pt \
        --num-images 5

Prints per-lead Pearson correlation, amplitude comparison, and key diagnostics.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ecg_common import GRID_LEAD_LAYOUT, STANDARD_LEADS, canonical_lead_name
from ecg_data import load_wfdb_signals
from ecg_digitization import digitize_mask, digitize_image_adaptive
from ecg_segmentation import load_checkpoint, load_rgb_image, predict_mask_array


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two 1-D arrays."""
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def snr_db(extracted: np.ndarray, gt: np.ndarray) -> float:
    """Signal-to-Noise Ratio in dB."""
    noise = extracted - gt
    signal_power = np.mean(gt ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power < 1e-12:
        return 100.0
    return float(10 * np.log10(signal_power / noise_power))


def analyze_one_image(
    image_path: Path,
    source_dir: Path,
    model=None,
    image_size: int = 512,
    threshold: float = 0.5,
    device=None,
    use_adaptive: bool = False,
    num_samples: int = 1250,
) -> dict:
    """Run extraction pipeline on one training image and compare vs GT."""
    import torch

    record_name = image_path.stem
    hea_path = source_dir / f"{record_name}.hea"

    if not hea_path.exists():
        return {"error": f"No .hea file for {record_name}"}

    # Load ground truth.
    gt_signals = load_wfdb_signals(hea_path)

    # Load image.
    image_rgb = load_rgb_image(image_path)

    # Extract signals.
    if use_adaptive:
        extracted = digitize_image_adaptive(image_rgb, num_samples=num_samples)
    else:
        if model is None or device is None:
            return {"error": "Model not loaded"}
        mask = predict_mask_array(model, image_rgb, image_size=image_size,
                                  threshold=threshold, device=device)
        extracted = digitize_mask(mask, num_samples=num_samples, image_rgb=image_rgb)

    # Compare each lead.
    results = {}
    for lead_name in STANDARD_LEADS:
        ext = extracted.get(lead_name)
        gt_full = gt_signals.get(lead_name)

        if ext is None or gt_full is None:
            results[lead_name] = {"error": "missing"}
            continue

        # If it's a 3-channel rhythm strip at the bottom (V1, II, V5) it would be full 5000 samples,
        # but our extraction pipeline currently only outputs 1250 samples for all leads.
        # Find which column this lead is in the 3x4 grid to slice the matching GT.
        col_idx = None
        for row in GRID_LEAD_LAYOUT:
            for ci, ln in enumerate(row):
                if canonical_lead_name(ln) == lead_name:
                    col_idx = ci
                    break
            if col_idx is not None:
                break

        if col_idx is not None and len(gt_full) >= 5000:
            start = col_idx * 1250
            end = start + 1250
            gt_slice = gt_full[start:end]
        else:
            # Fallback for unmapped or short leads.
            gt_slice = gt_full[:len(ext)]

        if len(ext) == 0:
            results[lead_name] = {"error": "extracted empty"}
            continue

        # Resample extracted if lengths differ.
        if len(ext) != len(gt_slice):
            src_x = np.linspace(0, 1, len(ext))
            tgt_x = np.linspace(0, 1, len(gt_slice))
            ext_resampled = np.interp(tgt_x, src_x, ext)
        else:
            ext_resampled = ext

        corr = pearson_corr(ext_resampled, gt_slice)
        snr = snr_db(ext_resampled, gt_slice)

        results[lead_name] = {
            "pearson": corr,
            "snr_db": snr,
            "ext_range": [float(ext.min()), float(ext.max())],
            "gt_range": [float(gt_slice.min()), float(gt_slice.max())],
            "ext_std": float(ext.std()),
            "gt_std": float(gt_slice.std()),
            "ext_mean": float(ext.mean()),
            "gt_mean": float(gt_slice.mean()),
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Validate ECG extraction against WFDB ground truth")
    parser.add_argument("--source-dir", type=Path, required=True, help="Train directory with .png/.hea/.dat")
    parser.add_argument("--checkpoint", type=Path, default=None, help="UNet checkpoint (omit for adaptive mode)")
    parser.add_argument("--num-images", type=int, default=5, help="Number of images to validate")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--num-samples", type=int, default=1250)
    args = parser.parse_args()

    import torch

    # Find training images.
    image_paths = sorted(
        p for p in args.source_dir.iterdir()
        if p.suffix.lower() == ".png" and (args.source_dir / f"{p.stem}.hea").exists()
    )[:args.num_images]

    if not image_paths:
        print("No image/hea pairs found!")
        return

    use_adaptive = args.checkpoint is None
    model = None
    device = None

    if not use_adaptive:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, _, _ = load_checkpoint(args.checkpoint, device)
        print(f"Loaded model on {device}\n")
    else:
        print("Using adaptive (Otsu+Viterbi) mode — no model\n")

    all_pearsons = []
    all_snrs = []

    for img_path in image_paths:
        print(f"{'='*60}")
        print(f"Record: {img_path.stem}")
        print(f"{'='*60}")

        results = analyze_one_image(
            img_path, args.source_dir,
            model=model, image_size=args.image_size,
            threshold=0.5, device=device,
            use_adaptive=use_adaptive,
            num_samples=args.num_samples,
        )

        if "error" in results:
            print(f"  ERROR: {results['error']}")
            continue

        print(f"  {'Lead':<6} {'Pearson':>8} {'SNR(dB)':>8} {'Ext Range':>20} {'GT Range':>20} {'Ext σ':>8} {'GT σ':>8}")
        print(f"  {'-'*80}")

        for lead_name in STANDARD_LEADS:
            r = results.get(lead_name, {})
            if "error" in r:
                print(f"  {lead_name:<6} {'MISSING':>8}")
                continue

            p = r["pearson"]
            s = r["snr_db"]
            er = r["ext_range"]
            gr = r["gt_range"]
            all_pearsons.append(p)
            all_snrs.append(s)

            print(f"  {lead_name:<6} {p:>8.3f} {s:>8.1f} [{er[0]:>8.4f}, {er[1]:>8.4f}] [{gr[0]:>8.4f}, {gr[1]:>8.4f}] {r['ext_std']:>8.4f} {r['gt_std']:>8.4f}")

        print()

    if all_pearsons:
        print(f"{'='*60}")
        print(f"OVERALL SUMMARY ({len(all_pearsons)} lead measurements)")
        print(f"{'='*60}")
        print(f"  Mean Pearson: {np.mean(all_pearsons):.4f}")
        print(f"  Mean SNR(dB): {np.mean(all_snrs):.1f}")
        print(f"  Pearson > 0.5: {sum(1 for p in all_pearsons if p > 0.5)}/{len(all_pearsons)}")
        print(f"  Pearson > 0.7: {sum(1 for p in all_pearsons if p > 0.7)}/{len(all_pearsons)}")
        print()

        # Key diagnostics.
        mean_p = np.mean(all_pearsons)
        if mean_p < 0.1:
            print("  ⚠️  DIAGNOSIS: Signal may be INVERTED (y-axis flip) or completely wrong shape.")
            print("     → Check if -(trace - baseline) should be +(trace - baseline)")
        elif mean_p < 0.3:
            print("  ⚠️  DIAGNOSIS: Weak correlation. Mask quality may be the bottleneck.")
            print("     → Train longer or check mask alignment.")
        elif mean_p < 0.6:
            print("  ℹ️  DIAGNOSIS: Moderate correlation. Post-processing improvements may help.")
        else:
            print("  ✅ DIAGNOSIS: Good correlation! Focus on amplitude and time calibration.")


if __name__ == "__main__":
    main()

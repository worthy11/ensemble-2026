from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from ecg_common import canonical_lead_name, list_images


@dataclass
class DatasetSummary:
    train_count: int
    val_count: int
    output_dir: Path


def load_metadata(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def render_trace_mask(metadata: dict, image_shape: tuple[int, int], thickness: int) -> np.ndarray:
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)

    for lead in metadata["leads"]:
        pixels = lead.get("plotted_pixels", [])
        if len(pixels) < 2:
            continue

        points = np.asarray(pixels, dtype=np.float32)
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        points = np.round(points).astype(np.int32)
        cv2.polylines(mask, [points], isClosed=False, color=255, thickness=thickness)

    return mask


def write_mask_for_image(image_path: Path, json_path: Path, output_mask_path: Path, thickness: int) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    metadata = load_metadata(json_path)
    mask = render_trace_mask(metadata, image.shape[:2], thickness=thickness)
    output_mask_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_mask_path), mask)


def generate_masks(source_dir: Path, output_dir: Path, thickness: int, limit: int | None = None) -> int:
    image_paths = list_images(source_dir)
    if limit is not None:
        image_paths = image_paths[:limit]

    generated = 0
    for image_path in tqdm(image_paths, desc="generate-masks"):
        json_path = source_dir / f"{image_path.stem}.json"
        if not json_path.exists():
            continue

        write_mask_for_image(image_path, json_path, output_dir / f"{image_path.stem}.png", thickness)
        generated += 1

    return generated


def load_wfdb_signals(hea_path: Path) -> dict[str, np.ndarray]:
    """Load ground-truth ECG signals from a WFDB .hea/.dat file pair.

    Returns a dict mapping canonical lead name → 1-D float32 array in millivolts.
    Uses the official ``wfdb`` library which automatically applies gain/baseline.
    """
    try:
        import wfdb  # lazy import — only needed when GT signals are available
    except ImportError as exc:
        raise ImportError("Install the 'wfdb' package:  pip install wfdb") from exc

    record_stem = str(hea_path.with_suffix(""))
    signals, fields = wfdb.rdsamp(record_stem)
    # signals: (n_samples, n_leads) float64, already in physical units (mV)
    out: dict[str, np.ndarray] = {}
    for i, name in enumerate(fields["sig_name"]):
        out[canonical_lead_name(name)] = signals[:, i].astype(np.float32)
    return out


def prepare_dataset(
    source_dir: Path,
    output_dir: Path,
    val_ratio: float,
    seed: int,
    thickness: int,
    limit: int | None = None,
) -> DatasetSummary:
    image_paths = list_images(source_dir)
    pairs: list[tuple[Path, Path]] = []

    for image_path in image_paths:
        json_path = source_dir / f"{image_path.stem}.json"
        if json_path.exists():
            pairs.append((image_path, json_path))

    if not pairs:
        raise FileNotFoundError(f"No image/json pairs found in {source_dir}")

    if limit is not None:
        pairs = pairs[:limit]

    random.Random(seed).shuffle(pairs)

    val_count = int(round(len(pairs) * val_ratio))
    if len(pairs) > 1:
        val_count = min(max(val_count, 1), len(pairs) - 1)
    else:
        val_count = 0

    val_pairs = pairs[:val_count]
    train_pairs = pairs[val_count:]

    for split_name, split_pairs in (("train", train_pairs), ("val", val_pairs)):
        images_dir = output_dir / split_name / "images"
        masks_dir = output_dir / split_name / "masks"
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        for image_path, json_path in tqdm(split_pairs, desc=f"prepare-{split_name}"):
            shutil.copy2(image_path, images_dir / image_path.name)
            write_mask_for_image(image_path, json_path, masks_dir / f"{image_path.stem}.png", thickness)

    return DatasetSummary(train_count=len(train_pairs), val_count=len(val_pairs), output_dir=output_dir)
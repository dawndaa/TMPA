"""Qualitative visualization helpers for strict remote-sensing CTTA."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .corruptions import apply_corruption


BASE_PALETTE = [
    [0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255],
    [255, 255, 0], [255, 0, 255], [0, 255, 255], [128, 64, 128],
    [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153],
    [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35],
]


def should_save_vis(sample_idx: int, interval: int) -> bool:
    if interval <= 0:
        raise ValueError(f"vis_interval must be > 0, got {interval}")
    return int(sample_idx) % int(interval) == 0


def _palette(num_classes: int):
    return [BASE_PALETTE[i % len(BASE_PALETTE)] for i in range(int(num_classes))]


def _to_mask_numpy(mask) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask)
    while mask.ndim > 2 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D segmentation mask, got shape {mask.shape}")
    return mask.astype(np.int64, copy=False)


def colorize_mask(mask, num_classes: int, ignore_index: int = 255) -> np.ndarray:
    mask_np = _to_mask_numpy(mask)
    rgb = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
    for class_idx, color in enumerate(_palette(num_classes)):
        rgb[mask_np == class_idx] = np.asarray(color, dtype=np.uint8)
    rgb[mask_np == ignore_index] = np.array([255, 255, 255], dtype=np.uint8)
    return rgb


def _save_label(mask, path: Path) -> None:
    mask_np = _to_mask_numpy(mask)
    if mask_np.size and (mask_np.min() < 0 or mask_np.max() > 255):
        Image.fromarray(mask_np.astype(np.int32), mode="I").save(path)
    else:
        Image.fromarray(mask_np.astype(np.uint8), mode="L").save(path)


def _reliability_heatmap(reliability) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(reliability, torch.Tensor):
        reliability = reliability.detach().float().cpu().numpy()
    rel = np.asarray(reliability, dtype=np.float32)
    while rel.ndim > 2 and rel.shape[0] == 1:
        rel = rel[0]
    if rel.ndim != 2:
        raise ValueError(f"Expected a 2D reliability map, got shape {rel.shape}")
    rel = np.nan_to_num(rel, nan=0.0, posinf=1.0, neginf=0.0)
    rel = np.clip(rel, 0.0, 1.0)

    # Blue (low reliability) -> green/yellow -> red (high reliability).
    red = rel
    blue = 1.0 - rel
    green = 1.0 - np.abs(2.0 * rel - 1.0)
    rgb = np.stack([red, green, blue], axis=-1)
    rgb = np.uint8(np.clip(rgb, 0.0, 1.0) * 255.0)
    return rel, rgb


def save_remote_visualization(
    *,
    dataset,
    sample_idx: int,
    image_name: str,
    pred,
    gt,
    reliability,
    num_classes: int,
    vis_root: str,
    set_id: str,
    corruption: str,
    severity: int,
    ignore_index: int = 255,
) -> str:
    """Save original inputs, segmentation outputs, and the RSAP reliability map."""
    image_path = dataset.image_list[int(sample_idx)]
    sample_dir = (
        Path(vis_root)
        / str(set_id)
        / str(corruption)
        / f"{int(sample_idx):06d}_{str(image_name)}"
    )
    sample_dir.mkdir(parents=True, exist_ok=True)

    clean_rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    corrupted_rgb = apply_corruption(
        clean_rgb,
        corruption_name=corruption,
        severity=int(severity),
        sample_idx=int(sample_idx),
        daf_root=getattr(dataset, "daf_root", None),
    )

    Image.fromarray(clean_rgb, mode="RGB").save(sample_dir / "clean.png")
    Image.fromarray(corrupted_rgb, mode="RGB").save(sample_dir / "corrupted.png")

    _save_label(gt, sample_dir / "gt_label.png")
    _save_label(pred, sample_dir / "prediction_label.png")
    Image.fromarray(
        colorize_mask(gt, num_classes, ignore_index=ignore_index), mode="RGB"
    ).save(sample_dir / "gt_vis.png")
    Image.fromarray(
        colorize_mask(pred, num_classes, ignore_index=ignore_index), mode="RGB"
    ).save(sample_dir / "prediction_vis.png")

    if reliability is not None:
        rel, rel_rgb = _reliability_heatmap(reliability)
        np.save(sample_dir / "reliability.npy", rel)
        Image.fromarray(rel_rgb, mode="RGB").save(sample_dir / "reliability.png")

    metadata = {
        "dataset": str(set_id),
        "corruption": str(corruption),
        "corruption_severity": int(severity),
        "sample_idx": int(sample_idx),
        "image_name": str(image_name),
        "image_path": str(image_path),
        "has_reliability": reliability is not None,
    }
    with (sample_dir / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    return os.fspath(sample_dir)

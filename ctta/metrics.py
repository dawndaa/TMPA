"""Metrics for continual remote-sensing segmentation streams."""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np

from utils.utils import total_area_to_metrics


def _summary_from_result_tuples(results) -> Dict[str, float]:
    if not results:
        return {'aAcc': float('nan'), 'mIoU': float('nan'), 'mAcc': float('nan')}

    grouped = tuple(zip(*results))
    if len(grouped) != 4:
        raise ValueError('Expected intersect_and_union result tuples with four elements.')

    total_area_intersect = sum(grouped[0]).clone().detach().cpu().numpy()
    total_area_union = sum(grouped[1]).clone().detach().cpu().numpy()
    total_area_pred_label = sum(grouped[2]).clone().detach().cpu().numpy()
    total_area_label = sum(grouped[3]).clone().detach().cpu().numpy()

    raw = total_area_to_metrics(
        total_area_intersect,
        total_area_union,
        total_area_pred_label,
        total_area_label,
    )

    summary = {}
    for key, value in raw.items():
        mean_value = float(np.nanmean(value) * 100.0)
        summary[key if key == 'aAcc' else 'm' + key] = round(mean_value, 2)
    return summary


def summarize_results(results) -> Dict[str, float]:
    return _summary_from_result_tuples(results)


def summarize_progress(results, percentages: Iterable[float]) -> Dict[str, Dict[str, float]]:
    """Return cumulative metrics at requested fractions of a temporal stream."""
    percentages = sorted(set(float(p) for p in percentages))
    output: Dict[str, Dict[str, float]] = {}
    n = len(results)
    for pct in percentages:
        if not 0.0 < pct <= 1.0:
            raise ValueError(f'CTTA progress fractions must be in (0, 1], got {pct}')
        count = max(1, int(np.ceil(n * pct))) if n else 0
        output[f'{pct:.2f}'] = _summary_from_result_tuples(results[:count])
    return output

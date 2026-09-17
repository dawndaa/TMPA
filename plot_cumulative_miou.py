"""Plot cumulative mIoU curves for the five LoveDA CTTA ablations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


METHOD_SPECS = (
    ('M0', 'm0_domain', 'loveda_domain_*.json'),
    ('M1', 'm1_source_consistency', 'loveda_domain_*.json'),
    ('M2', 'm2_prompt_feature_consistency', 'loveda_domain_*.json'),
    ('M3', 'm3_cmac', 'loveda_domain_*.json'),
    ('M4', 'm4_safs', 'loveda_domain_*.json'),
)
METHOD_COLORS = {
    'M0': '#4C78A8',
    'M1': '#F58518',
    'M2': '#54A24B',
    'M3': '#E45756',
    'M4': '#B279A2',
}


def _find_report(evaluation_root: Path, method_directory: str, pattern: str) -> Path:
    candidates = sorted((evaluation_root / method_directory).glob(pattern))
    if len(candidates) != 1:
        raise ValueError(
            f'Expected one LoveDA report in {evaluation_root / method_directory}, '
            f'found {len(candidates)}.'
        )
    return candidates[0]


def _total_image_count(report: dict) -> int:
    return sum(int(domain['num_samples']) for domain in report['domains'])


def _load_curve(report_path: Path):
    with report_path.open('r', encoding='utf-8') as stream:
        report = json.load(stream)

    exact_curve = report.get('stream_curve')
    if exact_curve:
        image_counts = [int(point['num_images']) for point in exact_curve]
        miou_values = [float(point['mIoU']) for point in exact_curve]
        return image_counts, miou_values, True, _total_image_count(report)

    progress_points = report.get('stream_progress')
    if not progress_points:
        raise ValueError(f'No cumulative metrics found in {report_path}.')

    total_images = _total_image_count(report)
    image_counts = []
    miou_values = []
    for fraction_text, metrics in sorted(
        progress_points.items(), key=lambda item: float(item[0])
    ):
        fraction = float(fraction_text)
        image_counts.append(max(1, math.ceil(total_images * fraction)))
        miou_values.append(float(metrics['mIoU']))
    return image_counts, miou_values, False, total_images


def _render_plot(series_by_method, output_path: Path, total_images: int):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 6), dpi=180)

    for method_label, (image_counts, miou_values, _, _) in series_by_method.items():
        axis.plot(
            image_counts,
            miou_values,
            color=METHOD_COLORS[method_label],
            linewidth=2.0,
            marker='o' if len(image_counts) <= 20 else None,
            markersize=4.5,
            label=method_label,
        )

    all_values = [
        value
        for image_counts, miou_values, _, _ in series_by_method.values()
        for value in miou_values
    ]
    value_floor = max(0.0, math.floor(min(all_values) - 2.0))
    value_ceiling = min(100.0, math.ceil(max(all_values) + 2.0))
    axis.set_xlim(1, total_images)
    axis.set_ylim(value_floor, value_ceiling)
    axis.set_xlabel('Number of images processed cumulatively')
    axis.set_ylabel('Cumulative mIoU (%)')
    axis.set_title(
        f'LoveDA cumulative mIoU across {total_images} validation images',
        pad=30,
    )
    axis.grid(True, linestyle='--', linewidth=0.7, alpha=0.35)
    axis.legend(
        frameon=True,
        ncol=5,
        loc='lower center',
        bbox_to_anchor=(0.5, 1.14),
    )

    exact_point_counts = {
        len(method_series[0]) for method_series in series_by_method.values()
    }
    if exact_point_counts != {total_images}:
        stored_count = min(exact_point_counts)
        axis.text(
            0.01,
            0.02,
            f'Existing reports contain {stored_count} cumulative checkpoints; '
            'the plotted segments are not per-image measurements.',
            transform=axis.transAxes,
            fontsize=8.5,
            color='#555555',
        )

    figure.tight_layout(rect=(0, 0, 1, 0.88))
    figure.savefig(output_path, bbox_inches='tight')
    figure.savefig(output_path.with_suffix('.svg'), bbox_inches='tight')
    plt.close(figure)


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        '--evaluation-dir',
        type=Path,
        default=Path('save_result/loveda_urban_gpu1_clean_fixed'),
        help='Directory containing the five method subdirectories.',
    )
    argument_parser.add_argument(
        '--output',
        type=Path,
        default=Path('figures/loveda_cumulative_miou_m0_m4.png'),
        help='PNG output path; an SVG with the same stem is also written.',
    )
    arguments = argument_parser.parse_args()

    series_by_method = {}
    expected_counts = None
    expected_total = None
    for method_label, method_directory, filename_pattern in METHOD_SPECS:
        report_path = _find_report(
            arguments.evaluation_dir,
            method_directory,
            filename_pattern,
        )
        curve = _load_curve(report_path)
        image_counts, _, _, total_images = curve
        if expected_counts is None:
            expected_counts = image_counts
            expected_total = total_images
        elif image_counts != expected_counts or total_images != expected_total:
            raise ValueError('The five method reports do not share one stream order.')
        series_by_method[method_label] = curve

    _render_plot(series_by_method, arguments.output, expected_total)
    print(f'Wrote {arguments.output}')
    print(f'Wrote {arguments.output.with_suffix(".svg")}')


if __name__ == '__main__':
    main()

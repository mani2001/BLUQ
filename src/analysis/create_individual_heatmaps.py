#!/usr/bin/env python3
"""
Create individual heatmap figures for each metric from the dashboard.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

TASK_NAMES = {
    'qa': 'Question\nAnswering',
    'rc': 'Reading\nComprehension',
    'ci': 'Commonsense\nInference',
    'drs': 'Dialogue\nResponse',
    'ds': 'Document\nSummarization',
}

MODEL_ORDER = [
    'gemma-2-9b-it',
    'mistral-7b',
    'mistral-7b-instruct',
    'phi-2',
    'stablelm-2-1.6b',
    'tinyllama-1.1b',
]


def load_results(path):
    with open(path, 'r') as f:
        return json.load(f)


def create_individual_heatmap(results, metric, title, cmap, output_path,
                               dtype_filter='float32', conformal_method=None,
                               vmin=None, vmax=None, fmt='.1f',
                               cbar_label=None, invert_cmap=False):
    """Create a single heatmap figure for one metric."""

    # Filter results
    filtered = [r for r in results if r['dtype'] == dtype_filter]
    if conformal_method:
        filtered = [r for r in filtered if r['conformal_method'] == conformal_method]

    # Get unique models and tasks
    models = [m for m in MODEL_ORDER if any(r['model'] == m for r in filtered)]
    tasks = ['qa', 'rc', 'ci', 'drs', 'ds']

    # Build matrix
    matrix = np.full((len(models), len(tasks)), np.nan)
    for r in filtered:
        if r['model'] in models:
            i = models.index(r['model'])
            j = tasks.index(r['task'])
            value = r[metric]
            if metric in ['accuracy', 'coverage_rate']:
                value *= 100  # Convert to percentage
            matrix[i, j] = value

    # Add average column
    avg_col = np.nanmean(matrix, axis=1, keepdims=True)
    matrix_with_avg = np.hstack([matrix, avg_col])

    # Task labels with Avg
    task_labels = [TASK_NAMES.get(t, t) for t in tasks] + ['Avg']

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Use appropriate colormap
    if invert_cmap:
        cmap_obj = plt.cm.get_cmap(cmap + '_r')
    else:
        cmap_obj = plt.cm.get_cmap(cmap)

    # Plot heatmap
    if vmin is not None and vmax is not None:
        im = ax.imshow(matrix_with_avg, cmap=cmap_obj, aspect='auto', vmin=vmin, vmax=vmax)
    else:
        im = ax.imshow(matrix_with_avg, cmap=cmap_obj, aspect='auto')

    # Add text annotations
    for i in range(len(models)):
        for j in range(len(task_labels)):
            val = matrix_with_avg[i, j]
            if not np.isnan(val):
                # Determine text color based on background
                normalized = (val - (vmin or np.nanmin(matrix_with_avg))) / \
                            ((vmax or np.nanmax(matrix_with_avg)) - (vmin or np.nanmin(matrix_with_avg)))
                text_color = 'white' if (normalized < 0.3 or normalized > 0.7) else 'black'
                ax.text(j, i, f'{val:{fmt}}', ha='center', va='center',
                       fontsize=14, fontweight='bold', color=text_color)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(task_labels)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(task_labels, fontsize=11)
    ax.set_yticklabels(models, fontsize=11)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    if cbar_label:
        cbar.set_label(cbar_label, fontsize=12)

    # Set title
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)

    # Add grid lines
    ax.set_xticks(np.arange(len(task_labels) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(models) + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linewidth=2)
    ax.tick_params(which='minor', size=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    results_path = Path('./outputs/results/all_results_merged.json')
    output_dir = Path('./outputs/results/figures')
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(results_path)

    # Filter for base strategy only (or take first available)
    base_results = [r for r in results if r.get('strategy', 'base') == 'base']
    if not base_results:
        base_results = results

    dtype = 'float32'

    # 1. Accuracy Heatmap
    create_individual_heatmap(
        base_results,
        metric='accuracy',
        title=f'Accuracy (%) - {dtype}',
        cmap='RdYlGn',
        output_path=output_dir / f'heatmap_accuracy_{dtype}.png',
        dtype_filter=dtype,
        conformal_method='lac',  # Accuracy is same for both methods
        vmin=20, vmax=85,
        fmt='.1f',
        cbar_label='Accuracy (%)'
    )

    # 2. Coverage Rate Heatmap
    create_individual_heatmap(
        base_results,
        metric='coverage_rate',
        title=f'Coverage Rate (%) - {dtype}',
        cmap='RdYlGn',
        output_path=output_dir / f'heatmap_coverage_rate_{dtype}.png',
        dtype_filter=dtype,
        conformal_method='lac',  # Use LAC for coverage
        vmin=88, vmax=96,
        fmt='.1f',
        cbar_label='Coverage Rate (%)'
    )

    # 3. Set Size (LAC) Heatmap - smaller is better, so invert colormap
    create_individual_heatmap(
        base_results,
        metric='avg_set_size',
        title=f'Average Set Size (LAC) - {dtype}',
        cmap='RdYlGn',
        output_path=output_dir / f'heatmap_set_size_lac_{dtype}.png',
        dtype_filter=dtype,
        conformal_method='lac',
        vmin=1.0, vmax=6.0,
        fmt='.1f',
        cbar_label='Set Size (smaller = higher certainty)',
        invert_cmap=True  # Green for small (good), red for large (bad)
    )

    # 4. Set Size (APS) Heatmap - smaller is better, so invert colormap
    create_individual_heatmap(
        base_results,
        metric='avg_set_size',
        title=f'Average Set Size (APS) - {dtype}',
        cmap='RdYlGn',
        output_path=output_dir / f'heatmap_set_size_aps_{dtype}.png',
        dtype_filter=dtype,
        conformal_method='aps',
        vmin=3.0, vmax=6.5,
        fmt='.1f',
        cbar_label='Set Size (smaller = higher certainty)',
        invert_cmap=True  # Green for small (good), red for large (bad)
    )

    print(f"\nAll 4 individual heatmaps generated for {dtype}!")


if __name__ == "__main__":
    main()

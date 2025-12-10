#!/usr/bin/env python3
"""
Claims Visualization Script

Generates figures for verifying the three paper claims.
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model size mapping (in billions of parameters)
MODEL_SIZES = {
    'tinyllama-1.1b': 1.1,
    'stablelm-2-1.6b': 1.6,
    'phi-2': 2.7,
    'gemma-2b': 2.0,
    'gemma-2b-it': 2.0,
    'gemma-2-2b-it': 2.0,
    'mistral-7b': 7.0,
    'mistral-7b-instruct': 7.0,
    'gemma-2-9b-it': 9.0,
}

MODEL_DISPLAY_NAMES = {
    'tinyllama-1.1b': 'TinyLlama\n(1.1B)',
    'stablelm-2-1.6b': 'StableLM-2\n(1.6B)',
    'phi-2': 'Phi-2\n(2.7B)',
    'gemma-2b-it': 'Gemma-2B-IT\n(2B)',
    'gemma-2-2b-it': 'Gemma-2-2B-IT\n(2B)',
    'mistral-7b': 'Mistral-7B\n(7B)',
    'mistral-7b-instruct': 'Mistral-7B-Inst\n(7B)',
    'gemma-2-9b-it': 'Gemma-2-9B-IT\n(9B)',
}


def load_results(results_path: str) -> List[Dict]:
    """Load results from JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)


def filter_results(
    results: List[Dict],
    conformal_method: Optional[str] = None,
    dtype: Optional[str] = None
) -> List[Dict]:
    """Filter results by criteria."""
    filtered = results
    if conformal_method:
        filtered = [r for r in filtered if r.get('conformal_method') == conformal_method]
    if dtype:
        filtered = [r for r in filtered if r.get('dtype') == dtype]
    return filtered


def plot_claim1_accuracy_vs_setsize(
    results: List[Dict],
    output_dir: Path,
    conformal_method: str = 'lac',
    dtype: str = 'float16'
):
    """
    Plot Claim 1: Accuracy vs Set Size scatter plot.

    Shows negative correlation between accuracy and prediction set size.
    """
    import matplotlib.pyplot as plt
    from scipy import stats

    filtered = filter_results(results, conformal_method, dtype)

    if len(filtered) < 3:
        logger.warning("Not enough data for Claim 1 visualization")
        return

    # Extract data
    accuracies = [r['accuracy'] for r in filtered]
    set_sizes = [r['avg_set_size'] for r in filtered]
    models = [r['model'] for r in filtered]
    tasks = [r['task'] for r in filtered]

    # Calculate correlation
    correlation, p_value = stats.pearsonr(accuracies, set_sizes)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Color by model
    unique_models = list(set(models))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_models)))
    model_colors = {m: colors[i] for i, m in enumerate(unique_models)}

    # Plot each point
    for acc, ss, model, task in zip(accuracies, set_sizes, models, tasks):
        ax.scatter(acc, ss, c=[model_colors[model]], s=100, alpha=0.7)

    # Add trend line
    z = np.polyfit(accuracies, set_sizes, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(accuracies), max(accuracies), 100)
    ax.plot(x_line, p(x_line), 'r--', alpha=0.8, linewidth=2,
            label=f'Trend line (r={correlation:.3f}, p={p_value:.4f})')

    # Legend for models
    for model in unique_models:
        display_name = MODEL_DISPLAY_NAMES.get(model, model)
        ax.scatter([], [], c=[model_colors[model]], label=display_name.replace('\n', ' '), s=100)

    ax.set_xlabel('Accuracy', fontsize=12)
    ax.set_ylabel('Average Prediction Set Size', fontsize=12)
    ax.set_title(f'Claim 1: Accuracy vs Uncertainty ({conformal_method.upper()}, {dtype})\n'
                 f'Higher accuracy correlates with smaller set sizes (higher certainty)',
                 fontsize=14)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Add annotation
    status = "SUPPORTED" if correlation < 0 and p_value < 0.05 else "NOT SUPPORTED"
    ax.annotate(f'Claim Status: {status}',
                xy=(0.02, 0.98), xycoords='axes fraction',
                fontsize=11, fontweight='bold',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen' if 'SUPPORTED' in status else 'lightyellow'))

    plt.tight_layout()
    output_path = output_dir / f'claim1_accuracy_vs_setsize_{conformal_method}_{dtype}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved Claim 1 figure to {output_path}")


def plot_claim2_modelsize_vs_uncertainty(
    results: List[Dict],
    output_dir: Path,
    conformal_method: str = 'lac',
    dtype: str = 'float16'
):
    """
    Plot Claim 2: Model Size vs Average Set Size.

    Tests if larger models show more uncertainty.
    """
    import matplotlib.pyplot as plt
    from scipy import stats

    filtered = filter_results(results, conformal_method, dtype)

    # Aggregate by model
    model_data = {}
    for r in filtered:
        model = r['model']
        if model not in MODEL_SIZES:
            continue
        if model not in model_data:
            model_data[model] = {'set_sizes': [], 'accuracies': []}
        model_data[model]['set_sizes'].append(r['avg_set_size'])
        model_data[model]['accuracies'].append(r['accuracy'])

    if len(model_data) < 3:
        logger.warning("Not enough models for Claim 2 visualization")
        return

    # Compute means
    models = list(model_data.keys())
    sizes = [MODEL_SIZES[m] for m in models]
    mean_set_sizes = [np.mean(model_data[m]['set_sizes']) for m in models]
    std_set_sizes = [np.std(model_data[m]['set_sizes']) for m in models]
    mean_accuracies = [np.mean(model_data[m]['accuracies']) for m in models]

    # Correlation
    correlation, p_value = stats.pearsonr(sizes, mean_set_sizes)

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left plot: Size vs Set Size
    ax1 = axes[0]
    colors = plt.cm.viridis(np.array(mean_accuracies) / max(mean_accuracies))

    for i, (size, ss, std, model) in enumerate(zip(sizes, mean_set_sizes, std_set_sizes, models)):
        ax1.errorbar(size, ss, yerr=std, fmt='o', markersize=12, capsize=5,
                     color=colors[i], alpha=0.8)
        display_name = MODEL_DISPLAY_NAMES.get(model, model).replace('\n', ' ')
        ax1.annotate(display_name, (size, ss), textcoords="offset points",
                     xytext=(5, 5), fontsize=8)

    # Trend line
    z = np.polyfit(sizes, mean_set_sizes, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(sizes), max(sizes), 100)
    ax1.plot(x_line, p(x_line), 'r--', alpha=0.8, linewidth=2,
             label=f'Trend (r={correlation:.3f}, p={p_value:.4f})')

    ax1.set_xlabel('Model Size (Billions of Parameters)', fontsize=12)
    ax1.set_ylabel('Mean Prediction Set Size', fontsize=12)
    ax1.set_title(f'Claim 2: Model Size vs Uncertainty\n({conformal_method.upper()}, {dtype})', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add status
    status = "SUPPORTED" if correlation > 0 and p_value < 0.05 else "NOT SUPPORTED"
    ax1.annotate(f'Claim Status: {status}',
                 xy=(0.02, 0.98), xycoords='axes fraction',
                 fontsize=11, fontweight='bold',
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightgreen' if 'SUPPORTED' in status else 'lightyellow'))

    # Right plot: Bar chart sorted by size
    ax2 = axes[1]
    sorted_idx = np.argsort(sizes)
    sorted_models = [models[i] for i in sorted_idx]
    sorted_ss = [mean_set_sizes[i] for i in sorted_idx]
    sorted_std = [std_set_sizes[i] for i in sorted_idx]
    sorted_sizes = [sizes[i] for i in sorted_idx]

    x_pos = range(len(sorted_models))
    bars = ax2.bar(x_pos, sorted_ss, yerr=sorted_std, capsize=5, alpha=0.7,
                   color=plt.cm.coolwarm(np.linspace(0, 1, len(sorted_models))))

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([MODEL_DISPLAY_NAMES.get(m, m) for m in sorted_models],
                        fontsize=9, rotation=45, ha='right')
    ax2.set_ylabel('Mean Prediction Set Size', fontsize=12)
    ax2.set_title('Models Sorted by Size\n(Smaller -> Larger)', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = output_dir / f'claim2_modelsize_vs_uncertainty_{conformal_method}_{dtype}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved Claim 2 figure to {output_path}")


def plot_claim3_base_vs_instruct(
    results: List[Dict],
    output_dir: Path,
    conformal_method: str = 'lac',
    dtype: str = 'float16'
):
    """
    Plot Claim 3: Base vs Instruction-tuned model comparison.
    """
    import matplotlib.pyplot as plt
    from scipy import stats

    MODEL_PAIRS = {'mistral-7b': 'mistral-7b-instruct'}

    filtered = filter_results(results, conformal_method, dtype)
    available_models = set(r['model'] for r in filtered)

    # Find available pairs
    available_pairs = {b: i for b, i in MODEL_PAIRS.items()
                       if b in available_models and i in available_models}

    if not available_pairs:
        logger.warning("No base/instruct pairs found for Claim 3 visualization")
        return

    # Collect comparison data
    tasks = ['qa', 'rc', 'ci', 'drs', 'ds']
    task_names = {
        'qa': 'QA\n(MMLU)',
        'rc': 'RC\n(CosmosQA)',
        'ci': 'CI\n(HellaSwag)',
        'drs': 'DRS\n(HaluDial)',
        'ds': 'DS\n(HaluSum)'
    }

    fig, axes = plt.subplots(1, len(available_pairs), figsize=(12, 6))
    if len(available_pairs) == 1:
        axes = [axes]

    for ax_idx, (base_model, instruct_model) in enumerate(available_pairs.items()):
        ax = axes[ax_idx]

        base_results = {r['task']: r for r in filtered if r['model'] == base_model}
        instruct_results = {r['task']: r for r in filtered if r['model'] == instruct_model}

        common_tasks = [t for t in tasks if t in base_results and t in instruct_results]

        if not common_tasks:
            continue

        base_ss = [base_results[t]['avg_set_size'] for t in common_tasks]
        instruct_ss = [instruct_results[t]['avg_set_size'] for t in common_tasks]

        x = np.arange(len(common_tasks))
        width = 0.35

        bars1 = ax.bar(x - width/2, base_ss, width, label=f'{base_model} (Base)',
                       color='steelblue', alpha=0.8)
        bars2 = ax.bar(x + width/2, instruct_ss, width, label=f'{instruct_model} (Instruct)',
                       color='coral', alpha=0.8)

        ax.set_ylabel('Average Set Size', fontsize=12)
        ax.set_title(f'Base vs Instruct: {base_model.split("-")[0].title()}\n({conformal_method.upper()}, {dtype})',
                     fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([task_names.get(t, t) for t in common_tasks], fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        # Statistical test
        t_stat, p_value = stats.ttest_rel(instruct_ss, base_ss)
        mean_diff = np.mean(np.array(instruct_ss) - np.array(base_ss))

        status = "SUPPORTED" if mean_diff > 0 and p_value < 0.05 else "NOT SUPPORTED"
        ax.annotate(f'Mean diff: {mean_diff:.3f}\np={p_value:.4f}\n{status}',
                    xy=(0.98, 0.98), xycoords='axes fraction',
                    fontsize=9, fontweight='bold',
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round',
                              facecolor='lightgreen' if 'SUPPORTED' in status else 'lightyellow'))

    plt.tight_layout()
    output_path = output_dir / f'claim3_base_vs_instruct_{conformal_method}_{dtype}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved Claim 3 figure to {output_path}")


def plot_summary_dashboard(
    results: List[Dict],
    output_dir: Path,
    conformal_method: str = 'lac',
    dtype: str = 'float16'
):
    """Create a summary dashboard of all claims."""
    import matplotlib.pyplot as plt
    from scipy import stats

    filtered = filter_results(results, conformal_method, dtype)

    fig = plt.figure(figsize=(16, 12))

    # Claim 1: Accuracy vs Set Size (top left)
    ax1 = fig.add_subplot(2, 2, 1)
    accuracies = [r['accuracy'] for r in filtered]
    set_sizes = [r['avg_set_size'] for r in filtered]
    models = [r['model'] for r in filtered]

    unique_models = list(set(models))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_models)))
    model_colors = {m: colors[i] for i, m in enumerate(unique_models)}

    for acc, ss, model in zip(accuracies, set_sizes, models):
        ax1.scatter(acc, ss, c=[model_colors[model]], s=60, alpha=0.7)

    correlation, p_value = stats.pearsonr(accuracies, set_sizes)
    z = np.polyfit(accuracies, set_sizes, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(accuracies), max(accuracies), 100)
    ax1.plot(x_line, p(x_line), 'r--', linewidth=2)

    ax1.set_xlabel('Accuracy')
    ax1.set_ylabel('Avg Set Size')
    ax1.set_title(f'Claim 1: Accuracy vs Certainty\nr={correlation:.3f}, p={p_value:.4f}')
    ax1.grid(True, alpha=0.3)

    # Claim 2: Model Size vs Uncertainty (top right)
    ax2 = fig.add_subplot(2, 2, 2)
    model_data = {}
    for r in filtered:
        model = r['model']
        if model not in MODEL_SIZES:
            continue
        if model not in model_data:
            model_data[model] = []
        model_data[model].append(r['avg_set_size'])

    if model_data:
        models_list = list(model_data.keys())
        sizes = [MODEL_SIZES[m] for m in models_list]
        mean_ss = [np.mean(model_data[m]) for m in models_list]

        ax2.scatter(sizes, mean_ss, s=100, c='steelblue', alpha=0.8)
        for size, ss, model in zip(sizes, mean_ss, models_list):
            ax2.annotate(model.split('-')[0], (size, ss), fontsize=8, xytext=(3, 3),
                         textcoords='offset points')

        if len(sizes) >= 3:
            corr2, p2 = stats.pearsonr(sizes, mean_ss)
            ax2.set_title(f'Claim 2: Model Size vs Uncertainty\nr={corr2:.3f}, p={p2:.4f}')
        else:
            ax2.set_title('Claim 2: Model Size vs Uncertainty')

    ax2.set_xlabel('Model Size (B params)')
    ax2.set_ylabel('Mean Set Size')
    ax2.grid(True, alpha=0.3)

    # Claim 3: Base vs Instruct (bottom left)
    ax3 = fig.add_subplot(2, 2, 3)
    MODEL_PAIRS = {'mistral-7b': 'mistral-7b-instruct'}

    pair_data = []
    for base, instruct in MODEL_PAIRS.items():
        base_r = [r for r in filtered if r['model'] == base]
        inst_r = [r for r in filtered if r['model'] == instruct]
        if base_r and inst_r:
            base_ss = np.mean([r['avg_set_size'] for r in base_r])
            inst_ss = np.mean([r['avg_set_size'] for r in inst_r])
            pair_data.append((base, instruct, base_ss, inst_ss))

    if pair_data:
        x = np.arange(len(pair_data))
        width = 0.35
        base_vals = [p[2] for p in pair_data]
        inst_vals = [p[3] for p in pair_data]

        ax3.bar(x - width/2, base_vals, width, label='Base', color='steelblue')
        ax3.bar(x + width/2, inst_vals, width, label='Instruct', color='coral')
        ax3.set_xticks(x)
        ax3.set_xticklabels([p[0].split('-')[0].title() for p in pair_data])
        ax3.legend()
        ax3.set_ylabel('Mean Set Size')
        ax3.set_title('Claim 3: Base vs Instruction-tuned')
    ax3.grid(True, alpha=0.3, axis='y')

    # Coverage and Set Size by Model (bottom right)
    ax4 = fig.add_subplot(2, 2, 4)
    model_coverage = {}
    for r in filtered:
        model = r['model']
        if model not in model_coverage:
            model_coverage[model] = {'coverage': [], 'set_size': []}
        model_coverage[model]['coverage'].append(r['coverage_rate'])
        model_coverage[model]['set_size'].append(r['avg_set_size'])

    models_sorted = sorted(model_coverage.keys(),
                           key=lambda m: MODEL_SIZES.get(m, 0))
    x = np.arange(len(models_sorted))

    mean_cov = [np.mean(model_coverage[m]['coverage']) for m in models_sorted]
    mean_ss = [np.mean(model_coverage[m]['set_size']) for m in models_sorted]

    ax4_twin = ax4.twinx()
    bars = ax4.bar(x, mean_cov, alpha=0.7, color='steelblue', label='Coverage')
    line = ax4_twin.plot(x, mean_ss, 'ro-', linewidth=2, markersize=8, label='Set Size')

    ax4.axhline(y=0.9, color='green', linestyle='--', alpha=0.7, label='90% target')
    ax4.set_xticks(x)
    ax4.set_xticklabels([m.split('-')[0] for m in models_sorted], rotation=45, ha='right', fontsize=8)
    ax4.set_ylabel('Coverage Rate', color='steelblue')
    ax4_twin.set_ylabel('Avg Set Size', color='red')
    ax4.set_title('Coverage & Set Size by Model')
    ax4.legend(loc='upper left', fontsize=8)
    ax4.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'Claims Verification Summary ({conformal_method.upper()}, {dtype})',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = output_dir / f'claims_summary_dashboard_{conformal_method}_{dtype}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved summary dashboard to {output_path}")


def generate_all_visualizations(results_path: str, output_dir: str):
    """Generate all claim visualizations."""
    results = load_results(results_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for method in ['lac', 'aps']:
        for dtype in ['float16', 'float32']:
            logger.info(f"Generating visualizations for {method.upper()}, {dtype}")
            try:
                plot_claim1_accuracy_vs_setsize(results, output_dir, method, dtype)
                plot_claim2_modelsize_vs_uncertainty(results, output_dir, method, dtype)
                plot_claim3_base_vs_instruct(results, output_dir, method, dtype)
                plot_summary_dashboard(results, output_dir, method, dtype)
            except Exception as e:
                logger.warning(f"Failed for {method}/{dtype}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate claim verification visualizations")
    parser.add_argument(
        '--results-path',
        type=str,
        default='./outputs/results/all_results_merged.json',
        help='Path to results JSON file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./outputs/results/figures/claims',
        help='Directory to save figures'
    )

    args = parser.parse_args()
    generate_all_visualizations(args.results_path, args.output_dir)


if __name__ == "__main__":
    main()

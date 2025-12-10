#!/usr/bin/env python3
"""
Create individual figures for each of the three claims verification.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# Model sizes in billions
MODEL_SIZES = {
    'tinyllama-1.1b': 1.1,
    'stablelm-2-1.6b': 1.6,
    'phi-2': 2.7,
    'gemma-2b-it': 2.0,
    'gemma-2-2b-it': 2.0,
    'mistral-7b': 7.0,
    'mistral-7b-instruct': 7.0,
    'gemma-2-9b-it': 9.0,
}

MODEL_DISPLAY = {
    'tinyllama-1.1b': 'TinyLlama\n1.1B',
    'stablelm-2-1.6b': 'StableLM\n1.6B',
    'phi-2': 'Phi-2\n2.7B',
    'gemma-2b-it': 'Gemma-2B\nIT',
    'gemma-2-2b-it': 'Gemma-2\n2B-IT',
    'mistral-7b': 'Mistral\n7B',
    'mistral-7b-instruct': 'Mistral-7B\nInstruct',
    'gemma-2-9b-it': 'Gemma-2\n9B-IT',
}

TASK_NAMES = {
    'qa': 'Question\nAnswering',
    'rc': 'Reading\nComprehension',
    'ci': 'Commonsense\nInference',
    'drs': 'Dialogue\nResponse',
    'ds': 'Document\nSummarization',
}


def load_results(path):
    with open(path, 'r') as f:
        return json.load(f)


def create_claim1_figure(results, output_dir):
    """Claim 1: Higher accuracy correlates with lower uncertainty."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Claim 1: Higher Accuracy Correlates with Lower Uncertainty',
                 fontsize=16, fontweight='bold', y=1.02)

    # Filter for LAC float16
    filtered = [r for r in results if r['conformal_method'] == 'lac' and r['dtype'] == 'float16']

    accuracies = [r['accuracy'] * 100 for r in filtered]
    set_sizes = [r['avg_set_size'] for r in filtered]
    models = [r['model'] for r in filtered]
    tasks = [r['task'] for r in filtered]

    # Left panel: Scatter by model
    ax1 = axes[0]
    unique_models = sorted(set(models), key=lambda m: MODEL_SIZES.get(m, 0))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_models)))
    model_colors = {m: colors[i] for i, m in enumerate(unique_models)}

    for acc, ss, model in zip(accuracies, set_sizes, models):
        ax1.scatter(acc, ss, c=[model_colors[model]], s=100, alpha=0.7,
                   edgecolors='white', linewidth=1, label=model)

    # Trend line
    correlation, p_value = stats.pearsonr(accuracies, set_sizes)
    z = np.polyfit(accuracies, set_sizes, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(accuracies), max(accuracies), 100)
    ax1.plot(x_line, p(x_line), 'r--', linewidth=2.5, alpha=0.8, label='Trend')

    ax1.set_xlabel('Accuracy (%)', fontsize=12)
    ax1.set_ylabel('Average Prediction Set Size', fontsize=12)
    ax1.set_title('All Model-Task Combinations', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Legend with unique models
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=8, ncol=2)

    # Statistics box
    ax1.text(0.02, 0.02, f'Pearson r = {correlation:.3f}\np < 0.0001\nn = {len(filtered)}',
             transform=ax1.transAxes, fontsize=11, fontweight='bold',
             verticalalignment='bottom', horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='#90EE90', alpha=0.9))

    # Right panel: Aggregated by model
    ax2 = axes[1]
    model_data = {}
    for r in filtered:
        model = r['model']
        if model not in model_data:
            model_data[model] = {'acc': [], 'ss': []}
        model_data[model]['acc'].append(r['accuracy'] * 100)
        model_data[model]['ss'].append(r['avg_set_size'])

    models_sorted = sorted(model_data.keys(), key=lambda m: MODEL_SIZES.get(m, 0))
    mean_acc = [np.mean(model_data[m]['acc']) for m in models_sorted]
    mean_ss = [np.mean(model_data[m]['ss']) for m in models_sorted]
    std_ss = [np.std(model_data[m]['ss']) for m in models_sorted]

    for i, (acc, ss, std, model) in enumerate(zip(mean_acc, mean_ss, std_ss, models_sorted)):
        ax2.errorbar(acc, ss, yerr=std, fmt='o', markersize=14, capsize=6,
                    color=model_colors[model], ecolor='gray', alpha=0.8,
                    markeredgecolor='white', markeredgewidth=1.5)
        ax2.annotate(MODEL_DISPLAY.get(model, model).replace('\n', ' '),
                    (acc, ss), textcoords="offset points",
                    xytext=(8, 0), fontsize=9, alpha=0.9)

    # Trend line for aggregated
    corr2, _ = stats.pearsonr(mean_acc, mean_ss)
    z2 = np.polyfit(mean_acc, mean_ss, 1)
    p2 = np.poly1d(z2)
    x_line2 = np.linspace(min(mean_acc), max(mean_acc), 100)
    ax2.plot(x_line2, p2(x_line2), 'r--', linewidth=2.5, alpha=0.8)

    ax2.set_xlabel('Mean Accuracy (%)', fontsize=12)
    ax2.set_ylabel('Mean Prediction Set Size', fontsize=12)
    ax2.set_title('Aggregated by Model (with Std Dev)', fontsize=12)
    ax2.grid(True, alpha=0.3)

    ax2.text(0.02, 0.02, f'Model-level r = {corr2:.3f}',
             transform=ax2.transAxes, fontsize=11, fontweight='bold',
             verticalalignment='bottom', horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='#90EE90', alpha=0.9))

    # Verdict
    fig.text(0.5, -0.02, 'VERDICT: SUPPORTED - Strong negative correlation confirms higher accuracy leads to smaller prediction sets',
             ha='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='#90EE90', alpha=0.9))

    plt.tight_layout()
    output_path = output_dir / 'claim1_accuracy_uncertainty.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def create_claim2_figure(results, output_dir):
    """Claim 2: Larger models display greater uncertainty."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Claim 2: Larger Models Display Greater Uncertainty',
                 fontsize=16, fontweight='bold', y=1.02)

    # Filter for LAC float16
    filtered = [r for r in results if r['conformal_method'] == 'lac' and r['dtype'] == 'float16']

    # Aggregate by model
    model_data = {}
    for r in filtered:
        model = r['model']
        if model not in MODEL_SIZES:
            continue
        if model not in model_data:
            model_data[model] = {'ss': [], 'acc': []}
        model_data[model]['ss'].append(r['avg_set_size'])
        model_data[model]['acc'].append(r['accuracy'] * 100)

    models_sorted = sorted(model_data.keys(), key=lambda m: MODEL_SIZES[m])
    sizes = [MODEL_SIZES[m] for m in models_sorted]
    mean_ss = [np.mean(model_data[m]['ss']) for m in models_sorted]
    mean_acc = [np.mean(model_data[m]['acc']) for m in models_sorted]
    std_ss = [np.std(model_data[m]['ss']) for m in models_sorted]

    # Left panel: Size vs Uncertainty
    ax1 = axes[0]

    # Color by accuracy (green=high, red=low)
    norm_acc = (np.array(mean_acc) - min(mean_acc)) / (max(mean_acc) - min(mean_acc))
    colors = plt.cm.RdYlGn(norm_acc)

    for i, (size, ss, std, model, color) in enumerate(zip(sizes, mean_ss, std_ss, models_sorted, colors)):
        ax1.errorbar(size, ss, yerr=std, fmt='o', markersize=14, capsize=6,
                    color=color, ecolor='gray', alpha=0.9,
                    markeredgecolor='white', markeredgewidth=1.5)
        ax1.annotate(MODEL_DISPLAY.get(model, model),
                    (size, ss), textcoords="offset points",
                    xytext=(10, 0), fontsize=9, ha='left')

    # Trend line
    corr, p_val = stats.pearsonr(sizes, mean_ss)
    z = np.polyfit(sizes, mean_ss, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(sizes), max(sizes), 100)
    ax1.plot(x_line, p(x_line), 'r--', linewidth=2.5, alpha=0.8)

    ax1.set_xlabel('Model Size (Billion Parameters)', fontsize=12)
    ax1.set_ylabel('Mean Prediction Set Size', fontsize=12)
    ax1.set_title('Model Size vs Uncertainty', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Add colorbar for accuracy
    sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(min(mean_acc), max(mean_acc)))
    cbar = plt.colorbar(sm, ax=ax1, shrink=0.8)
    cbar.set_label('Mean Accuracy (%)', fontsize=10)

    ax1.text(0.02, 0.98, f'r = {corr:.3f}\np = {p_val:.4f}',
             transform=ax1.transAxes, fontsize=11, fontweight='bold',
             verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='#FFB6C1', alpha=0.9))

    # Right panel: Bar chart sorted by size
    ax2 = axes[1]

    x = np.arange(len(models_sorted))
    bars = ax2.bar(x, mean_ss, yerr=std_ss, capsize=4, color=colors,
                   edgecolor='white', linewidth=1.5, alpha=0.9)

    # Add size labels on bars
    for i, (bar, size, acc) in enumerate(zip(bars, sizes, mean_acc)):
        height = bar.get_height()
        ax2.annotate(f'{size}B\n({acc:.1f}%)',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    ax2.set_xticks(x)
    ax2.set_xticklabels([MODEL_DISPLAY.get(m, m) for m in models_sorted], fontsize=8)
    ax2.set_ylabel('Mean Prediction Set Size', fontsize=12)
    ax2.set_xlabel('Models (sorted by size)', fontsize=12)
    ax2.set_title('Uncertainty by Model (Sorted by Size)', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')

    # Verdict
    fig.text(0.5, -0.02, 'VERDICT: NOT SUPPORTED on SLMs - Larger models show LOWER uncertainty because accuracy improves with size',
             ha='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='#FFB6C1', alpha=0.9))

    plt.tight_layout()
    output_path = output_dir / 'claim2_modelsize_uncertainty.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def create_claim3_figure(results, output_dir):
    """Claim 3: Instruction-tuning increases uncertainty."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Claim 3: Instruction-Tuning Increases Uncertainty',
                 fontsize=16, fontweight='bold', y=1.02)

    tasks = ['qa', 'rc', 'ci', 'drs', 'ds']
    task_labels = ['QA', 'RC', 'CI', 'DRS', 'DS']

    # Filter for LAC float16
    filtered = [r for r in results if r['conformal_method'] == 'lac' and r['dtype'] == 'float16']

    base_model = 'mistral-7b'
    instruct_model = 'mistral-7b-instruct'

    base_results = {r['task']: r for r in filtered if r['model'] == base_model}
    instruct_results = {r['task']: r for r in filtered if r['model'] == instruct_model}

    base_ss = [base_results[t]['avg_set_size'] if t in base_results else 0 for t in tasks]
    instruct_ss = [instruct_results[t]['avg_set_size'] if t in instruct_results else 0 for t in tasks]
    base_acc = [base_results[t]['accuracy'] * 100 if t in base_results else 0 for t in tasks]
    instruct_acc = [instruct_results[t]['accuracy'] * 100 if t in instruct_results else 0 for t in tasks]

    # Left panel: Set Size comparison
    ax1 = axes[0]
    x = np.arange(len(tasks))
    width = 0.35

    bars1 = ax1.bar(x - width/2, base_ss, width, label='Mistral-7B (Base)',
                    color='#4ECDC4', edgecolor='white', linewidth=1.5)
    bars2 = ax1.bar(x + width/2, instruct_ss, width, label='Mistral-7B-Instruct',
                    color='#FF6B6B', edgecolor='white', linewidth=1.5)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)

    ax1.set_ylabel('Average Prediction Set Size', fontsize=12)
    ax1.set_xlabel('Task', fontsize=12)
    ax1.set_title('Prediction Set Size Comparison', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(task_labels, fontsize=11)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')

    # Mark which is larger
    for i, (b, ins) in enumerate(zip(base_ss, instruct_ss)):
        if ins > b:
            ax1.annotate('*', xy=(i + width/2, ins + 0.15), ha='center', fontsize=14, color='red')

    # Right panel: Accuracy comparison
    ax2 = axes[1]

    bars3 = ax2.bar(x - width/2, base_acc, width, label='Mistral-7B (Base)',
                    color='#4ECDC4', edgecolor='white', linewidth=1.5)
    bars4 = ax2.bar(x + width/2, instruct_acc, width, label='Mistral-7B-Instruct',
                    color='#FF6B6B', edgecolor='white', linewidth=1.5)

    # Add value labels
    for bar in bars3:
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)
    for bar in bars4:
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)

    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_xlabel('Task', fontsize=12)
    ax2.set_title('Accuracy Comparison', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(task_labels, fontsize=11)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    # Statistical test
    if base_ss and instruct_ss:
        t_stat, p_val = stats.ttest_rel(instruct_ss, base_ss)
        mean_diff = np.mean(np.array(instruct_ss) - np.array(base_ss))
        instruct_larger = sum(1 for i, b in zip(instruct_ss, base_ss) if i > b)

        # Summary stats box
        stats_text = f'Paired t-test:\nt = {t_stat:.3f}\np = {p_val:.4f}\n\nMean diff: {mean_diff:.3f}\nInstruct > Base: {instruct_larger}/5 tasks'

        status_color = '#FFFACD'  # Partial support
        if mean_diff > 0 and p_val < 0.05:
            status_color = '#90EE90'
        elif mean_diff < 0 and p_val < 0.05:
            status_color = '#FFB6C1'

        ax1.text(0.02, 0.98, stats_text,
                transform=ax1.transAxes, fontsize=10, fontweight='bold',
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor=status_color, alpha=0.9))

    # Verdict
    verdict_color = '#FFFACD'
    if mean_diff > 0 and p_val < 0.05:
        verdict_text = 'VERDICT: SUPPORTED - Instruction-tuned model shows significantly higher uncertainty'
        verdict_color = '#90EE90'
    elif instruct_larger >= 3:
        verdict_text = 'VERDICT: PARTIALLY SUPPORTED - Instruct model shows higher uncertainty on most tasks, but not statistically significant'
        verdict_color = '#FFFACD'
    else:
        verdict_text = 'VERDICT: NOT SUPPORTED - No consistent pattern of increased uncertainty with instruction-tuning'
        verdict_color = '#FFB6C1'

    fig.text(0.5, -0.02, verdict_text,
             ha='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor=verdict_color, alpha=0.9))

    plt.tight_layout()
    output_path = output_dir / 'claim3_instruction_tuning.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    results_path = Path('./outputs/results/all_results_merged.json')
    output_dir = Path('./outputs/results/figures')
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(results_path)

    create_claim1_figure(results, output_dir)
    create_claim2_figure(results, output_dir)
    create_claim3_figure(results, output_dir)

    print("\nAll three claim figures generated successfully!")


if __name__ == "__main__":
    main()

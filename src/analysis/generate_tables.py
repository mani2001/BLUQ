#!/usr/bin/env python3
"""
Generate Comparison Tables

Creates paper-style comparison tables from benchmark results.
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

TASK_DISPLAY = {
    'qa': 'QA',
    'rc': 'RC',
    'ci': 'CI',
    'drs': 'DRS',
    'ds': 'DS'
}


def load_results(results_path: str) -> List[Dict]:
    """Load results from JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)


def generate_accuracy_table(
    results: List[Dict],
    dtype: str = 'float16'
) -> str:
    """Generate accuracy comparison table (Markdown format)."""
    filtered = [r for r in results if r['dtype'] == dtype and r['conformal_method'] == 'lac']

    # Get unique models and tasks
    models = sorted(set(r['model'] for r in filtered), key=lambda m: MODEL_SIZES.get(m, 0))
    tasks = ['qa', 'rc', 'ci', 'drs', 'ds']

    # Build table
    lines = []
    lines.append(f"## Accuracy by Model and Task ({dtype})")
    lines.append("")
    header = "| Model | Size | " + " | ".join(TASK_DISPLAY.get(t, t) for t in tasks) + " | Mean |"
    lines.append(header)
    lines.append("|" + "---|" * (len(tasks) + 3))

    for model in models:
        size = MODEL_SIZES.get(model, '?')
        model_results = {r['task']: r for r in filtered if r['model'] == model}

        row = [f"**{model}**", f"{size}B"]
        accuracies = []
        for task in tasks:
            if task in model_results:
                acc = model_results[task]['accuracy']
                row.append(f"{acc:.1%}")
                accuracies.append(acc)
            else:
                row.append("-")

        mean_acc = np.mean(accuracies) if accuracies else 0
        row.append(f"**{mean_acc:.1%}**")
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def generate_coverage_table(
    results: List[Dict],
    conformal_method: str = 'lac',
    dtype: str = 'float16'
) -> str:
    """Generate coverage rate comparison table."""
    filtered = [r for r in results
                if r['dtype'] == dtype and r['conformal_method'] == conformal_method]

    models = sorted(set(r['model'] for r in filtered), key=lambda m: MODEL_SIZES.get(m, 0))
    tasks = ['qa', 'rc', 'ci', 'drs', 'ds']

    lines = []
    lines.append(f"## Coverage Rate by Model and Task ({conformal_method.upper()}, {dtype})")
    lines.append("")
    lines.append("Target: 90% coverage")
    lines.append("")
    header = "| Model | " + " | ".join(TASK_DISPLAY.get(t, t) for t in tasks) + " | Mean | Meets 90% |"
    lines.append(header)
    lines.append("|" + "---|" * (len(tasks) + 3))

    for model in models:
        model_results = {r['task']: r for r in filtered if r['model'] == model}

        row = [f"**{model}**"]
        coverages = []
        meets_guarantee = []

        for task in tasks:
            if task in model_results:
                cov = model_results[task]['coverage_rate']
                meets = model_results[task].get('meets_guarantee', cov >= 0.9)
                symbol = "" if meets else "*"
                row.append(f"{cov:.1%}{symbol}")
                coverages.append(cov)
                meets_guarantee.append(meets)
            else:
                row.append("-")

        mean_cov = np.mean(coverages) if coverages else 0
        meets_count = sum(meets_guarantee)
        row.append(f"**{mean_cov:.1%}**")
        row.append(f"{meets_count}/{len(meets_guarantee)}")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("*Coverage below 90% target")

    return "\n".join(lines)


def generate_setsize_table(
    results: List[Dict],
    conformal_method: str = 'lac',
    dtype: str = 'float16'
) -> str:
    """Generate average set size comparison table."""
    filtered = [r for r in results
                if r['dtype'] == dtype and r['conformal_method'] == conformal_method]

    models = sorted(set(r['model'] for r in filtered), key=lambda m: MODEL_SIZES.get(m, 0))
    tasks = ['qa', 'rc', 'ci', 'drs', 'ds']

    lines = []
    lines.append(f"## Average Prediction Set Size ({conformal_method.upper()}, {dtype})")
    lines.append("")
    lines.append("Smaller set size = higher certainty (better)")
    lines.append("")
    header = "| Model | " + " | ".join(TASK_DISPLAY.get(t, t) for t in tasks) + " | Mean |"
    lines.append(header)
    lines.append("|" + "---|" * (len(tasks) + 2))

    for model in models:
        model_results = {r['task']: r for r in filtered if r['model'] == model}

        row = [f"**{model}**"]
        set_sizes = []

        for task in tasks:
            if task in model_results:
                ss = model_results[task]['avg_set_size']
                row.append(f"{ss:.2f}")
                set_sizes.append(ss)
            else:
                row.append("-")

        mean_ss = np.mean(set_sizes) if set_sizes else 0
        row.append(f"**{mean_ss:.2f}**")
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def generate_lac_vs_aps_table(
    results: List[Dict],
    dtype: str = 'float16'
) -> str:
    """Generate LAC vs APS comparison table."""
    lac_results = [r for r in results if r['dtype'] == dtype and r['conformal_method'] == 'lac']
    aps_results = [r for r in results if r['dtype'] == dtype and r['conformal_method'] == 'aps']

    models = sorted(set(r['model'] for r in lac_results), key=lambda m: MODEL_SIZES.get(m, 0))

    lines = []
    lines.append(f"## LAC vs APS Comparison ({dtype})")
    lines.append("")
    lines.append("| Model | LAC Coverage | APS Coverage | LAC Set Size | APS Set Size | Winner |")
    lines.append("|---|---|---|---|---|---|")

    for model in models:
        lac_model = [r for r in lac_results if r['model'] == model]
        aps_model = [r for r in aps_results if r['model'] == model]

        if not lac_model or not aps_model:
            continue

        lac_cov = np.mean([r['coverage_rate'] for r in lac_model])
        aps_cov = np.mean([r['coverage_rate'] for r in aps_model])
        lac_ss = np.mean([r['avg_set_size'] for r in lac_model])
        aps_ss = np.mean([r['avg_set_size'] for r in aps_model])

        # Winner: better coverage with smaller set size
        # Prefer the one that meets 90% with smaller sets
        lac_meets = lac_cov >= 0.9
        aps_meets = aps_cov >= 0.9

        if lac_meets and aps_meets:
            winner = "LAC" if lac_ss < aps_ss else "APS"
        elif lac_meets:
            winner = "LAC"
        elif aps_meets:
            winner = "APS"
        else:
            winner = "LAC" if lac_cov > aps_cov else "APS"

        lines.append(f"| **{model}** | {lac_cov:.1%} | {aps_cov:.1%} | {lac_ss:.2f} | {aps_ss:.2f} | {winner} |")

    return "\n".join(lines)


def generate_base_vs_instruct_table(
    results: List[Dict],
    dtype: str = 'float16'
) -> str:
    """Generate base vs instruction-tuned comparison table."""
    MODEL_PAIRS = {'mistral-7b': 'mistral-7b-instruct'}

    filtered = [r for r in results if r['dtype'] == dtype]
    available_models = set(r['model'] for r in filtered)

    lines = []
    lines.append(f"## Base vs Instruction-Tuned Comparison ({dtype})")
    lines.append("")

    for base_model, instruct_model in MODEL_PAIRS.items():
        if base_model not in available_models or instruct_model not in available_models:
            continue

        lines.append(f"### {base_model.title()} Family")
        lines.append("")
        lines.append("| Task | Method | Base Acc | Inst Acc | Base SetSize | Inst SetSize | Inst More Uncertain |")
        lines.append("|---|---|---|---|---|---|---|")

        for method in ['lac', 'aps']:
            base_r = {r['task']: r for r in filtered
                      if r['model'] == base_model and r['conformal_method'] == method}
            inst_r = {r['task']: r for r in filtered
                      if r['model'] == instruct_model and r['conformal_method'] == method}

            for task in ['qa', 'rc', 'ci', 'drs', 'ds']:
                if task not in base_r or task not in inst_r:
                    continue

                base = base_r[task]
                inst = inst_r[task]

                more_uncertain = "Yes" if inst['avg_set_size'] > base['avg_set_size'] else "No"

                lines.append(
                    f"| {TASK_DISPLAY.get(task, task)} | {method.upper()} | "
                    f"{base['accuracy']:.1%} | {inst['accuracy']:.1%} | "
                    f"{base['avg_set_size']:.2f} | {inst['avg_set_size']:.2f} | {more_uncertain} |"
                )

        lines.append("")

    return "\n".join(lines)


def generate_summary_statistics(results: List[Dict]) -> str:
    """Generate overall summary statistics."""
    lines = []
    lines.append("## Summary Statistics")
    lines.append("")

    # Count by model
    models = set(r['model'] for r in results)
    lines.append(f"**Total models evaluated:** {len(models)}")
    lines.append("")
    for model in sorted(models, key=lambda m: MODEL_SIZES.get(m, 0)):
        model_r = [r for r in results if r['model'] == model]
        dtypes = set(r['dtype'] for r in model_r)
        tasks = set(r['task'] for r in model_r)
        lines.append(f"- {model}: {len(tasks)} tasks, dtypes: {sorted(dtypes)}")

    lines.append("")
    lines.append("**Coverage guarantee (90%) achievement:**")
    lines.append("")

    for method in ['lac', 'aps']:
        method_r = [r for r in results if r['conformal_method'] == method]
        meets = sum(1 for r in method_r if r.get('meets_guarantee', r['coverage_rate'] >= 0.9))
        total = len(method_r)
        lines.append(f"- {method.upper()}: {meets}/{total} ({meets/total:.1%})")

    return "\n".join(lines)


def generate_all_tables(results_path: str, output_path: str):
    """Generate all comparison tables and save to markdown file."""
    results = load_results(results_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sections = []

    # Header
    sections.append("# BLUQ Benchmark Results")
    sections.append("")
    sections.append("Benchmarking Small Language Models via Uncertainty Quantification")
    sections.append("")
    sections.append("---")
    sections.append("")

    # Summary
    sections.append(generate_summary_statistics(results))
    sections.append("")
    sections.append("---")
    sections.append("")

    # Tables for each dtype
    for dtype in ['float16', 'float32']:
        dtype_results = [r for r in results if r['dtype'] == dtype]
        if not dtype_results:
            continue

        sections.append(f"# Results ({dtype})")
        sections.append("")

        sections.append(generate_accuracy_table(results, dtype))
        sections.append("")
        sections.append("---")
        sections.append("")

        sections.append(generate_coverage_table(results, 'lac', dtype))
        sections.append("")
        sections.append(generate_coverage_table(results, 'aps', dtype))
        sections.append("")
        sections.append("---")
        sections.append("")

        sections.append(generate_setsize_table(results, 'lac', dtype))
        sections.append("")
        sections.append(generate_setsize_table(results, 'aps', dtype))
        sections.append("")
        sections.append("---")
        sections.append("")

        sections.append(generate_lac_vs_aps_table(results, dtype))
        sections.append("")
        sections.append("---")
        sections.append("")

        sections.append(generate_base_vs_instruct_table(results, dtype))
        sections.append("")

    # Write to file
    with open(output_path, 'w') as f:
        f.write("\n".join(sections))

    logger.info(f"Tables saved to {output_path}")

    # Also generate JSON summary
    json_path = output_path.with_suffix('.json')
    summary = {
        'models': sorted(set(r['model'] for r in results)),
        'tasks': sorted(set(r['task'] for r in results)),
        'dtypes': sorted(set(r['dtype'] for r in results)),
        'methods': sorted(set(r['conformal_method'] for r in results)),
        'total_experiments': len(results),
        'per_model': {}
    }

    for model in summary['models']:
        model_r = [r for r in results if r['model'] == model]
        summary['per_model'][model] = {
            'mean_accuracy': float(np.mean([r['accuracy'] for r in model_r])),
            'mean_coverage': float(np.mean([r['coverage_rate'] for r in model_r])),
            'mean_set_size': float(np.mean([r['avg_set_size'] for r in model_r])),
            'experiments': len(model_r)
        }

    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Summary JSON saved to {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate comparison tables from benchmark results")
    parser.add_argument(
        '--results-path',
        type=str,
        default='./outputs/results/all_results_merged.json',
        help='Path to results JSON file'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default='./outputs/results/comparison_tables.md',
        help='Path to save markdown tables'
    )

    args = parser.parse_args()
    generate_all_tables(args.results_path, args.output_path)


if __name__ == "__main__":
    main()

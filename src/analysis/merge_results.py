#!/usr/bin/env python3
"""
Merge Results Script

Consolidates all results from model-specific directories into a single all_results.json.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Set

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_result_files(results_dir: Path) -> List[Path]:
    """Find all results JSON files in model subdirectories."""
    result_files = []

    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue
        if model_dir.name in ['figures', 'logs', 'probabilities']:
            continue

        # Look for results_*.json files
        for f in model_dir.glob('results_*.json'):
            result_files.append(f)

    return result_files


def merge_results(results_dir: str, output_path: str) -> Dict:
    """
    Merge all results from model directories.

    Args:
        results_dir: Path to results directory
        output_path: Path to save merged results

    Returns:
        Dictionary with merge statistics
    """
    results_dir = Path(results_dir)
    output_path = Path(output_path)

    all_results = []
    seen_keys: Set[tuple] = set()  # For deduplication

    result_files = find_result_files(results_dir)
    logger.info(f"Found {len(result_files)} result files")

    for file_path in result_files:
        try:
            with open(file_path, 'r') as f:
                results = json.load(f)

            if not isinstance(results, list):
                results = [results]

            for r in results:
                # Create unique key for deduplication
                key = (
                    r.get('model'),
                    r.get('task'),
                    r.get('dtype'),
                    r.get('strategy'),
                    r.get('conformal_method')
                )

                if key in seen_keys:
                    logger.debug(f"Skipping duplicate: {key}")
                    continue

                seen_keys.add(key)
                all_results.append(r)

            logger.info(f"  Loaded {len(results)} entries from {file_path.name}")

        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")

    # Also load existing all_results.json if present
    existing_path = results_dir / 'all_results.json'
    if existing_path.exists():
        try:
            with open(existing_path, 'r') as f:
                existing = json.load(f)
            for r in existing:
                key = (
                    r.get('model'),
                    r.get('task'),
                    r.get('dtype'),
                    r.get('strategy'),
                    r.get('conformal_method')
                )
                if key not in seen_keys:
                    seen_keys.add(key)
                    all_results.append(r)
            logger.info(f"  Merged {len(existing)} entries from existing all_results.json")
        except Exception as e:
            logger.warning(f"Failed to load existing all_results.json: {e}")

    # Sort by model, task, dtype, method
    all_results.sort(key=lambda x: (
        x.get('model', ''),
        x.get('task', ''),
        x.get('dtype', ''),
        x.get('conformal_method', '')
    ))

    # Save merged results
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"Saved {len(all_results)} total results to {output_path}")

    # Generate statistics
    models = set(r['model'] for r in all_results)
    tasks = set(r['task'] for r in all_results)
    dtypes = set(r['dtype'] for r in all_results)
    methods = set(r['conformal_method'] for r in all_results)

    stats = {
        'total_entries': len(all_results),
        'models': sorted(models),
        'tasks': sorted(tasks),
        'dtypes': sorted(dtypes),
        'methods': sorted(methods),
        'model_task_coverage': {}
    }

    # Coverage matrix
    for model in sorted(models):
        model_results = [r for r in all_results if r['model'] == model]
        model_tasks = set(r['task'] for r in model_results)
        model_dtypes = set(r['dtype'] for r in model_results)
        stats['model_task_coverage'][model] = {
            'tasks': sorted(model_tasks),
            'dtypes': sorted(model_dtypes),
            'task_count': len(model_tasks)
        }

    return stats


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Merge results from model directories")
    parser.add_argument(
        '--results-dir',
        type=str,
        default='./outputs/results',
        help='Path to results directory'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default='./outputs/results/all_results_merged.json',
        help='Path to save merged results'
    )

    args = parser.parse_args()

    stats = merge_results(args.results_dir, args.output_path)

    print("\n" + "=" * 60)
    print("MERGE STATISTICS")
    print("=" * 60)
    print(f"Total entries: {stats['total_entries']}")
    print(f"Models: {stats['models']}")
    print(f"Tasks: {stats['tasks']}")
    print(f"Data types: {stats['dtypes']}")
    print(f"Methods: {stats['methods']}")
    print("\nModel coverage:")
    for model, coverage in stats['model_task_coverage'].items():
        print(f"  {model}: {coverage['task_count']}/5 tasks, dtypes: {coverage['dtypes']}")
    print("=" * 60)


if __name__ == "__main__":
    main()

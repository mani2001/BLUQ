#!/usr/bin/env python3
"""
Regenerate all visualizations from benchmark results.
Loads all result JSON files and creates updated figures.
"""

import json
import logging
from pathlib import Path
from src.visualization.result_visualizer import ResultVisualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Output directory for figures
    output_dir = Path("outputs/results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all result JSON files
    results_dir = Path("outputs/results")
    result_files = list(results_dir.glob("*/results_*.json"))

    logger.info(f"Found {len(result_files)} result files")

    # Create visualizer
    visualizer = ResultVisualizer(output_dir=str(output_dir))

    # Load all results
    for result_file in result_files:
        logger.info(f"Loading {result_file}")
        visualizer.load_results_from_json(str(result_file))

    logger.info(f"Loaded {len(visualizer.results)} total results")

    # List unique models and tasks
    models = sorted(set(r.model_name for r in visualizer.results))
    tasks = sorted(set(r.task_name for r in visualizer.results))
    logger.info(f"Models: {models}")
    logger.info(f"Tasks: {tasks}")

    # Generate all visualizations
    visualizer.generate_all_visualizations(dtype_filter='float16')

    logger.info(f"Visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()

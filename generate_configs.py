"""
generate_configs.py
Generate default configuration files for the benchmark.
"""

import logging
from pathlib import Path

from src.data.dataset_config import DatasetConfigManager
from src.models.model_config import ModelConfigManager

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Generate default configuration files."""
    logger.info("Generating default configuration files...")
    
    config_dir = Path("./configs")
    config_dir.mkdir(exist_ok=True)
    
    # Generate dataset config
    logger.info("\nGenerating dataset configuration...")
    data_manager = DatasetConfigManager()
    data_manager.print_summary()
    data_manager.save(config_dir / "dataset_config.yaml", format='yaml')
    data_manager.save(config_dir / "dataset_config.json", format='json')
    
    # Generate model config
    logger.info("\nGenerating model configuration...")
    model_manager = ModelConfigManager()
    model_manager.print_summary()
    model_manager.save(config_dir / "model_config.yaml", format='yaml')
    model_manager.save(config_dir / "model_config.json", format='json')
    
    # Generate minimal test config
    logger.info("\nGenerating minimal test configuration...")
    from src.models.model_config import ModelBenchmarkConfig, DefaultModelConfigs
    
    minimal_config = ModelBenchmarkConfig(
        models={
            "tinyllama-1.1b": DefaultModelConfigs.create_pipeline_config("tinyllama-1.1b"),
            "phi-2": DefaultModelConfigs.create_pipeline_config("phi-2")
        },
        cache_dir="./models/cache",
        output_dir="./results/test"
    )
    
    minimal_manager = ModelConfigManager(minimal_config)
    minimal_manager.save(config_dir / "model_config_minimal.yaml", format='yaml')
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Configuration files saved to: {config_dir}")
    logger.info(f"{'='*80}")
    logger.info("\nYou can now:")
    logger.info("  1. Edit the YAML files to customize settings")
    logger.info("  2. Run: python run_benchmark.py --data-config configs/dataset_config.yaml --model-config configs/model_config.yaml")
    logger.info("  3. Or quick test: python run_benchmark.py --quick-test --models tinyllama-1.1b")


if __name__ == "__main__":
    main()
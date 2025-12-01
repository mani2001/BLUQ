#!/usr/bin/env python3
"""
BLUQ Setup and Run Script
Cross-platform Python script to set up environment and run benchmarks.
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler


def setup_logging(log_dir: Path, timestamp: str) -> logging.Logger:
    """Set up logging to both file and console."""
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"benchmark_{timestamp}.log"

    # Create logger
    logger = logging.getLogger('BLUQ')
    logger.setLevel(logging.DEBUG)

    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_format)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger, log_file


# Colors for terminal output (ANSI codes)
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color

    @classmethod
    def disable(cls):
        """Disable colors for non-TTY output."""
        cls.RED = cls.GREEN = cls.YELLOW = cls.BLUE = cls.NC = ''


# Disable colors if not a TTY
if not sys.stdout.isatty():
    Colors.disable()


def print_header(text: str):
    print(f"\n{Colors.BLUE}{'='*70}{Colors.NC}")
    print(f"{Colors.BLUE}{text}{Colors.NC}")
    print(f"{Colors.BLUE}{'='*70}{Colors.NC}\n")


def print_success(text: str):
    print(f"{Colors.GREEN}[OK]{Colors.NC} {text}")


def print_warning(text: str):
    print(f"{Colors.YELLOW}[WARN]{Colors.NC} {text}")


def print_error(text: str):
    print(f"{Colors.RED}[ERROR]{Colors.NC} {text}")


def print_info(text: str):
    print(f"{Colors.BLUE}[INFO]{Colors.NC} {text}")


def run_command(cmd: list, cwd: str = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    print_info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True
    )
    if check and result.returncode != 0:
        print_error(f"Command failed: {result.stderr}")
        raise subprocess.CalledProcessError(result.returncode, cmd)
    return result


def check_gpu():
    """Check for available GPU."""
    print_header("Checking GPU Availability")

    # Check for NVIDIA GPU
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,memory.free', '--format=csv'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print_success("NVIDIA GPU detected:")
            print(result.stdout)
            return 'cuda'
    except FileNotFoundError:
        pass

    # Check for Apple Silicon
    import platform
    if platform.system() == 'Darwin' and platform.machine() == 'arm64':
        print_success("Apple Silicon detected (MPS available)")
        return 'mps'

    print_warning("No GPU detected - will run on CPU (this will be slow)")
    return 'cpu'


def setup_environment(project_root: Path, venv_dir: Path, skip_setup: bool):
    """Set up Python virtual environment and install dependencies."""
    print_header("Setting Up Environment")

    if skip_setup:
        print_info("Skipping environment setup")
        return

    os.chdir(project_root)

    # Create virtual environment
    if not venv_dir.exists():
        print_info("Creating virtual environment...")
        subprocess.run([sys.executable, '-m', 'venv', str(venv_dir)], check=True)
        print_success("Virtual environment created")
    else:
        print_info("Virtual environment already exists")

    # Determine pip path
    if sys.platform == 'win32':
        pip_path = venv_dir / 'Scripts' / 'pip'
        python_path = venv_dir / 'Scripts' / 'python'
    else:
        pip_path = venv_dir / 'bin' / 'pip'
        python_path = venv_dir / 'bin' / 'python'

    # Upgrade pip
    print_info("Upgrading pip...")
    subprocess.run([str(pip_path), 'install', '--upgrade', 'pip', '-q'], check=True)

    # Install requirements
    print_info("Installing dependencies...")
    requirements_file = project_root / 'requirements.txt'
    subprocess.run([str(pip_path), 'install', '-r', str(requirements_file), '-q'], check=True)

    # Install visualization dependencies
    print_info("Installing visualization dependencies...")
    subprocess.run([str(pip_path), 'install', 'matplotlib', 'seaborn', 'psutil', '-q'], check=True)

    print_success("All dependencies installed")

    return python_path


def generate_configs(project_root: Path, python_path: Path):
    """Generate configuration files if they don't exist."""
    print_header("Generating Configuration Files")

    config_dir = project_root / 'configs'
    dataset_config = config_dir / 'dataset_config.yaml'
    model_config = config_dir / 'model_config.yaml'

    if not dataset_config.exists() or not model_config.exists():
        print_info("Generating default configuration files...")
        subprocess.run([str(python_path), 'generate_configs.py'], cwd=project_root, check=True)
        print_success("Configuration files generated")
    else:
        print_info("Configuration files already exist")


def verify_setup(project_root: Path, python_path: Path, skip_verify: bool):
    """Run verification script."""
    print_header("Verifying Setup")

    if skip_verify:
        print_info("Skipping verification")
        return True

    print_info("Running verification script...")
    verify_script = project_root / 'scripts' / 'verify_setup.py'

    result = subprocess.run(
        [str(python_path), str(verify_script)],
        cwd=project_root
    )

    if result.returncode == 0:
        print_success("Verification passed")
        return True
    else:
        print_error("Verification failed!")
        print_info("Please check the errors above and fix them before running the benchmark")
        return False


def run_benchmark(
    project_root: Path,
    python_path: Path,
    mode: str,
    tasks: list,
    models: list,
    dtypes: list,
    num_samples: int,
    max_batch_size: int,
    output_dir: Path,
    strategies: list,
    conformal_methods: list
):
    """Run the benchmark."""
    print_header("Running Benchmark")

    os.chdir(project_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build command
    cmd = [
        str(python_path),
        'run_full_benchmark.py',
        '--mode', mode,
        '--tasks', *tasks,
        '--models', *models,
        '--dtypes', *dtypes,
        '--output-dir', str(output_dir),
        '--strategies', *strategies,
        '--conformal-methods', *conformal_methods,
    ]

    if num_samples:
        cmd.extend(['--num-samples', str(num_samples)])

    if max_batch_size:
        cmd.extend(['--max-batch-size', str(max_batch_size)])

    # Log the command
    print_info(f"Executing: {' '.join(cmd)}")

    # Save run info
    run_log = output_dir / 'run_command.log'
    with open(run_log, 'w') as f:
        f.write(f"Command: {' '.join(cmd)}\n")
        f.write(f"Started: {datetime.now().isoformat()}\n")

    # Run benchmark
    start_time = time.time()

    process = subprocess.Popen(
        cmd,
        cwd=project_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    # Stream output
    for line in iter(process.stdout.readline, ''):
        print(line, end='')

    process.wait()

    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    # Record completion
    with open(run_log, 'a') as f:
        f.write(f"Completed: {datetime.now().isoformat()}\n")
        f.write(f"Duration: {minutes}m {seconds}s\n")
        f.write(f"Return code: {process.returncode}\n")

    return process.returncode, minutes, seconds


def main():
    # Initialize timestamp for logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser(
        description="BLUQ Setup and Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick test with default settings
    python setup_and_run.py --mode short

    # Full benchmark with all tasks
    python setup_and_run.py --mode long --tasks qa rc ci drs ds

    # Skip setup if environment is already configured
    python setup_and_run.py --skip-setup --mode short

    # Custom configuration
    python setup_and_run.py --num-samples 500 --tasks qa --models tinyllama-1.1b
        """
    )

    parser.add_argument(
        '--mode', '-m',
        choices=['short', 'long', 'custom'],
        default='short',
        help='Benchmark mode: short (100 samples), long (10000 samples)'
    )

    parser.add_argument(
        '--tasks', '-t',
        nargs='+',
        default=['qa', 'rc', 'ci'],
        help='Tasks to run: qa rc ci drs ds'
    )

    parser.add_argument(
        '--models', '-M',
        nargs='+',
        default=['tinyllama-1.1b', 'phi-2'],
        help='Models to evaluate'
    )

    parser.add_argument(
        '--dtypes', '-d',
        nargs='+',
        default=['float16'],
        choices=['float16', 'float32'],
        help='Data types to evaluate'
    )

    parser.add_argument(
        '--num-samples', '-n',
        type=int,
        default=None,
        help='Number of samples (overrides mode)'
    )

    parser.add_argument(
        '--max-batch-size', '-b',
        type=int,
        default=None,
        help='Maximum batch size (default: auto-detect from GPU tier - 128 for A100/H100, 64 mid-range, 32 low-end)'
    )

    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./outputs/results',
        help='Output directory for results'
    )

    parser.add_argument(
        '--strategies',
        nargs='+',
        default=['base'],
        choices=['base', 'shared_instruction', 'task_specific'],
        help='Prompting strategies'
    )

    parser.add_argument(
        '--conformal-methods',
        nargs='+',
        default=['lac', 'aps'],
        choices=['lac', 'aps'],
        help='Conformal prediction methods'
    )

    parser.add_argument(
        '--skip-setup', '-s',
        action='store_true',
        help='Skip environment setup'
    )

    parser.add_argument(
        '--skip-verify', '-v',
        action='store_true',
        help='Skip verification step'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print configuration and exit without running'
    )

    args = parser.parse_args()

    # Determine paths
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent
    venv_dir = project_root / '.venv'
    log_dir = project_root / 'logs'
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir

    # Set up logging
    logger, log_file = setup_logging(log_dir, timestamp)
    logger.info(f"Starting BLUQ Benchmark Runner")
    logger.info(f"Log file: {log_file}")

    # Print configuration
    print_header("BLUQ Benchmark Configuration")
    print(f"Project Root:    {project_root}")
    print(f"Virtual Env:     {venv_dir}")
    print(f"Mode:            {args.mode}")
    print(f"Tasks:           {' '.join(args.tasks)}")
    print(f"Models:          {' '.join(args.models)}")
    print(f"Data Types:      {' '.join(args.dtypes)}")
    print(f"Strategies:      {' '.join(args.strategies)}")
    print(f"CP Methods:      {' '.join(args.conformal_methods)}")
    print(f"Output Dir:      {output_dir}")
    print(f"Log File:        {log_file}")
    print(f"Skip Setup:      {args.skip_setup}")
    print(f"Skip Verify:     {args.skip_verify}")
    if args.num_samples:
        print(f"Num Samples:     {args.num_samples}")
    if args.max_batch_size:
        print(f"Max Batch Size:  {args.max_batch_size}")
    else:
        print(f"Max Batch Size:  auto-detect (based on GPU tier)")

    # Log configuration
    logger.info(f"Configuration: mode={args.mode}, tasks={args.tasks}, models={args.models}")
    logger.info(f"Output directory: {output_dir}")

    if args.dry_run:
        print_info("\nDry run mode - exiting without execution")
        logger.info("Dry run mode - exiting")
        return 0

    # Check GPU
    gpu_type = check_gpu()
    logger.info(f"GPU type detected: {gpu_type}")

    # Setup environment
    if args.skip_setup:
        # Determine python path from existing venv
        if sys.platform == 'win32':
            python_path = venv_dir / 'Scripts' / 'python'
        else:
            python_path = venv_dir / 'bin' / 'python'

        if not python_path.exists():
            print_warning(f"Virtual environment not found at {venv_dir}")
            print_info("Using system Python...")
            python_path = Path(sys.executable)
    else:
        python_path = setup_environment(project_root, venv_dir, args.skip_setup)
        if python_path is None:
            if sys.platform == 'win32':
                python_path = venv_dir / 'Scripts' / 'python'
            else:
                python_path = venv_dir / 'bin' / 'python'

    # Generate configs
    generate_configs(project_root, python_path)

    # Verify setup
    if not verify_setup(project_root, python_path, args.skip_verify):
        return 1

    # Run benchmark
    return_code, minutes, seconds = run_benchmark(
        project_root=project_root,
        python_path=python_path,
        mode=args.mode,
        tasks=args.tasks,
        models=args.models,
        dtypes=args.dtypes,
        num_samples=args.num_samples,
        max_batch_size=args.max_batch_size,
        output_dir=output_dir,
        strategies=args.strategies,
        conformal_methods=args.conformal_methods
    )

    # Summary
    print_header("Benchmark Complete")
    print(f"Duration: {minutes}m {seconds}s")
    print(f"Results saved to: {output_dir}")
    print(f"Log file: {log_file}")
    print(f"GPU Type: {gpu_type}")

    if output_dir.exists():
        print("\nOutput files:")
        for f in sorted(output_dir.iterdir()):
            print(f"  {f.name}")

    # Final logging
    logger.info(f"Benchmark complete. Duration: {minutes}m {seconds}s")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Return code: {return_code}")

    if return_code == 0:
        print_success("\nBenchmark finished successfully!")
        logger.info("Benchmark finished successfully")
    else:
        print_error(f"\nBenchmark finished with errors (code: {return_code})")
        logger.error(f"Benchmark finished with errors (code: {return_code})")

    return return_code


if __name__ == "__main__":
    sys.exit(main())

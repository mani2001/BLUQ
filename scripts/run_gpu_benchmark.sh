#!/bin/bash
#===============================================================================
# BLUQ GPU Benchmark Runner
# Automates setup and execution of the benchmark suite on GPU environments
#===============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default configuration
VENV_DIR="${PROJECT_ROOT}/.venv"
PYTHON_VERSION="python3"
MODE="short"
TASKS="qa rc ci"
MODELS="tinyllama-1.1b phi-2"
DTYPES="float16"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/results"
SKIP_SETUP=false
SKIP_VERIFY=false
DRY_RUN=false

#===============================================================================
# Logging Setup
#===============================================================================

# Create logs directory
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "$LOG_DIR"

# Generate timestamp for log files
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/benchmark_${TIMESTAMP}.log"
ERROR_LOG="${LOG_DIR}/benchmark_${TIMESTAMP}_error.log"

# Tee output to both console and log file
exec 1> >(tee -a "$LOG_FILE")
exec 2> >(tee -a "$ERROR_LOG" >&2)

log_message() {
    local level="$1"
    local message="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[${timestamp}] [${level}] ${message}" >> "$LOG_FILE"
}

log_info() {
    log_message "INFO" "$1"
}

log_warn() {
    log_message "WARN" "$1"
}

log_error() {
    log_message "ERROR" "$1"
}

#===============================================================================
# Helper Functions
#===============================================================================

print_header() {
    echo -e "\n${BLUE}================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

BLUQ Benchmark Runner - Sets up environment and runs benchmarks on GPU

OPTIONS:
    -h, --help              Show this help message
    -m, --mode MODE         Benchmark mode: short (100 samples), long (10000 samples), custom
                            Default: short
    -t, --tasks TASKS       Tasks to run (space-separated): qa rc ci drs ds
                            Default: "qa rc ci"
    -M, --models MODELS     Models to evaluate (space-separated)
                            Default: "tinyllama-1.1b phi-2"
    -d, --dtypes DTYPES     Data types (space-separated): float16 float32
                            Default: "float16"
    -n, --num-samples N     Number of samples (overrides mode)
    -o, --output DIR        Output directory for results
                            Default: ./outputs/results
    -s, --skip-setup        Skip environment setup (use if already set up)
    -v, --skip-verify       Skip verification step
    --dry-run               Print configuration and exit without running

EXAMPLES:
    # Quick test with default settings
    $(basename "$0") --mode short

    # Full benchmark with all tasks
    $(basename "$0") --mode long --tasks "qa rc ci drs ds" --models "tinyllama-1.1b phi-2 gemma-2b"

    # Skip setup if environment is already configured
    $(basename "$0") --skip-setup --mode short

    # Custom number of samples
    $(basename "$0") --num-samples 500 --tasks qa --models tinyllama-1.1b

EOF
}

#===============================================================================
# Parse Arguments
#===============================================================================

NUM_SAMPLES=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        -t|--tasks)
            TASKS="$2"
            shift 2
            ;;
        -M|--models)
            MODELS="$2"
            shift 2
            ;;
        -d|--dtypes)
            DTYPES="$2"
            shift 2
            ;;
        -n|--num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -s|--skip-setup)
            SKIP_SETUP=true
            shift
            ;;
        -v|--skip-verify)
            SKIP_VERIFY=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

#===============================================================================
# Display Configuration
#===============================================================================

print_header "BLUQ Benchmark Configuration"

echo "Project Root:    ${PROJECT_ROOT}"
echo "Virtual Env:     ${VENV_DIR}"
echo "Mode:            ${MODE}"
echo "Tasks:           ${TASKS}"
echo "Models:          ${MODELS}"
echo "Data Types:      ${DTYPES}"
echo "Output Dir:      ${OUTPUT_DIR}"
echo "Skip Setup:      ${SKIP_SETUP}"
echo "Skip Verify:     ${SKIP_VERIFY}"
echo "Log File:        ${LOG_FILE}"
if [[ -n "$NUM_SAMPLES" ]]; then
    echo "Num Samples:     ${NUM_SAMPLES}"
fi
echo ""

log_info "Configuration: mode=${MODE}, tasks=${TASKS}, models=${MODELS}, dtypes=${DTYPES}"

if [[ "$DRY_RUN" == "true" ]]; then
    print_info "Dry run mode - exiting without execution"
    log_info "Dry run mode - exiting"
    exit 0
fi

#===============================================================================
# Step 1: Check GPU Availability
#===============================================================================

print_header "Step 1: Checking GPU Availability"

# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    print_success "NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
    GPU_TYPE="cuda"
# Check for Apple Silicon
elif [[ "$(uname)" == "Darwin" ]] && [[ "$(uname -m)" == "arm64" ]]; then
    print_success "Apple Silicon detected (MPS available)"
    GPU_TYPE="mps"
else
    print_warning "No GPU detected - will run on CPU (this will be slow)"
    GPU_TYPE="cpu"
fi

#===============================================================================
# Step 2: Environment Setup
#===============================================================================

if [[ "$SKIP_SETUP" == "false" ]]; then
    print_header "Step 2: Setting Up Environment"

    cd "$PROJECT_ROOT"

    # Create virtual environment if it doesn't exist
    if [[ ! -d "$VENV_DIR" ]]; then
        print_info "Creating virtual environment..."
        $PYTHON_VERSION -m venv "$VENV_DIR"
        print_success "Virtual environment created"
    else
        print_info "Virtual environment already exists"
    fi

    # Activate virtual environment
    source "${VENV_DIR}/bin/activate"
    print_success "Virtual environment activated"

    # Upgrade pip
    print_info "Upgrading pip..."
    pip install --upgrade pip -q

    # Install requirements
    print_info "Installing dependencies..."
    pip install -r requirements.txt -q

    # Install additional dependencies for visualization
    print_info "Installing visualization dependencies..."
    pip install matplotlib seaborn psutil -q

    print_success "All dependencies installed"
else
    print_header "Step 2: Skipping Environment Setup"

    # Activate virtual environment
    if [[ -d "$VENV_DIR" ]]; then
        source "${VENV_DIR}/bin/activate"
        print_success "Virtual environment activated"
    else
        print_warning "Virtual environment not found at ${VENV_DIR}"
        print_info "Attempting to use system Python..."
    fi
fi

#===============================================================================
# Step 3: Generate Configurations
#===============================================================================

print_header "Step 3: Generating Configuration Files"

cd "$PROJECT_ROOT"

if [[ ! -f "configs/dataset_config.yaml" ]] || [[ ! -f "configs/model_config.yaml" ]]; then
    print_info "Generating default configuration files..."
    python generate_configs.py
    print_success "Configuration files generated"
else
    print_info "Configuration files already exist"
fi

#===============================================================================
# Step 4: Verification
#===============================================================================

if [[ "$SKIP_VERIFY" == "false" ]]; then
    print_header "Step 4: Verifying Setup"

    cd "$PROJECT_ROOT"

    print_info "Running verification script..."
    if python scripts/verify_setup.py; then
        print_success "Verification passed"
    else
        print_error "Verification failed!"
        print_info "Please check the errors above and fix them before running the benchmark"
        exit 1
    fi
else
    print_header "Step 4: Skipping Verification"
fi

#===============================================================================
# Step 5: Run Benchmark
#===============================================================================

print_header "Step 5: Running Benchmark"

cd "$PROJECT_ROOT"

# Build command
CMD="python run_full_benchmark.py"
CMD+=" --mode ${MODE}"
CMD+=" --tasks ${TASKS}"
CMD+=" --models ${MODELS}"
CMD+=" --dtypes ${DTYPES}"
CMD+=" --output-dir ${OUTPUT_DIR}"

if [[ -n "$NUM_SAMPLES" ]]; then
    CMD+=" --num-samples ${NUM_SAMPLES}"
fi

# Log the command
echo "Executing: $CMD"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Save the command to a log file
echo "Command: $CMD" > "${OUTPUT_DIR}/run_command.log"
echo "Started: $(date)" >> "${OUTPUT_DIR}/run_command.log"
echo "GPU Type: ${GPU_TYPE}" >> "${OUTPUT_DIR}/run_command.log"

# Run the benchmark
START_TIME=$(date +%s)

eval $CMD

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

#===============================================================================
# Summary
#===============================================================================

print_header "Benchmark Complete"

echo "Duration: ${MINUTES}m ${SECONDS}s"
echo "Results saved to: ${OUTPUT_DIR}"
echo "Log file: ${LOG_FILE}"
echo "Error log: ${ERROR_LOG}"
echo ""

# List output files
if [[ -d "$OUTPUT_DIR" ]]; then
    echo "Output files:"
    ls -la "$OUTPUT_DIR" | tail -n +2
fi

echo ""
print_success "Benchmark finished successfully!"

# Record completion
echo "Completed: $(date)" >> "${OUTPUT_DIR}/run_command.log"
echo "Duration: ${MINUTES}m ${SECONDS}s" >> "${OUTPUT_DIR}/run_command.log"

# Final logging
log_info "Benchmark complete. Duration: ${MINUTES}m ${SECONDS}s"
log_info "Results saved to: ${OUTPUT_DIR}"

# Clean up old logs (keep last 10)
cleanup_old_logs() {
    local log_count=$(ls -1 "${LOG_DIR}"/benchmark_*.log 2>/dev/null | wc -l)
    if [[ $log_count -gt 20 ]]; then
        print_info "Cleaning up old log files..."
        ls -1t "${LOG_DIR}"/benchmark_*.log | tail -n +21 | xargs rm -f 2>/dev/null || true
        ls -1t "${LOG_DIR}"/benchmark_*_error.log | tail -n +21 | xargs rm -f 2>/dev/null || true
    fi
}

cleanup_old_logs

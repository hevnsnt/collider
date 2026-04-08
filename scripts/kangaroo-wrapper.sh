#!/bin/bash
#
# Kangaroo Wrapper Script
# Wraps JeanLucPons/Kangaroo for easy Bitcoin puzzle solving
#
# Usage:
#   ./kangaroo-wrapper.sh setup              # Clone and build Kangaroo
#   ./kangaroo-wrapper.sh puzzle <N> <pubkey> # Solve puzzle N with public key
#   ./kangaroo-wrapper.sh resume <workfile>   # Resume from work file
#   ./kangaroo-wrapper.sh status <workfile>   # Show work file status

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
KANGAROO_DIR="$PROJECT_ROOT/external/Kangaroo"
KANGAROO_BIN="$KANGAROO_DIR/Kangaroo"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Setup: Clone and build JeanLucPons/Kangaroo
setup() {
    log_info "Setting up JeanLucPons/Kangaroo..."

    mkdir -p "$PROJECT_ROOT/external"

    if [ -d "$KANGAROO_DIR" ]; then
        log_info "Kangaroo directory exists, pulling latest..."
        cd "$KANGAROO_DIR"
        git pull
    else
        log_info "Cloning JeanLucPons/Kangaroo..."
        git clone https://github.com/JeanLucPons/Kangaroo.git "$KANGAROO_DIR"
        cd "$KANGAROO_DIR"
    fi

    log_info "Building Kangaroo..."

    # Detect OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux build
        make clean 2>/dev/null || true
        make -j$(nproc)
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS build (may need adjustments)
        log_warning "macOS build may require manual adjustments"
        make clean 2>/dev/null || true
        make -j$(sysctl -n hw.ncpu)
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        # Windows - use Visual Studio project
        log_warning "Windows: Build using Visual Studio solution file"
        log_info "Open Kangaroo.sln in Visual Studio and build Release x64"
    fi

    if [ -f "$KANGAROO_BIN" ]; then
        log_success "Kangaroo built successfully: $KANGAROO_BIN"
    else
        log_warning "Build may have failed. Check for Kangaroo binary."
    fi
}

# Detect GPU configuration
detect_gpus() {
    if command -v nvidia-smi &> /dev/null; then
        GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
        GPU_IDS=$(seq -s, 0 $((GPU_COUNT-1)))
        log_info "Detected $GPU_COUNT NVIDIA GPU(s): $GPU_IDS"
        echo "$GPU_IDS"
    else
        log_warning "No NVIDIA GPUs detected, will use CPU mode"
        echo ""
    fi
}

# Solve a puzzle
solve_puzzle() {
    local PUZZLE_NUM="$1"
    local PUBKEY="$2"
    local WORK_FILE="${3:-puzzle${PUZZLE_NUM}.work}"

    if [ -z "$PUZZLE_NUM" ] || [ -z "$PUBKEY" ]; then
        log_error "Usage: $0 puzzle <puzzle_number> <public_key_hex> [work_file]"
        echo ""
        echo "Example:"
        echo "  $0 puzzle 135 02e0a8b039..."
        exit 1
    fi

    if [ ! -f "$KANGAROO_BIN" ]; then
        log_error "Kangaroo not found. Run '$0 setup' first."
        exit 1
    fi

    # Calculate range
    local RANGE_LOW=$((PUZZLE_NUM - 1))
    local RANGE_HIGH=$PUZZLE_NUM

    log_info "Solving Puzzle #$PUZZLE_NUM"
    log_info "Public Key: $PUBKEY"
    log_info "Range: 2^$RANGE_LOW to 2^$RANGE_HIGH"
    log_info "Work File: $WORK_FILE"

    # Detect GPUs
    local GPU_IDS=$(detect_gpus)

    # Build command
    local CMD="$KANGAROO_BIN -t $PUBKEY -r $RANGE_LOW:$RANGE_HIGH"

    if [ -n "$GPU_IDS" ]; then
        CMD="$CMD -gpu $GPU_IDS"
    fi

    # Resume if work file exists
    if [ -f "$WORK_FILE" ]; then
        log_info "Resuming from existing work file..."
        CMD="$CMD -i $WORK_FILE"
    fi

    # Save progress
    CMD="$CMD -w $WORK_FILE"

    log_info "Running: $CMD"
    echo ""

    # Create output directory
    mkdir -p "$PROJECT_ROOT/output"

    # Run Kangaroo
    $CMD 2>&1 | tee "$PROJECT_ROOT/output/puzzle${PUZZLE_NUM}_$(date +%Y%m%d_%H%M%S).log"
}

# Resume from work file
resume() {
    local WORK_FILE="$1"

    if [ -z "$WORK_FILE" ] || [ ! -f "$WORK_FILE" ]; then
        log_error "Usage: $0 resume <work_file>"
        exit 1
    fi

    if [ ! -f "$KANGAROO_BIN" ]; then
        log_error "Kangaroo not found. Run '$0 setup' first."
        exit 1
    fi

    log_info "Resuming from: $WORK_FILE"

    # Extract target from work file
    local TARGET=$(grep "^DP:" "$WORK_FILE" | head -1 | cut -d' ' -f2 || echo "")

    if [ -z "$TARGET" ]; then
        log_error "Could not extract target from work file"
        exit 1
    fi

    local GPU_IDS=$(detect_gpus)
    local CMD="$KANGAROO_BIN -i $WORK_FILE -w $WORK_FILE"

    if [ -n "$GPU_IDS" ]; then
        CMD="$CMD -gpu $GPU_IDS"
    fi

    log_info "Running: $CMD"
    $CMD
}

# Show work file status
status() {
    local WORK_FILE="$1"

    if [ -z "$WORK_FILE" ] || [ ! -f "$WORK_FILE" ]; then
        log_error "Usage: $0 status <work_file>"
        exit 1
    fi

    log_info "Work file status: $WORK_FILE"
    echo ""

    # Parse work file (format varies by version)
    echo "File size: $(du -h "$WORK_FILE" | cut -f1)"
    echo "Lines: $(wc -l < "$WORK_FILE")"

    # Count distinguished points
    local DP_COUNT=$(grep -c "^DP:" "$WORK_FILE" 2>/dev/null || echo "N/A")
    echo "Distinguished Points: $DP_COUNT"

    # Show first few lines
    echo ""
    echo "Header:"
    head -10 "$WORK_FILE"
}

# Show help
show_help() {
    echo "Kangaroo Wrapper - Bitcoin Puzzle Solver"
    echo ""
    echo "Usage:"
    echo "  $0 setup                          Clone and build JeanLucPons/Kangaroo"
    echo "  $0 puzzle <N> <pubkey> [workfile] Solve puzzle #N with public key"
    echo "  $0 resume <workfile>              Resume from work file"
    echo "  $0 status <workfile>              Show work file status"
    echo "  $0 help                           Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 setup"
    echo "  $0 puzzle 135 02e0a8b039282faf6fe0fd769cfbc4b6b4cf8758ba68220eac420e32b91ddfa90"
    echo "  $0 resume puzzle135.work"
    echo ""
    echo "Requirements:"
    echo "  - Git (for cloning)"
    echo "  - C++ compiler (g++ or clang++)"
    echo "  - CUDA toolkit (for GPU acceleration)"
    echo ""
    echo "Notes:"
    echo "  - Public key must be in compressed (02/03) or uncompressed (04) hex format"
    echo "  - Puzzle #135 requires the public key which is exposed when the address"
    echo "    has made an outgoing transaction"
    echo "  - Work files allow resuming long-running searches"
}

# Main dispatch
case "${1:-help}" in
    setup)
        setup
        ;;
    puzzle)
        solve_puzzle "$2" "$3" "$4"
        ;;
    resume)
        resume "$2"
        ;;
    status)
        status "$2"
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        log_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac

#!/bin/bash

# Size sweep experiment script
# Tests slowdown behavior by varying kernel sizes
# Case 1: Fix GEMV, vary GEMM size
# Case 2: Fix GEMM, vary GEMV size

set -e

# Default parameters
NUM_SMS=""  # Empty means use all SMs
NUM_ITERS=8
REPEATS=3
OUTPUT_DIR="size_sweep_results"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_sms)
            NUM_SMS="$2"
            shift 2
            ;;
        --num_iters)
            NUM_ITERS="$2"
            shift 2
            ;;
        --repeats)
            REPEATS="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --num_sms <N>       Number of SMs to use (default: all SMs)"
            echo "  --num_iters <N>     Number of iterations per run (default: 8)"
            echo "  --repeats <N>       Number of measurement repeats (default: 3)"
            echo "  --output_dir <path> Output directory (default: size_sweep_results)"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Determine SM argument
if [ -z "$NUM_SMS" ]; then
    SM_ARG=""
    echo "Using all available SMs"
else
    SM_ARG="--min_sms $NUM_SMS --max_sms $NUM_SMS"
    echo "Using $NUM_SMS SMs"
fi

EXECUTABLE="./build/GEMV_GEMM_contention_test"

if [ ! -f "$EXECUTABLE" ]; then
    echo "Error: Executable not found at $EXECUTABLE"
    echo "Please build the project first: cd build && cmake --build . -j"
    exit 1
fi

echo "================================================"
echo "Size Sweep Experiment"
echo "================================================"
echo "Iterations per run: $NUM_ITERS"
echo "Measurement repeats: $REPEATS"
echo "Output directory: $OUTPUT_DIR"
echo ""

# =============================================================================
# Case 1: Fix GEMV size (32768x32768), vary GEMM size
# =============================================================================
echo "================================================"
echo "Case 1: Fixed GEMV (32768x32768), Varying GEMM"
echo "================================================"

FIXED_GEMV_M=32768
FIXED_GEMV_N=32768
CASE1_CSV="$OUTPUT_DIR/case1_fixed_gemv_vary_gemm.csv"

# GEMM sizes to test (varying from small to large)
# Targeting range from ~5ms to ~300ms per iteration
GEMM_SIZES=(1024 1536 2048 2560 3072 3584 4096 4608 5120)

echo "Testing GEMM sizes: ${GEMM_SIZES[@]}"
echo ""

# Create header
echo "num_sms,gemv_M,gemv_N,gemm_size,mode,gemv_gflops,gemm_gflops,gemv_time_ms,gemm_time_ms,gemv_slowdown,gemm_slowdown,overlap_pct" > "$CASE1_CSV"

for GEMM_SIZE in "${GEMM_SIZES[@]}"; do
    echo "Running: GEMV=${FIXED_GEMV_M}x${FIXED_GEMV_N}, GEMM=${GEMM_SIZE}^3"
    
    # Use a temporary file for this run
    TEMP_CSV="${OUTPUT_DIR}/temp_case1_${GEMM_SIZE}.csv"
    
    $EXECUTABLE \
        $SM_ARG \
        --gemv_M $FIXED_GEMV_M \
        --gemv_N $FIXED_GEMV_N \
        --gemm_size $GEMM_SIZE \
        --num_iters $NUM_ITERS \
        --repeats $REPEATS \
        --csv "$TEMP_CSV" 2>&1 | grep -E "(Testing with|GEMV Only|GEMM Only|Concurrent)"
    
    # Append results (skip header)
    tail -n +2 "$TEMP_CSV" >> "$CASE1_CSV"
    rm "$TEMP_CSV"
    
    echo ""
done

echo "Case 1 results saved to: $CASE1_CSV"
echo ""

# =============================================================================
# Case 2: Fix GEMM size (4096), vary GEMV size
# =============================================================================
echo "================================================"
echo "Case 2: Fixed GEMM (4096^3), Varying GEMV"
echo "================================================"

FIXED_GEMM_SIZE=4096
CASE2_CSV="$OUTPUT_DIR/case2_fixed_gemm_vary_gemv.csv"

# GEMV sizes to test (varying from small to large)
# Targeting range from ~5ms to ~300ms per iteration
# GEMV scales as M*N, so we vary the matrix size
GEMV_SIZES=(8192 12288 16384 20480 24576 28672 32768 36864 40960)

echo "Testing GEMV sizes (MxN): ${GEMV_SIZES[@]}"
echo ""

# Create header
echo "num_sms,gemv_M,gemv_N,gemm_size,mode,gemv_gflops,gemm_gflops,gemv_time_ms,gemm_time_ms,gemv_slowdown,gemm_slowdown,overlap_pct" > "$CASE2_CSV"

for GEMV_SIZE in "${GEMV_SIZES[@]}"; do
    echo "Running: GEMV=${GEMV_SIZE}x${GEMV_SIZE}, GEMM=${FIXED_GEMM_SIZE}^3"
    
    # Use a temporary file for this run
    TEMP_CSV="${OUTPUT_DIR}/temp_case2_${GEMV_SIZE}.csv"
    
    $EXECUTABLE \
        $SM_ARG \
        --gemv_M $GEMV_SIZE \
        --gemv_N $GEMV_SIZE \
        --gemm_size $FIXED_GEMM_SIZE \
        --num_iters $NUM_ITERS \
        --repeats $REPEATS \
        --csv "$TEMP_CSV" 2>&1 | grep -E "(Testing with|GEMV Only|GEMM Only|Concurrent)"
    
    # Append results (skip header)
    tail -n +2 "$TEMP_CSV" >> "$CASE2_CSV"
    rm "$TEMP_CSV"
    
    echo ""
done

echo "Case 2 results saved to: $CASE2_CSV"
echo ""

echo "================================================"
echo "Experiment Complete!"
echo "================================================"
echo "Results directory: $OUTPUT_DIR"
echo "  Case 1 (fixed GEMV, vary GEMM): $CASE1_CSV"
echo "  Case 2 (fixed GEMM, vary GEMV): $CASE2_CSV"
echo ""
echo "To plot results, run:"
echo "  python3 plot_size_sweep.py --input_dir $OUTPUT_DIR"

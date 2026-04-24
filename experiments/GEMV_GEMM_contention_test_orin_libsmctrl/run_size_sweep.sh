#!/bin/bash

set -euo pipefail

NUM_SMS=""
NUM_ITERS=8
REPEATS=3
OUTPUT_DIR="size_sweep_results"

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
            echo "  --num_sms <N>       Requested SM count"
            echo "  --num_iters <N>     Iterations per run"
            echo "  --repeats <N>       Measurement repeats"
            echo "  --output_dir <dir>  Output directory"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

mkdir -p "$OUTPUT_DIR"

if [[ -z "$NUM_SMS" ]]; then
    SM_ARG=""
    echo "Using all available SMs (TPC-aligned sweep)"
else
    SM_ARG="--min_sms $NUM_SMS --max_sms $NUM_SMS"
    echo "Using requested SM count $NUM_SMS (rounded to TPC granularity)"
fi

EXECUTABLE="./build/GEMV_GEMM_contention_test_orin_libsmctrl"

if [[ ! -x "$EXECUTABLE" ]]; then
    echo "Executable not found at $EXECUTABLE"
    exit 1
fi

CASE1_CSV="$OUTPUT_DIR/case1_fixed_gemv_vary_gemm.csv"
CASE2_CSV="$OUTPUT_DIR/case2_fixed_gemm_vary_gemv.csv"

echo "num_sms,tpc_count,sms_per_tpc,gemv_M,gemv_N,gemm_size,mode,gemv_gflops,gemm_gflops,gemv_time_ms,gemm_time_ms,gemv_slowdown,gemm_slowdown,overlap_pct" > "$CASE1_CSV"
echo "num_sms,tpc_count,sms_per_tpc,gemv_M,gemv_N,gemm_size,mode,gemv_gflops,gemm_gflops,gemv_time_ms,gemm_time_ms,gemv_slowdown,gemm_slowdown,overlap_pct" > "$CASE2_CSV"

FIXED_GEMV_M=8192
FIXED_GEMV_N=8192
FIXED_GEMM_SIZE=1024

GEMM_SIZES=(512 768 1024 1280 1536)
GEMV_SIZES=(4096 6144 8192 10240 12288)

for GEMM_SIZE in "${GEMM_SIZES[@]}"; do
    TEMP_CSV="${OUTPUT_DIR}/temp_case1_${GEMM_SIZE}.csv"
    echo "Running case 1 with GEMM=${GEMM_SIZE}"
    $EXECUTABLE $SM_ARG --gemv_M $FIXED_GEMV_M --gemv_N $FIXED_GEMV_N --gemm_size $GEMM_SIZE \
        --num_iters $NUM_ITERS --repeats $REPEATS --csv "$TEMP_CSV"
    tail -n +2 "$TEMP_CSV" >> "$CASE1_CSV"
    rm -f "$TEMP_CSV"
done

for GEMV_SIZE in "${GEMV_SIZES[@]}"; do
    TEMP_CSV="${OUTPUT_DIR}/temp_case2_${GEMV_SIZE}.csv"
    echo "Running case 2 with GEMV=${GEMV_SIZE}"
    $EXECUTABLE $SM_ARG --gemv_M $GEMV_SIZE --gemv_N $GEMV_SIZE --gemm_size $FIXED_GEMM_SIZE \
        --num_iters $NUM_ITERS --repeats $REPEATS --csv "$TEMP_CSV"
    tail -n +2 "$TEMP_CSV" >> "$CASE2_CSV"
    rm -f "$TEMP_CSV"
done

echo "Results saved to $OUTPUT_DIR"

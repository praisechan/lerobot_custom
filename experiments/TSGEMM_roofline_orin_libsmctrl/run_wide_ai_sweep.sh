#!/bin/bash

set -euo pipefail

NUM_SMS=""
REPEATS=1
REAL_ITERS=2
COMPUTE_REPEATS=2048
OUTPUT_DIR="wide_ai_sweep_results"
DIMS=(16 32 64 128 256 512 1024 2048 3072 4096)

while [[ $# -gt 0 ]]; do
    case $1 in
        --num_sms)
            NUM_SMS="$2"
            shift 2
            ;;
        --repeats)
            REPEATS="$2"
            shift 2
            ;;
        --real_iters)
            REAL_ITERS="$2"
            shift 2
            ;;
        --compute_repeats)
            COMPUTE_REPEATS="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --dims)
            IFS=',' read -r -a DIMS <<< "$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "  --num_sms <N>            Requested SM count"
            echo "  --repeats <N>            Measurement repeats"
            echo "  --real_iters <N>         Real kernel iterations"
            echo "  --compute_repeats <N>    Compute roof repeats"
            echo "  --output_dir <dir>       Output directory"
            echo "  --dims a,b,c             Comma-separated square GEMM dimensions"
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

EXECUTABLE="./build/TSGEMM_roofline_orin_libsmctrl"
if [[ ! -x "$EXECUTABLE" ]]; then
    echo "Executable not found at $EXECUTABLE"
    exit 1
fi

SUMMARY_CSV="$OUTPUT_DIR/summary.csv"
echo "M,K,N,peak_bandwidth_GBps,required_full_bw_GFLOPS,min_sms_compute_for_full_bw,max_real_GFLOPS,max_predicted_GFLOPS,arithmetic_intensity" > "$SUMMARY_CSV"

for DIM in "${DIMS[@]}"; do
    CSV_PATH="$OUTPUT_DIR/shape_${DIM}.csv"
    echo "================================================"
    echo "Running synthetic wide-AI sweep point with M=K=N=${DIM}"
    echo "================================================"
    $EXECUTABLE $SM_ARG \
        --row_size "$DIM" \
        --K "$DIM" \
        --N "$DIM" \
        --real_iters "$REAL_ITERS" \
        --compute_repeats "$COMPUTE_REPEATS" \
        --repeats "$REPEATS" \
        --csv "$CSV_PATH"

    python3 - "$CSV_PATH" "$SUMMARY_CSV" "$DIM" <<'PY'
import pandas as pd
import sys

csv_path, summary_path, dim = sys.argv[1:]
df = pd.read_csv(csv_path)
peak_bw = df["bandwidth_roof_GBps"].max()
required = df["required_full_bw_GFLOPS"].iloc[0]
compute_ok = df[df["compute_roof_GFLOPS"] >= required]
min_sms = int(compute_ok["num_sms"].iloc[0]) if not compute_ok.empty else -1
max_real = df["real_GFLOPS"].max()
max_pred = df["predicted_GFLOPS"].max()
ai = df["arithmetic_intensity_flops_per_byte"].iloc[0]
with open(summary_path, "a", encoding="utf-8") as f:
    f.write(f"{dim},{dim},{dim},{peak_bw:.6f},{required:.6f},{min_sms},{max_real:.6f},{max_pred:.6f},{ai:.6f}\n")
PY
done

echo "Results saved to $OUTPUT_DIR"

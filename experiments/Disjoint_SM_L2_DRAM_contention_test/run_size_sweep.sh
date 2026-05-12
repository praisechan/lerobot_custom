#!/usr/bin/env bash

set -euo pipefail

MEM_SIZES="16 48 128 512 1024"
COMPUTE_SIZES="512 1024 1536 2048"
MEM_SMS=8
COMPUTE_SMS=8
ITERATIONS=3
REPEATS=3
MMA_REPEATS=8
OUTPUT_DIR="size_sweep_results"
FLUSH_L2=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mem_sizes)
            MEM_SIZES="$2"
            shift 2
            ;;
        --compute_sizes)
            COMPUTE_SIZES="$2"
            shift 2
            ;;
        --mem_sms)
            MEM_SMS="$2"
            shift 2
            ;;
        --compute_sms)
            COMPUTE_SMS="$2"
            shift 2
            ;;
        --iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        --repeats)
            REPEATS="$2"
            shift 2
            ;;
        --mma_repeats)
            MMA_REPEATS="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --flush_l2)
            FLUSH_L2=1
            shift
            ;;
        -h|--help)
            cat <<EOF
Usage: $0 [options]

Options:
  --mem_sizes "16 48 128"       Working-set MiB values to sweep
  --compute_sizes "512 1536"    WMMA square matrix sizes to sweep
  --mem_sms N                   Memory Green Context SM count (default: 8)
  --compute_sms N               Compute Green Context SM count (default: 8)
  --iterations N                Timed launches per repeat (default: 3)
  --repeats N                   Repeats per configuration (default: 3)
  --mma_repeats N               WMMA repeats per loaded tile (default: 8)
  --output_dir PATH             Output directory (default: size_sweep_results)
  --flush_l2                    Enable benchmark L2 flush before timed launches
EOF
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXECUTABLE="$SCRIPT_DIR/build/Disjoint_SM_L2_DRAM_contention_test"

if [[ ! -x "$EXECUTABLE" ]]; then
    echo "Error: executable not found at $EXECUTABLE"
    echo "Build first:"
    echo "  cmake -S $SCRIPT_DIR -B $SCRIPT_DIR/build && cmake --build $SCRIPT_DIR/build -j"
    exit 1
fi

mkdir -p "$OUTPUT_DIR/raw"
SUMMARY_CSV="$OUTPUT_DIR/size_sweep_summary.csv"
echo "mem_mib,compute_size,mem_sms,compute_sms,iterations,repeats,mma_repeats,flush_l2,mem_bandwidth_isolated_gib_s,mem_bandwidth_concurrent_gib_s,compute_isolated_tflops,compute_concurrent_tflops,mem_time_isolated_ms,mem_time_concurrent_ms,compute_time_isolated_ms,compute_time_concurrent_ms,wall_time_ms,mem_slowdown,compute_slowdown,overlap_pct" > "$SUMMARY_CSV"

FLUSH_ARG=()
if [[ "$FLUSH_L2" == "1" ]]; then
    FLUSH_ARG=(--flush_l2)
fi

echo "============================================================="
echo "Disjoint SM L2/DRAM size sweep"
echo "============================================================="
echo "Memory working sets (MiB): $MEM_SIZES"
echo "Compute sizes:             $COMPUTE_SIZES"
echo "SMs memory/compute:         $MEM_SMS / $COMPUTE_SMS"
echo "Iterations/repeats:         $ITERATIONS / $REPEATS"
echo "Output:                     $OUTPUT_DIR"
echo "============================================================="

for MEM_MIB in $MEM_SIZES; do
    for COMPUTE_SIZE in $COMPUTE_SIZES; do
        RUN_CSV="$OUTPUT_DIR/raw/mem_${MEM_MIB}_compute_${COMPUTE_SIZE}.csv"
        echo
        echo "Running mem_mib=$MEM_MIB compute_size=$COMPUTE_SIZE"
        "$EXECUTABLE" \
            --mem_sms "$MEM_SMS" \
            --compute_sms "$COMPUTE_SMS" \
            --mem_mib "$MEM_MIB" \
            --compute_size "$COMPUTE_SIZE" \
            --iterations "$ITERATIONS" \
            --repeats "$REPEATS" \
            --mma_repeats "$MMA_REPEATS" \
            "${FLUSH_ARG[@]}" \
            --csv "$RUN_CSV" | grep -E "^(Repeat|  \\[|Memory isolated|Compute isolated|Memory concurrent|Compute concurrent|Concurrent wall|Results written|Actual streaming)"

        python3 - "$RUN_CSV" "$SUMMARY_CSV" <<'PY'
import csv
import sys

run_csv, summary_csv = sys.argv[1], sys.argv[2]
with open(run_csv, newline="") as f:
    rows = list(csv.DictReader(f))

by_mode = {row["mode"]: row for row in rows}
mem = by_mode["isolated_memory"]
compute = by_mode["isolated_compute"]
conc = by_mode["concurrent"]

out = {
    "mem_mib": conc["mem_working_set_mib"],
    "compute_size": conc["compute_size"],
    "mem_sms": conc["mem_sms"],
    "compute_sms": conc["compute_sms"],
    "iterations": conc["iterations"],
    "repeats": conc["repeats"],
    "mma_repeats": conc["mma_repeats"],
    "flush_l2": conc["flush_l2"],
    "mem_bandwidth_isolated_gib_s": mem["mem_bandwidth_gib_s"],
    "mem_bandwidth_concurrent_gib_s": conc["mem_bandwidth_gib_s"],
    "compute_isolated_tflops": compute["compute_tflops"],
    "compute_concurrent_tflops": conc["compute_tflops"],
    "mem_time_isolated_ms": mem["mem_time_ms"],
    "mem_time_concurrent_ms": conc["mem_time_ms"],
    "compute_time_isolated_ms": compute["compute_time_ms"],
    "compute_time_concurrent_ms": conc["compute_time_ms"],
    "wall_time_ms": conc["wall_time_ms"],
    "mem_slowdown": conc["mem_slowdown"],
    "compute_slowdown": conc["compute_slowdown"],
    "overlap_pct": conc["overlap_pct"],
}

with open(summary_csv, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(out.keys()))
    writer.writerow(out)
PY
    done
done

echo
echo "Sweep complete: $SUMMARY_CSV"
echo "Plot with:"
echo "  python3 $SCRIPT_DIR/tools/plot_size_sweep.py $SUMMARY_CSV $OUTPUT_DIR"

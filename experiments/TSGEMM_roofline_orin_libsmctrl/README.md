# TS-GEMM Roofline Experiment for Jetson AGX Orin

This experiment implements the recommended three-part methodology for studying a tall-skinny GEMM on Orin:

1. **Bandwidth roof**: DRAM streaming kernel under `libsmctrl` masking
2. **Compute roof**: tensor-core WMMA microkernel with shared-memory-resident tiles
3. **Real TS-GEMM**: `A[M x K] * B[K x N] -> C[M x N]`, with varying `M`

The key output is the minimum SM count whose **compute roof** can sustain the compute demand implied by a **full-bandwidth** TS-GEMM stream:

`required_full_bw_GFLOPS = arithmetic_intensity * peak_bandwidth_GBps`

## Default Shape

- `M`: varies
- `K = 4096`
- `N = 16`
- FP16 inputs/outputs
- WMMA tile size: `16 x 16 x 16`

`N=16` keeps the GEMM genuinely tall-and-skinny and low-arithmetic-intensity relative to square GEMMs.

## Build

```bash
cd ~/lerobot_custom/experiments/TSGEMM_roofline_orin_libsmctrl
mkdir -p build
cd build
cmake ..
cmake --build . -j
```

## Single Run

```bash
./TSGEMM_roofline_orin_libsmctrl \
    --min_sms 2 --max_sms 16 --step_sms 2 \
    --row_size 8192 --K 4096 --N 16 \
    --real_iters 8 --compute_repeats 4096 --repeats 5 \
    --csv ../results.csv
```

Then plot:

```bash
python3 tools/plot.py --csv results.csv
```

## Row Sweep

```bash
./run_row_sweep.sh --num_sms 16
python3 plot_row_sweep.py --input_dir row_sweep_results
python3 plot_roofline_by_sm.py --input_dir row_sweep_results
```

`plot_roofline_by_sm.py` generates the figure with:

- X-axis: arithmetic intensity
- Y-axis: throughput
- one dedicated line per SM count (`2, 4, 6, ...`)

By default it plots both the predicted roofline line and the real TS-GEMM points for each SM count.

## Skinny-A / Fat-B N Sweep

If you want the more natural roofline spread for a skinny `A[M x K]` and fat `B[K x N]`, keep `M` fixed and sweep `N`:

```bash
./run_n_sweep.sh --M 16 --K 4096
python3 plot_n_sweep.py --input_dir n_sweep_results
python3 plot_roofline_by_sm.py --input_dir n_sweep_results --glob 'n_*.csv'
```

## Synthetic Wide-AI Sweep

To span arithmetic intensity from single digits into the thousands, use a separate synthetic square-GEMM family:

```bash
./run_wide_ai_sweep.sh
python3 plot_wide_ai_sweep.py --input_dir wide_ai_sweep_results
python3 plot_roofline_by_sm.py --input_dir wide_ai_sweep_results --glob 'shape_*.csv'
```

This sweep is for broad roofline coverage only. It is not meant to represent the skinny-`A`, fat-`B` workload directly.

## Output CSV

Each row contains:

- `bandwidth_roof_GBps`
- `compute_roof_GFLOPS`
- `memory_roof_GFLOPS = AI * bandwidth_roof_GBps`
- `predicted_GFLOPS = min(compute_roof, memory_roof)`
- `required_full_bw_GFLOPS = AI * peak_bandwidth_GBps`
- `real_GFLOPS`
- `implied_real_GBps = real_GFLOPS / AI`

These let you distinguish:

- **compute-limited**: `compute_roof < memory_roof`
- **memory-limited**: `memory_roof < compute_roof`
- **tracking quality**: `real_GFLOPS / predicted_GFLOPS`

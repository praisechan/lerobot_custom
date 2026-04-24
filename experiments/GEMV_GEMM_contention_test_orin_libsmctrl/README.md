# GEMV-GEMM Contention Test for Jetson AGX Orin

This experiment is an Orin-friendly port of `experiments/GEMV_GEMM_contention_test`.

The original benchmark depends on:

- CUDA Green Contexts from CUDA 13.x
- Blackwell/Hopper-oriented SM partitioning APIs

Jetson AGX Orin is a CUDA 12.6, Ampere-class system, so this version replaces Green Context partitioning with `libsmctrl` stream-launch masking. The benchmark still measures the same core question: how FP16 GEMV and FP16 WMMA GEMM interfere when both are forced onto the same limited subset of GPU compute resources.

## What Changed

- **Partitioning backend**: `libsmctrl` instead of Green Contexts
- **Sweep granularity**: TPC-aligned instead of arbitrary SM-aligned
- **Defaults**: reduced to sizes that are more practical on Orin
- **CSV shape**: keeps the original columns and adds `tpc_count` / `sms_per_tpc`

On Orin, `libsmctrl` masks at the **TPC** level. If your GPU has 2 SMs per TPC, a request like `--min_sms 3 --max_sms 9` is rounded to valid masked points such as 4, 6, 8 SMs.

## Build

```bash
cd ~/lerobot_custom/experiments/GEMV_GEMM_contention_test_orin_libsmctrl
mkdir -p build
cd build
cmake ..
cmake --build . -j
```

## Run

```bash
./GEMV_GEMM_contention_test_orin_libsmctrl
```

Example with a small sweep:

```bash
./GEMV_GEMM_contention_test_orin_libsmctrl \
    --min_sms 2 \
    --max_sms 8 \
    --step_sms 2 \
    --gemv_M 4096 \
    --gemv_N 4096 \
    --gemm_size 512 \
    --num_iters 2 \
    --repeats 2 \
    --csv ./results_smoke.csv
```

## Options

| Option | Description | Default |
| --- | --- | --- |
| `--min_sms <N>` | Minimum requested SM count | 2 |
| `--max_sms <N>` | Maximum requested SM count | all SMs |
| `--step_sms <N>` | Requested SM step | SMs per TPC |
| `--gemv_M <N>` | GEMV rows | 8192 |
| `--gemv_N <N>` | GEMV cols | 8192 |
| `--gemm_size <N>` | GEMM dimension | 1024 |
| `--num_iters <N>` | Launches per measurement | 8 |
| `--repeats <N>` | Repeats per sweep point | 5 |
| `--perfect_sync` | Device-sync before paired launches | off |
| `--csv <path>` | Output CSV | `./results.csv` |

## Output

The executable writes rows for:

- `isolated_gemv`
- `isolated_gemm`
- `concurrent`

matching the original benchmark so the copied plotting scripts keep working.

## Plotting

```bash
python3 tools/plot.py ./results.csv
```

For the size sweep helper:

```bash
./run_size_sweep.sh --num_sms 8
python3 plot_size_sweep.py --input_dir size_sweep_results
```

# SM-limited Read Bandwidth Sweep

This microbenchmark sweeps SM counts and measures DRAM read bandwidth using CUDA Green Contexts (execution affinity). If Green Contexts are not supported by the installed CUDA headers/driver, it falls back to the primary context and still performs the sweep by varying grid size so you can observe scaling.

## Build

```bash
cd ~/lerobot_custom/experiments/simple_sm_util_test
mkdir -p build
cd build
cmake ..
cmake --build . -j
```

## Run

```bash
./sm_bw_sweep --min_sms 1 --max_sms 120 --step 1 --bytes 1073741824 --repeats 5 --csv ./results.csv
```

## Output

The CSV contains:
- `sm_count`: requested SM count
- `green_used`: 1 if Green Contexts were used, 0 otherwise
- `time_ms_mean`, `time_ms_std`: kernel timing stats
- `total_bytes_read`: bytes read per measurement
- `read_GBps_mean`, `read_GBps_std`: achieved read bandwidth
- `checksum`: simple 64-bit checksum to prevent dead-code elimination
- `tpb`, `vec_bytes`, `unroll`, `blocks`, `iters`: launch configuration

## Design Notes

- The kernel is a read-only streaming load with vectorized 16B loads and loop unrolling.
- Loads are accumulated into a 64-bit checksum and only one 64-bit value per thread is written to a small `sink` buffer. This write traffic is negligible compared to read traffic.
- The default `--bytes` value is 1 GiB to avoid cache residency (L2 is much smaller). Set it higher for larger working sets.
- `-Xptxas -dlcm=cg` is enabled to discourage L1 caching and reflect DRAM+L2 behavior.
- Green Contexts currently accept an SM count but do not allow explicit SM ID selection. The subset of SMs is implementation-defined; for analysis, treat it as a fixed subset for each `sm_count`.

## Plot

```bash
python3 tools/plot.py --csv ./results.csv --out ./bw_vs_sm.png
```

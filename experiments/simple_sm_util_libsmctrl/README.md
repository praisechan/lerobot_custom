# SM-Limited DRAM Read Bandwidth Sweep

This microbenchmark measures DRAM read bandwidth as a function of active SM/TPC count using libsmctrl for hardware compute partitioning.

## Purpose

Empirically determine the **minimum number of SMs required to saturate GPU DRAM read bandwidth** on your system by restricting kernel execution to progressively larger subsets of SMs.

## How It Works

1. **SM/TPC Masking**: Uses [libsmctrl](../../3rdparty/BulletServe/csrc/) to restrict kernel execution to specific TPC subsets
2. **Read-Only Kernel**: Streams data from global memory with vectorized loads (float4 = 16B)
3. **Bandwidth Measurement**: Times with CUDA events and computes achieved read bandwidth
4. **Statistical Analysis**: Repeats each configuration multiple times, reports mean and standard deviation

## Kernel Design

The benchmark kernel is designed to measure **pure read bandwidth** with minimal write traffic:

- **Vectorized Loads**: Each thread loads `float4` (16 bytes) per access
- **Loop Unrolling**: 4x unroll for instruction-level parallelism
- **Streaming Pattern**: Each thread reads a unique stride through memory (no cache reuse)
- **Checksum Accumulation**: XOR-based mixing prevents dead-code elimination
- **Minimal Writes**: One 8-byte write per block (negligible vs. GiB of reads)

### Dead-Code Elimination Prevention

The kernel accumulates loaded values into a register checksum and writes one `uint64_t` per block to a sink buffer. With typical configurations:
- **Reads**: ~1 GiB total (default `--bytes 1073741824`)
- **Writes**: ~8 KB (1024 blocks × 8 bytes)
- **Write Ratio**: ~0.0008% of read traffic (negligible)

### Memory Access Pattern

- **Working Set**: Default 1 GiB (exceeds L2 cache on all modern GPUs)  
- **Access Pattern**: Stride = total threads, ensuring each thread reads unique addresses
- **L1 Caching**: Disabled via `-Xptxas -dlcm=cg` to reflect DRAM+L2 behavior

## Build

```bash
cd ~/lerobot_custom/experiments/simple_sm_util_test
mkdir -p build
cd build
cmake ..
cmake --build . -j
```

**Requirements:**
- CUDA 11.0+ (for CMake CUDA support)
- libsmctrl built from `../../3rdparty/BulletServe/csrc/` (automatically handled by CMake)
- GPU with compute capability ≥ 3.5

## Usage

### Basic Sweep (Default)

Sweep from 1 TPC to maximum available, step 1:

```bash
./sm_bw_sweep
```

### Custom Range

Sweep a specific TPC range:

```bash
./sm_bw_sweep --min_sms 1 --max_sms 120 --step 5
```

### Large Working Set

Use 4 GiB working set instead of 1 GiB:

```bash
./sm_bw_sweep --bytes 4294967296
```

### Full Parameter Example

```bash
./sm_bw_sweep \
    --min_sms 1 \
    --max_sms 120 \
    --step 1 \
    --bytes 1073741824 \
    --tpb 256 \
    --repeats 10 \
    --csv ./my_results.csv
```

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--min_sms N` | Minimum TPC count | 1 |
| `--max_sms N` | Maximum TPC count | All available |
| `--step N` | Step size for sweep | 1 |
| `--bytes N` | Total bytes read per measurement | 1073741824 (1 GiB) |
| `--tpb N` | Threads per block | 256 |
| `--iters N` | Iterations per thread (overrides auto-calc) | Auto |
| `--repeats N` | Measurements per config (for statistics) | 5 |
| `--csv PATH` | Output CSV file path | `./results.csv` |

## Output

### CSV Format

The output CSV contains the following columns:

| Column | Description |
|--------|-------------|
| `tpc_count` | Number of TPCs enabled for this measurement |
| `time_ms_mean` | Mean kernel execution time (ms) |
| `time_ms_std` | Standard deviation of execution time (ms) |
| `total_bytes_read` | Total bytes read in each measurement |
| `read_GBps_mean` | Mean achieved read bandwidth (GB/s) |
| `read_GBps_std` | Standard deviation of bandwidth (GB/s) |
| `checksum` | 64-bit checksum (sanity check for correctness) |
| `tpb` | Threads per block used |
| `vec_bytes` | Vector width in bytes (16 for float4) |
| `unroll` | Loop unroll factor (4) |
| `blocks` | Number of blocks launched |
| `iters` | Iterations per thread |

### Example Output

```
Device: NVIDIA H100
Total SMs: 132
Total TPCs: 66
SMs per TPC: 2

Configuration:
  SM/TPC range: 1 to 66 (step 1)
  Total bytes per measurement: 1073741824 (1.00 GiB)
  Threads per block: 256
  Repeats per config: 5
  Output CSV: ./results.csv

Starting sweep...
TPC_Count   Time(ms)        Read_BW(GB/s)   Checksum
-------------------------------------------------------------
1           125.342         8.56            0x1a2b3c4d5e6f7890
2           62.891          17.08           0x1a2b3c4d5e6f7890
4           31.672          33.92           0x1a2b3c4d5e6f7890
...
64          4.234           253.68          0x1a2b3c4d5e6f7890
66          4.201           255.52          0x1a2b3c4d5e6f7890

Sweep complete! Results written to ./results.csv
```

## Visualization

Plot bandwidth vs. TPC count:

```bash
python3 tools/plot.py --csv ./results.csv --out ./bandwidth_vs_tpc.png
```

This generates a line plot showing:
- X-axis: TPC count
- Y-axis: Read bandwidth (GB/s)
- Shaded region: ±1 standard deviation

**Expected Result**: Bandwidth increases linearly with TPC count until memory subsystem saturates, then plateaus.

## Interpretation

### Finding Saturation Point

The "knee" in the bandwidth curve indicates the minimum TPC count needed to saturate DRAM bandwidth:

1. **Below saturation**: Bandwidth scales linearly (compute-limited)
2. **At saturation**: Bandwidth plateaus (memory-limited)
3. **Saturation point**: Minimum TPCs for max bandwidth

Example: If bandwidth plateaus at 30 TPCs, then 30 TPCs (60 SMs on H100) saturate your ~3.35 TB/s HBM3.

### Factors Affecting Results

- **Memory Clock**: HBM frequency affects peak bandwidth
- **ECC**: Error correction reduces effective bandwidth
- **Thermal Throttling**: Can reduce bandwidth under sustained load
- **Background Activity**: Other processes using GPU memory
- **L2 Cache Size**: Working set must exceed L2 to reflect DRAM bandwidth

## Design Notes

### Why TPC Masking?

libsmctrl operates at **TPC granularity** (not individual SMs) because:
- Modern GPUs: 2 SMs per TPC
- Hardware partitioning is TPC-level in NVIDIA's QMD/TMD structures
- Masking 1 TPC disables both of its SMs

### Launch Configuration

For each TPC count, we launch:
- **Blocks**: `active_sms × 4` (4 blocks per SM for occupancy)
- **Threads/Block**: 256 (configurable via `--tpb`)
- **Iterations**: Auto-calculated to read `--bytes` total

This ensures sufficient work to saturate the enabled SMs.

### Why Default 1 GiB?

- Exceeds L2 cache on all modern GPUs (e.g., H100 has 50 MB L2)
- Ensures measurements reflect DRAM bandwidth, not L2 bandwidth
- Amortizes kernel launch overhead
- Large enough for stable timing measurements

For GPUs with very large L2, consider using `--bytes 4294967296` (4 GiB).

### Timing Methodology

- **Warm-up**: One untimed launch per configuration
- **Measurement**: CUDA events (`cudaEventRecord`) for GPU-side timing
- **Repeat**: Default 5 measurements per config for statistical validity
- **Output**: Mean and standard deviation of both time and bandwidth

## Hardware Compatibility

### Supported GPUs

All GPUs with **compute capability ≥ 3.5** (Kepler V2+):
- **Fully Tested**: Kepler, Maxwell, Pascal, Volta, Turing, Ampere
- **Experimental**: Ada Lovelace, Hopper (H100)
- **Note**: Blackwell (GB10, GB100) not yet supported by libsmctrl

### Not Supported

- Compute capability < 3.5 (Fermi, Kepler V1)
- **Blackwell GPUs**: GB10, GB100 require Green Contexts API instead

### Blackwell Users

If you have a Blackwell GPU, you need to use **NVIDIA Green Contexts** instead of libsmctrl:
- Requires CUDA 13.1+ with Runtime API
- See `simple_sm_util_test_old` for Green Contexts reference implementation
- Or wait for updated libsmctrl with Blackwell support

Check compatibility:
```bash
./sm_bw_sweep
# Will report error if GPU doesn't support SM masking
```

## Troubleshooting

### "libsmctrl_get_tpc_info_cuda failed"

Your GPU doesn't support SM masking (compute capability < 3.5). Upgrade to a newer GPU.

### Bandwidth lower than expected

1. **Check thermal throttling**: `nvidia-smi` shows GPU temperature and clocks
2. **Verify memory clock**: Memory clock may be throttled
3. **Check ECC status**: ECC reduces effective bandwidth by ~20%
4. **Increase working set**: Use `--bytes 4294967296` if L2 is very large
5. **Verify exclusive GPU access**: Stop other GPU processes

### Build failures

1. **CUDA not found**: Ensure CUDA toolkit is installed and `nvcc` is in PATH
2. **libsmctrl missing**: Check that `../../3rdparty/BulletServe/csrc/` exists
3. **Architecture mismatch**: CMake uses `native` detection; set manually if needed:
   ```cmake
   set(CMAKE_CUDA_ARCHITECTURES 80)  # For A100
   ```

## Comparison with Green Contexts

This benchmark uses **libsmctrl** instead of NVIDIA Green Contexts because:

| Feature | libsmctrl | Green Contexts |
|---------|-----------|----------------|
| **Hardware Support** | GPUs since 2012 (sm_35+) | H100+, Blackwell+ only |
| **CUDA Version** | 6.5 - 12.8 | 11.4+ (API), 13.1+ (Runtime) |
| **Complexity** | Simple API | More complex setup |
| **Portability** | High (works on older GPUs) | Low (newest hardware only) |

**Recommendation**: Use libsmctrl for broad compatibility. Switch to Green Contexts for H100+ with CUDA 13.1+.

## Related Tools

- **[LIBSMCTRL_REFERENCE.md](./LIBSMCTRL_REFERENCE.md)**: Complete libsmctrl API documentation
- **[libsmctrl paper](https://www.cs.unc.edu/~jbakita/rtas23.pdf)**: Academic paper on hardware compute partitioning

## Citation

If you use this benchmark in research, please cite:

```bibtex
@inproceedings{bakita2023hardware,
  title={Hardware Compute Partitioning on {NVIDIA} {GPUs}},
  author={Bakita, Joshua and Anderson, James H},
  booktitle={Proceedings of the 29th IEEE Real-Time and Embedded Technology and Applications Symposium},
  pages={54--66},
  year={2023}
}
```

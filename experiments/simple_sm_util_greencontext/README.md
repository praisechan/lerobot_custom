# SM Bandwidth Sweep with CUDA Green Contexts

A microbenchmark to empirically determine the minimum number of SMs required to saturate GPU DRAM read bandwidth using CUDA Green Contexts.

## Overview

This benchmark measures DRAM read bandwidth while varying the number of Streaming Multiprocessors (SMs) available for kernel execution. By using **CUDA Green Contexts** (introduced in CUDA 13.0), we can partition GPU resources and restrict kernel execution to specific SM subsets.

### Key Features

- **Green Context-based SM partitioning**: Programmatically restrict kernels to N SMs
- **Read-bandwidth kernel**: Streaming vectorized loads with minimal sink writes
- **Anti-DCE measures**: Checksum accumulation + tiny sink output per block
- **Statistical measurement**: Multiple repeats with mean and standard deviation
- **CSV output**: Results ready for plotting and analysis

## Requirements

- **CUDA Toolkit**: 13.0 or later (for Green Contexts support)
- **Compute Capability**: 6.0+ (Pascal or later)
  - Hopper (9.0+): Min 8 SMs, alignment 8
  - Ampere (8.x): Min 4 SMs, alignment 2
  - Volta/Turing (7.x): Min 2 SMs, alignment 2
  - Pascal (6.x): Min 2 SMs, alignment 2
- **Platform**: 64-bit Linux (Green Contexts not supported on 32-bit)
- **CMake**: 3.18+
- **Python 3**: For plotting (matplotlib, pandas)

## Build Instructions

```bash
cd ~/lerobot_custom/experiments/simple_sm_util_greencontext

# Create build directory
mkdir build && cd build

# Configure
cmake ..

# Build
cmake --build . -j

# Binary location: ./sm_bw_sweep
```

## Usage

### Basic Usage

```bash
# Default: sweep all SMs, 1 GiB workload
./sm_bw_sweep

# Custom SM range
./sm_bw_sweep --min_sms 8 --max_sms 64

# Custom workload size
./sm_bw_sweep --bytes 2147483648  # 2 GiB

# More repeats for stability
./sm_bw_sweep --repeats 10

# Custom output
./sm_bw_sweep --csv my_results.csv
```

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--min_sms <N>` | Minimum SM count | 8 |
| `--max_sms <N>` | Maximum SM count | Device max |
| `--bytes <N>` | Total bytes to read per measurement | 1073741824 (1 GiB) |
| `--tpb <N>` | Threads per block | 256 |
| `--iters <N>` | Override iterations (auto-calculated by default) | auto |
| `--repeats <N>` | Number of measurement repeats | 5 |
| `--csv <path>` | Output CSV path | ./results.csv |
| `--help` | Print usage | - |

### Example Output

```
=============================================================
SM Bandwidth Sweep using CUDA Green Contexts
=============================================================
Device: NVIDIA H100
Compute Capability: 9.0
Total SMs: 132
Configuration:
  SM Range: 8 - 132
  Bytes per measurement: 1073741824 (1.00 GiB)
  Threads per block: 256
  Repeats: 5
  CSV output: ./results.csv
=============================================================

Device SM Resource Info:
  Total SMs: 132
  Min SM Partition Size: 8
  SM Co-scheduled Alignment: 8

Testing 17 SM configurations...

Testing with 8 SMs... 245.32 ± 2.15 GB/s
Testing with 16 SMs... 487.61 ± 3.42 GB/s
Testing with 24 SMs... 723.45 ± 4.21 GB/s
Testing with 32 SMs... 955.23 ± 5.67 GB/s
Testing with 40 SMs... 1123.45 ± 6.32 GB/s
Testing with 48 SMs... 1245.67 ± 7.11 GB/s
Testing with 56 SMs... 1298.45 ± 6.89 GB/s  <- Saturation point
Testing with 64 SMs... 1302.34 ± 7.23 GB/s
...
```

## Kernel Design

### Read Bandwidth Kernel

The kernel is designed to maximize DRAM read throughput while minimizing write traffic:

```cuda
__global__ void bandwidth_read_kernel(
    const Vec4U32* input,    // 16B vectorized loads
    uint64_t* sink,          // Minimal sink output
    size_t num_elements,
    int iters_per_thread
) {
    uint64_t checksum = 0;
    
    // Streaming reads with vectorized load (16B per iteration)
    for (int iter = 0; iter < iters_per_thread; ++iter) {
        Vec4U32 data = input[tid + iter * stride];
        
        // Accumulate to prevent DCE
        checksum ^= data.x + data.y + data.z + data.w;
    }
    
    // One write per block (negligible vs read traffic)
    if (threadIdx.x == 0) {
        sink[blockIdx.x] = reduce(checksum);
    }
}
```

### Anti-DCE Strategy

To prevent the compiler from eliminating "useless" reads:

1. **Checksum accumulation**: All loaded values contribute to a register checksum using XOR and addition
2. **Sink write**: Each block writes a single `uint64_t` to global memory
3. **Sink traffic ratio**: With 1 GiB reads and ~1000 blocks, sink writes are ~8 KB (0.0008% of read traffic)

### Access Pattern

- **Streaming**: Each thread reads consecutive memory with stride = vector width
- **Coalesced**: Threads in a warp access consecutive 16B chunks
- **No reuse**: Working set >> cache capacity, pure DRAM bandwidth test

## CSV Output Format

Generated CSV has the following columns:

```csv
num_sms,mean_bw_gb_s,stdev_bw_gb_s
8,245.32,2.15
16,487.61,3.42
24,723.45,4.21
...
```

## Plotting Results

```bash
# Generate bandwidth vs SM count plot
python3 tools/plot.py results.csv

# Output: results.png
```

The plot shows:
- X-axis: Number of SMs
- Y-axis: Read bandwidth (GB/s)
- Error bars: Standard deviation
- Saturation point: Where bandwidth plateaus

## Understanding Results

### Saturation Point

The **bandwidth saturation point** is the minimum SM count where bandwidth plateaus. Beyond this point, adding more SMs provides diminishing returns.

Example interpretation:
- If bandwidth plateaus at 56 SMs with 1300 GB/s
- Memory system can sustain ~1.3 TB/s read bandwidth
- 56 SMs are sufficient to saturate DRAM
- Additional SMs would benefit compute-bound workloads but not memory-bound ones

### Expected Behavior

- **Linear growth**: Bandwidth increases proportionally with SM count
- **Saturation**: Bandwidth plateaus when memory bandwidth is saturated
- **Variability**: Small standard deviation indicates stable measurements

### Architecture-Specific Notes

**Hopper (H100/H200)**:
- 80 GB HBM3 models: ~3 TB/s theoretical bandwidth
- Expect saturation at 40-60 SMs depending on access pattern

**Ampere (A100)**:
- 40/80 GB HBM2e: ~1.5-2 TB/s theoretical bandwidth
- Expect saturation at 30-50 SMs

## Caveats & Limitations

### Green Context Limitations

From [green_context_reference.md](docs/green_context_reference.md):

1. **No concurrency guarantee**: Even with disjoint SM partitions, kernels may not run truly concurrently due to other resource contention (HW connections)

2. **SM overflow possible**: In certain scenarios (MPS with `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`, or CDP on Compute 9.x), workload may use more SMs than provisioned

3. **Thread safety**: Green context can be current to only ONE thread at a time

### Measurement Considerations

1. **System noise**: Other processes may affect measurements; run on idle system
2. **Thermal throttling**: Extended runs may trigger throttling; monitor GPU temperature
3. **Cache effects**: Warm-up iterations help stabilize measurements
4. **PCIe/NVLink**: This benchmark measures on-device DRAM bandwidth only

## References

- [CUDA Green Contexts Documentation](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html)
- [Green Context Reference Guide](docs/green_context_reference.md)
- [CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/)

## License

See parent repository license.

## Author

Part of the lerobot_custom experimental suite.

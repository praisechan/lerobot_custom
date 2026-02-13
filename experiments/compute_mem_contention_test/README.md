# Compute-Memory Contention Test with CUDA Green Contexts

A microbenchmark to measure contention between compute-bound (GEMM) and memory-bound (mem copy) kernels when they execute concurrently on the same restricted SM subset using CUDA Green Contexts.

## Overview

This benchmark explores how compute-intensive and memory-intensive workloads interfere with each other when forced to share GPU resources. By using **CUDA Green Contexts** (introduced in CUDA 13.0), we can partition GPU resources and restrict both kernels to the same N SMs, measuring how performance degrades under contention compared to isolated execution.

### Key Features

- **Dual-kernel contention measurement**: GEMM (compute-bound) vs Memory Copy (memory-bound)
- **Three execution modes**: Isolated mem, isolated GEMM, and concurrent execution
- **Green Context-based resource sharing**: Both kernels share the same SM partition
- **Concurrent execution with proper synchronization**: Uses separate streams with event-based sync
- **Comprehensive metrics**: Bandwidth, throughput, slowdown ratios, overlap percentage
- **Statistical measurement**: Multiple repeats with averaging
- **Rich visualization**: Four plots analyzing different aspects of contention

### Why This Matters

Understanding compute-memory contention is critical for:

- **Multi-tenancy scenarios**: MPS (Multi-Process Service) and MIG (Multi-Instance GPU) contexts
- **Pipeline optimization**: Overlapping compute and memory operations in ML inference/training
- **Resource allocation**: Determining optimal SM partitioning for mixed workloads
- **Performance modeling**: Predicting interference effects in shared GPU environments

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
cd ~/lerobot_custom/experiments/compute_mem_contention_test

# Create build directory
mkdir build && cd build

# Configure
cmake ..

# Build
cmake --build . -j

# Binary location: ./compute_mem_contention_test
```

## Usage

### Basic Usage

```bash
# Default: sweep all SMs, 2 GiB mem workload, 2048x2048 GEMM
./compute_mem_contention_test

# Custom SM range
./compute_mem_contention_test --min_sms 8 --max_sms 64

# Larger GEMM workload (longer runtime for clearer contention)
./compute_mem_contention_test --gemm_size 4096

# More memory bandwidth stress
./compute_mem_contention_test --mem_bytes 4294967296  # 4 GiB

# More repeats for stability
./compute_mem_contention_test --repeats 10

# Custom output
./compute_mem_contention_test --csv my_results.csv
```

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--min_sms <N>` | Minimum SM count | 8 |
| `--max_sms <N>` | Maximum SM count | Device max |
| `--mem_bytes <N>` | Bytes for mem copy kernel | 2147483648 (2 GiB) |
| `--gemm_size <N>` | Matrix dimension for GEMM (M=N=K) | 2048 |
| `--tpb_mem <N>` | Threads per block for mem copy | 256 |
| `--tpb_gemm <N>` | Threads per block for GEMM | 256 (not used, fixed tile size) |
| `--repeats <N>` | Number of measurement repeats | 5 |
| `--csv <path>` | Output CSV path | ./results.csv |
| `--help` | Print usage | - |

### Example Output

```
=============================================================
Compute-Memory Contention Test using CUDA Green Contexts
=============================================================
Device: NVIDIA H100
Compute Capability: 9.0
Total SMs: 132
Configuration:
  SM Range: 8 - 132
  Mem Copy Bytes: 2147483648 (2.00 GiB)
  GEMM Size: 2048 x 2048 x 2048
  Expected GEMM FLOPs: 17.18 GFLOPs
  Threads per block (mem): 256
  Repeats: 5
  CSV output: ./results.csv
=============================================================

Device SM Resource Info:
  Total SMs: 132
  Min SM Partition Size: 8
  SM Co-scheduled Alignment: 8

Testing 17 SM configurations...

Testing with 8 SMs:
  [Mem Only]  BW: 89.45 GB/s, Time: 23.50 ms
  [GEMM Only] Throughput: 245.32 GFLOPS, Time: 70.12 ms
  [Concurrent] Wall Time: 82.34 ms (overlap: 11.9%)
               Mem: 58.23 GB/s (65.1% retained), Time: 36.21 ms, Slowdown: 1.54x
               GEMM: 167.89 GFLOPS (68.4% retained), Time: 102.45 ms, Slowdown: 1.46x

Testing with 16 SMs:
  [Mem Only]  BW: 178.92 GB/s, Time: 11.75 ms
  [GEMM Only] Throughput: 490.65 GFLOPS, Time: 35.06 ms
  [Concurrent] Wall Time: 41.23 ms (overlap: 11.9%)
               Mem: 145.67 GB/s (81.4% retained), Time: 14.45 ms, Slowdown: 1.23x
               GEMM: 398.21 GFLOPS (81.2% retained), Time: 43.18 ms, Slowdown: 1.23x

...

Results written to: ./results.csv

=============================================================
Benchmark Complete!
=============================================================
```

## Output Format

### CSV Structure

The benchmark generates a CSV file with the following columns:

| Column | Description |
|--------|-------------|
| `num_sms` | Number of SMs in the partition |
| `mode` | Execution mode: `isolated_mem`, `isolated_gemm`, `concurrent` |
| `mem_bw_gb_s` | Memory bandwidth (GB/s) |
| `gemm_gflops` | GEMM throughput (GFLOPS) |
| `mem_time_ms` | Memory kernel time (ms) |
| `gemm_time_ms` | GEMM kernel time (ms) |
| `mem_slowdown` | Memory slowdown vs isolated (concurrent only) |
| `gemm_slowdown` | GEMM slowdown vs isolated (concurrent only) |
| `overlap_pct` | Time overlap percentage (concurrent only) |

### Example CSV

```csv
num_sms,mode,mem_bw_gb_s,gemm_gflops,mem_time_ms,gemm_time_ms,mem_slowdown,gemm_slowdown,overlap_pct
8,isolated_mem,89.450,0.0,23.50,0.0,1.0,0.0,0.0
8,isolated_gemm,0.0,245.320,0.0,70.12,0.0,1.0,0.0
8,concurrent,58.230,167.890,36.21,102.45,1.54,1.46,11.9
16,isolated_mem,178.920,0.0,11.75,0.0,1.0,0.0,0.0
16,isolated_gemm,0.0,490.650,0.0,35.06,0.0,1.0,0.0
16,concurrent,145.670,398.210,14.45,43.18,1.23,1.23,11.9
```

## Visualization

Use the provided plotting script to generate four analysis plots:

```bash
cd tools
python3 plot.py ../build/results.csv
```

This generates:

1. **`results_performance_retention.png`**: Performance retention percentage vs SM count
   - Shows what fraction of isolated performance is retained during concurrent execution
   - Higher is better (100% = no degradation)

2. **`results_absolute_performance.png`**: Absolute bandwidth and throughput comparison
   - Dual Y-axis plot showing isolated vs concurrent performance
   - Visualizes the gap between isolated and concurrent performance

3. **`results_slowdown_ratios.png`**: Slowdown ratios vs SM count
   - Shows how much slower kernels run during contention
   - Closer to 1.0x is better

4. **`results_concurrent_efficiency.png`**: Time overlap percentage
   - Measures how much kernels actually execute concurrently
   - Higher values indicate better concurrent execution

### Interpreting Results

#### Performance Retention
- **>90%**: Minimal contention, kernels largely independent
- **70-90%**: Moderate contention, some resource competition
- **<70%**: Significant contention, substantial interference

#### Slowdown Ratio
- **1.0-1.2x**: Minimal impact from contention
- **1.2-1.5x**: Moderate impact, noticeable degradation
- **>1.5x**: Significant impact, substantial slowdown

#### Overlap Percentage
- **>60%**: Kernels execute mostly concurrently (good parallelism)
- **30-60%**: Partial concurrent execution (some serialization)
- **<30%**: Mostly sequential execution (poor concurrency)

### Expected Patterns

1. **Low SM Count (High Contention)**: 30-50% performance degradation, overlap ~50-70%
2. **Medium SM Count (Moderate Contention)**: 10-30% degradation, overlap ~60-80%
3. **High SM Count (Low Contention)**: <10% degradation, overlap ~70-85%

Note: Bandwidth contention may persist even at high SM counts since DRAM is shared across all SMs.

## Implementation Details

### Kernel Descriptions

#### Memory Copy Kernel (Memory-Bound)
- **Purpose**: Stress memory bandwidth with streaming reads
- **Design**: Vectorized 16B loads (Vec4U32), coalesced access pattern
- **Anti-DCE**: Checksum accumulation with minimal sink write (1 uint64 per block)
- **Characteristics**: High bandwidth utilization, low compute intensity

#### GEMM Kernel (Compute-Bound)
- **Purpose**: Stress compute units with matrix multiplication
- **Design**: Tiled GEMM with 32×32 tiles using shared memory
- **Operation**: C = A × B (all FP32)
- **Characteristics**: High compute intensity, low bandwidth requirements

### Concurrent Execution Strategy

The benchmark uses a critical synchronization pattern to ensure true concurrent execution:

```cuda
// Create synchronization event
cudaEvent_t start_sync;
cudaEventCreate(&start_sync);
cudaEventRecord(start_sync, 0);  // Record on default stream

// Make both streams wait for sync event
cudaStreamWaitEvent(stream_mem, start_sync, 0);
cudaStreamWaitEvent(stream_gemm, start_sync, 0);

// Launch both kernels (they start simultaneously)
launch_memcpy_kernel<<<..., stream_mem>>>(...);
launch_gemm_kernel<<<..., stream_gemm>>>(...);

// Synchronize both streams
cudaStreamSynchronize(stream_mem);
cudaStreamSynchronize(stream_gemm);
```

This ensures:
- **Simultaneous start**: Both kernels begin at approximately the same time
- **No false serialization**: Prevents one kernel from completing before the other starts
- **Accurate overlap measurement**: Wall-clock time reflects true concurrent execution

### Workload Sizing

Default parameters are tuned for 100-500ms kernel runtimes for reliable contention measurement:

- **Memory Copy**: 2 GiB (adjusts based on GPU memory bandwidth)
- **GEMM**: 2048×2048×2048 (~17.2 GFLOPS, adjusts based on GPU compute throughput)

For different GPUs, you may need to adjust:
- Slower GPUs: Reduce `--gemm_size` to 1024 or `--mem_bytes` to 1 GiB
- Faster GPUs: Increase `--gemm_size` to 4096 or `--mem_bytes` to 4 GiB

### Green Context Architecture

Each SM count test creates:
1. **One Green Context** with N SMs
2. **Two Streams** within the same context:
   - `stream_mem`: For memory copy kernel
   - `stream_gemm`: For GEMM kernel
3. Both streams share the same N SMs (not separate partitions)

This design ensures true resource contention between kernels.

## Validation

### Verifying Concurrent Execution

Check that concurrent execution is actually happening:

1. **Wall-clock time check**: Concurrent time should be less than sum of individual times
   - Example: If mem=150ms and gemm=100ms alone, concurrent should be ~150ms (not 250ms)

2. **Overlap percentage**: Should be >50% for most SM counts
   - Low overlap (<30%) indicates sequential execution (bug)

3. **Nsight Systems profiling**:
   ```bash
   nsys profile -o contention_test ./compute_mem_contention_test --min_sms 16 --max_sms 16
   # Open contention_test.nsys-rep in Nsight Systems GUI
   # Verify overlapping kernel bars on both streams
   ```

### Troubleshooting

If kernels appear to run sequentially:

1. **Check stream creation**: Both must use `CU_STREAM_NON_BLOCKING`
2. **Verify Green Context**: Both streams should belong to the same context
3. **Inspect synchronization**: Ensure sync event is used before kernel launches
4. **Check resource usage**: Very high register/smem usage can prevent concurrent execution
5. **Profile with Nsight Systems**: Visual timeline shows actual execution pattern

## Technical Notes

- **Green Context Limitation**: Cannot split a partition after creation. Each SM count requires a new green context.
- **Occupancy**: Tune blocks per SM to maximize occupancy without resource conflicts.
- **Architecture Differences**: Test on both Ampere (alignment=2) and Hopper (alignment=8) if possible.
- **Memory Bandwidth**: Shared DRAM means bandwidth contention persists even at high SM counts.
- **Cache Effects**: GEMM may pollute L2 cache, affecting memory copy performance.

## Comparison to Related Work

This experiment differs from `simple_sm_util_greencontext`:

| Aspect | simple_sm_util_greencontext | compute_mem_contention_test |
|--------|----------------------------|----------------------------|
| **Workloads** | Single memory-bound kernel | Dual: compute + memory |
| **Execution** | Isolated only | Isolated + concurrent |
| **Goal** | Measure SM scaling for bandwidth | Measure interference between workload types |
| **Metrics** | Bandwidth vs SM count | Slowdown, retention, overlap |

## References

- **CUDA Green Contexts Documentation**: [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#green-contexts)
- **Related Experiment**: `experiments/simple_sm_util_greencontext/`
- **CUDA Streams and Events**: [CUDA C++ Programming Guide - Streams](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#streams)

## Citation

If you use this benchmark in your research, please cite:

```
@misc{compute_mem_contention_test,
  title={Compute-Memory Contention Benchmark with CUDA Green Contexts},
  author={Your Name},
  year={2026},
  howpublished={\url{https://github.com/yourusername/lerobot_custom}}
}
```

## License

This project follows the same license as the parent repository.

## Contributing

Improvements and bug reports are welcome! Please submit issues or pull requests through the main repository.

---

**Note**: This benchmark requires CUDA 13.0+ and is primarily tested on NVIDIA Hopper (H100) and Ampere (A100) architectures. Performance characteristics may vary on other GPU architectures.

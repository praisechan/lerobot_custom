# GEMV-GEMM Contention Test with CUDA Green Contexts

A microbenchmark to measure contention between GEMV (General Matrix-Vector multiplication, memory-bound) and GEMM (General Matrix-Matrix multiplication, compute-bound) kernels when they execute concurrently on the same restricted SM subset using CUDA Green Contexts.

**Uses FP16 precision with Tensor Cores for realistic ML workload simulation.**

## Overview

This benchmark explores how memory-intensive (GEMV) and compute-intensive (GEMM) workloads interfere with each other when forced to share GPU resources. By using **CUDA Green Contexts** (introduced in CUDA 13.0), we can partition GPU resources and restrict both kernels to the same N SMs, measuring how performance degrades under contention compared to isolated execution.

### Key Features

- **Dual-kernel contention measurement**: GEMV (memory-bound) vs GEMM (compute-bound)
- **FP16 precision with Tensor Cores**: GEMM uses WMMA API for Tensor Core acceleration
- **Three execution modes**: Isolated GEMV, isolated GEMM, and concurrent execution
- **Green Context-based resource sharing**: Both kernels share the same SM partition
- **Lockstep synchronized execution**: Per-iteration cross-stream synchronization ensures true concurrency
- **Comprehensive metrics**: GFLOPS, execution time, slowdown ratios, overlap percentage
- **Statistical measurement**: Multiple repeats with averaging
- **Rich visualization**: Four plots analyzing different aspects of contention

### Why GEMV vs GEMM?

**GEMV (Matrix-Vector Multiplication):**
- Operation: `y = A * x` where A is [M×N], x is [N], y is [M]
- Data type: FP16 (__half)
- FLOPs: 2×M×N (multiply and add)
- Memory: Reads M×N FP16 elements from A + N from x, writes M to y
- **Arithmetic Intensity**: ~1.0 FLOPs/byte (memory-bound, FP16 uses half bandwidth of FP32)
- Bottleneck: Memory bandwidth
- Real-world: Common in neural network inference (attention mechanisms, linear layers with batch size 1)

**GEMM (Matrix-Matrix Multiplication):**
- Operation: `C = A * B` where all matrices are square [N×N]
- Data type: FP16 (__half) with **Tensor Core acceleration via WMMA**
- FLOPs: 2×N³
- Memory: Reads 2×N² FP16 elements, writes N² FP16 elements
- **Arithmetic Intensity**: ~O(N) FLOPs/byte (compute-bound for large N)
- Bottleneck: Tensor Core throughput (much higher than FP32 CUDA cores)
- Real-world: Training and batched inference in modern ML workloads

### Why This Matters

Understanding GEMV-GEMM contention is critical for:

- **Mixed workload scheduling**: Optimal concurrent execution of different operation types
- **Multi-tenancy scenarios**: MPS (Multi-Process Service) and MIG (Multi-Instance GPU) contexts
- **Pipeline optimization**: Overlapping attention (GEMV-heavy) with FFN (GEMM-heavy) in transformers
- **Resource allocation**: Determining optimal SM partitioning for heterogeneous workloads
- **Performance modeling**: Predicting interference effects in shared GPU environments

## Requirements

- **CUDA Toolkit**: 13.0 or later (for Green Contexts support)
- **Compute Capability**: 7.0+ (**Volta or later for Tensor Cores**)
  - Hopper (9.0+): Min 8 SMs, alignment 8, **4th gen Tensor Cores**
  - Ampere (8.x): Min 4 SMs, alignment 2, **3rd gen Tensor Cores**
  - Volta/Turing (7.x): Min 2 SMs, alignment 2, **1st/2nd gen Tensor Cores**
- **Platform**: 64-bit Linux (Green Contexts not supported on 32-bit)
- **CMake**: 3.18+
- **Python 3**: For plotting (matplotlib, pandas)

## Build Instructions

```bash
cd ~/lerobot_custom/experiments/GEMV_GEMM_contention_test

# Create build directory
mkdir build && cd build

# Configure
cmake ..

# Build
cmake --build . -j

# Binary location: ./GEMV_GEMM_contention_test
```

## Usage

### Basic Usage

```bash
# Default: sweep all SMs, 32768×32768 GEMV, 2048×2048 GEMM, 8 iters each
./GEMV_GEMM_contention_test

# Custom SM range
./GEMV_GEMM_contention_test --min_sms 8 --max_sms 64

# Larger GEMV workload
./GEMV_GEMM_contention_test --gemv_M 49152 --gemv_N 49152

# Larger GEMM workload
./GEMV_GEMM_contention_test --gemm_size 3072

# More iterations (both kernels)
./GEMV_GEMM_contention_test --num_iters 16

# More repeats for stability
./GEMV_GEMM_contention_test --repeats 10

# Custom output
./GEMV_GEMM_contention_test --csv my_results.csv
```

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--min_sms <N>` | Minimum SM count | 8 |
| `--max_sms <N>` | Maximum SM count | Device max |
| `--gemv_M <N>` | GEMV matrix rows | 32768 |
| `--gemv_N <N>` | GEMV matrix columns | 32768 |
| `--gemm_size <N>` | Matrix dimension for GEMM (M=N=K) | 2048 |
| `--num_iters <N>` | Number of iterations for both kernels | 8 |
| `--tpb_gemv <N>` | Threads per block for GEMV | 256 |
| `--tpb_gemm <N>` | Threads per block for GEMM | 256 (not used, fixed tile size) |
| `--repeats <N>` | Number of measurement repeats | 5 |
| `--csv <path>` | Output CSV path | ./results.csv |
| `--help` | Print usage | - |

### Example Output

```
=============================================================
GEMV-GEMM Contention Test (FP16 with Tensor Cores)
=============================================================
Device: NVIDIA H100
Compute Capability: 9.0
Total SMs: 132
Configuration:
  SM Range: 8 - 132
  Unified Iterations: 8
  Data Type: FP16 (__half)
  GEMV: [32768 x 32768] matrix × vector (FP16)
  GEMV Expected FLOPs (total): 17.18 GFLOPs
  GEMM Size: 2048 x 2048 x 2048 (FP16 with WMMA Tensor Cores)
  GEMM Expected FLOPs (total): 137.44 GFLOPs
  Threads per block (GEMV): 256
  Repeats: 5
  CSV output: ./results.csv
=============================================================

Testing with 8 SMs:
  [GEMV Only]  Throughput: 76.37 GFLOPS, Time: 224.95 ms
  [GEMM Only]  Throughput: 4765.08 GFLOPS, Time: 28.84 ms
  [Concurrent] Wall Time: 252.73 ms (overlap: 0.4%)
               GEMV: 67.98 GFLOPS (89.0% retained), Time: 252.73 ms, Slowdown: 1.12x
               GEMM: 610.49 GFLOPS (12.8% retained), Time: 225.13 ms, Slowdown: 7.81x

Testing with 16 SMs:
  [GEMV Only]  Throughput: 78.05 GFLOPS, Time: 220.12 ms
  [GEMM Only]  Throughput: 4799.47 GFLOPS, Time: 28.64 ms
  [Concurrent] Wall Time: 249.60 ms (overlap: -0.3%)
               GEMV: 68.83 GFLOPS (88.2% retained), Time: 249.60 ms, Slowdown: 1.13x
               GEMM: 619.12 GFLOPS (12.9% retained), Time: 221.99 ms, Slowdown: 7.75x
...
```

## Output Format

### CSV Structure

The benchmark generates a CSV file with the following columns:

| Column | Description |
|--------|-------------|
| `num_sms` | Number of SMs in the partition |
| `mode` | Execution mode: `isolated_gemv`, `isolated_gemm`, `concurrent` |
| `gemv_gflops` | GEMV throughput (GFLOPS) |
| `gemm_gflops` | GEMM throughput (GFLOPS) |
| `gemv_time_ms` | GEMV kernel time (ms) |
| `gemm_time_ms` | GEMM kernel time (ms) |
| `gemv_slowdown` | GEMV slowdown vs isolated (concurrent only) |
| `gemm_slowdown` | GEMM slowdown vs isolated (concurrent only) |
| `overlap_pct` | Time overlap percentage (concurrent only) |

### Example CSV

```csv
num_sms,mode,gemv_gflops,gemm_gflops,gemv_time_ms,gemm_time_ms,gemv_slowdown,gemm_slowdown,overlap_pct
8,isolated_gemv,38.500,0.0,111.43,0.0,1.0,0.0,0.0
8,isolated_gemm,0.0,245.320,0.0,70.12,0.0,1.0,0.0
8,concurrent,35.200,223.450,121.88,76.89,1.09,1.10,31.0
16,isolated_gemv,48.500,0.0,88.45,0.0,1.0,0.0,0.0
16,isolated_gemm,0.0,490.650,0.0,35.06,0.0,1.0,0.0
16,concurrent,45.670,445.210,93.92,38.61,1.06,1.10,22.8
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

2. **`results_absolute_performance.png`**: Absolute GFLOPS comparison
   - Dual Y-axis plot showing isolated vs concurrent performance for both kernels
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

Compared to mem_copy-GEMM contention, GEMV-GEMM should show:

1. **Different contention characteristics**: Both do compute, but GEMV is memory-bottlenecked
2. **Potentially more slowdown**: Both access memory subsystem (L2 cache, DRAM)
3. **Better overlap**: GEMV does more work than simple mem_copy, better hiding latency
4. **Cache contention**: GEMM may evict GEMV's data from L2, hurting GEMV more

## Implementation Details

### Kernel Descriptions

#### GEMV Kernel (Memory-Bound)
- **Purpose**: Stress memory bandwidth with matrix-vector product
- **Data type**: FP16 (__half) with FP32 accumulation
- **Design**: Each thread computes one output element (dot product of matrix row with vector)
- **Iteration**: Launched multiple times from host for continuous work submission
- **Characteristics**: 
  - Arithmetic intensity: ~1.0 FLOPs/byte (FP16)
  - Memory-bound: Limited by DRAM bandwidth
  - Streaming access pattern for matrix A
  - 2x bandwidth efficiency vs FP32

#### GEMM Kernel (Compute-Bound with Tensor Cores)
- **Purpose**: Stress Tensor Cores with matrix multiplication
- **Data type**: FP16 (__half) with FP16 accumulation
- **Design**: WMMA (Warp Matrix Multiply-Accumulate) API with 16×16×16 tiles
- **Operation**: C = A × B (all FP16, using Tensor Cores)
- **Iteration**: Launched multiple times from host for continuous work submission
- **Characteristics**: 
  - Very high compute intensity (Tensor Core accelerated)
  - Compute-bound: Limited by Tensor Core throughput
  - ~8-16x faster than FP32 CUDA cores for GEMM
  - Native FP16 matrix operations

### Concurrent Execution Strategy

The benchmark uses a **per-iteration synchronization pattern** to ensure continuous concurrent execution:

```cuda
// For each iteration (e.g., 8 iterations)
for (int iter = 0; iter < num_iters; ++iter) {
    // Create synchronization event
    cudaEvent_t sync_event;
    cudaEventCreate(&sync_event);
    cudaEventRecord(sync_event, 0);  // Record on default stream

    // Make both streams wait for sync event (simultaneous start)
    cudaStreamWaitEvent(stream_gemv, sync_event, 0);
    cudaStreamWaitEvent(stream_gemm, sync_event, 0);

    // Launch both kernels (single iteration each)
    launch_gemv_kernel<<<..., stream_gemv>>>(...);  // Single GEMV
    launch_gemm_kernel<<<..., stream_gemm>>>(...);  // Single GEMM
    
    cudaEventDestroy(sync_event);
}

// Synchronize both streams
cudaStreamSynchronize(stream_gemv);
cudaStreamSynchronize(stream_gemm);
```

**Key Design Decisions:**
- **Host-side iteration**: Both kernels are launched multiple times from the host (not internal kernel loops)
- **Per-iteration sync**: Each iteration starts with both kernels synchronized for fair comparison
- **Continuous work submission**: Both streams have work throughout the entire measurement period
- **Matched iteration counts**: Both kernels run the same number of iterations with similar per-iteration latency

### Workload Sizing

Default parameters are tuned for matched kernel runtimes per iteration (~30ms for 16 SMs with Tensor Cores):

- **GEMV**: 32768×32768 matrix-vector product in FP16 (~17.2 GFLOPs total for 8 iterations)
  - Single iteration: ~2.15 GFLOPs, ~28ms at 76 GFLOPS throughput (16 SMs, FP16)
  - 8 iterations: ~225ms at 76 GFLOPS throughput
- **GEMM**: 2048×2048×2048 matrix multiplication in FP16 with Tensor Cores (~137.4 GFLOPs total for 8 iterations)
  - Single iteration: ~17.2 GFLOPs, ~3.6ms at 4800 GFLOPS throughput (16 SMs, Tensor Cores)
  - 8 iterations: ~29ms at 4800 GFLOPS throughput

**Note**: Both kernels are launched **iteratively from the host**, with per-iteration cross-stream synchronization (lockstep) to ensure truly concurrent execution. This prevents faster kernels from getting ahead and ensures fair contention measurement throughout.

### Green Context Architecture

Each SM count test creates:
1. **One Green Context** with N SMs
2. **Two Streams** within the same context:
   - `stream_gemv`: For GEMV kernel
   - `stream_gemm`: For GEMM kernel
3. Both streams share the same N SMs (not separate partitions)

This design ensures true resource contention between kernels.

## Validation

### Verifying Concurrent Execution

Check that concurrent execution is actually happening:

1. **Wall-clock time check**: Concurrent time should be less than sum of individual times
   - Example: If GEMV=220ms and GEMM=29ms alone, concurrent should be ~250ms (not 249ms)

2. **Overlap percentage**: Should be >10% for most SM counts
   - Low overlap (<10%) may indicate sequential execution or very different runtimes

3. **Nsight Systems profiling**:
   ```bash
   nsys profile -o gemv_gemm_test ./GEMV_GEMM_contention_test --min_sms 16 --max_sms 16
   # Open gemv_gemm_test.nsys-rep in Nsight Systems GUI
   # Verify overlapping kernel bars on both streams
   # Check "Tensor Core %%" metric to confirm Tensor Core usage for GEMM
   ```

### Verifying Tensor Core Usage

In Nsight Systems, you should see:
- **High "Tensor Core %" metric** for GEMM kernels (~80-90%)
- **WMMA instructions** in the kernel source view
- **Much higher GFLOPS** for GEMM vs FP32 implementation (~4800 vs ~530 GFLOPS on modern GPUs)

## Comparison to Related Work

This experiment differs from `compute_mem_contention_test`:

| Aspect | compute_mem_contention_test | GEMV_GEMM_contention_test |
|--------|----------------------------|---------------------------|
| **Memory Kernel** | Simple vectorized read | GEMV (compute + memory) |
| **Complexity** | Minimal compute | 2×M×N FLOPs |
| **Cache Usage** | Streaming, no reuse | Matrix rows accessed |
| **Expected Contention** | Primarily bandwidth | Bandwidth + cache interference |
| **Real-world Analog** | Data loading/transfer | Attention mechanisms, embedding lookups |

## Technical Notes

- **FP16 Precision**: All computations use FP16 (__half) data type for realistic modern ML workload simulation
- **Tensor Core Acceleration**: GEMM uses WMMA (Warp Matrix Multiply-Accumulate) API with 16×16×16 tiles
- **Tensor Core Benefits**: 
  - ~8-16x higher throughput than FP32 CUDA cores for matrix multiplication
  - Native hardware support for FP16 matrix operations
  - Critical for modern deep learning workloads
- **Lockstep Synchronization**: Per-iteration cross-stream synchronization prevents faster kernels from getting ahead
- **Host-side Iterations**: Both kernels are launched iteratively from the host (not internal kernel loops), ensuring both streams continuously have work throughout the measurement period
- **Matched Latencies**: Default sizes are tuned so single-iteration latencies are similar for meaningful contention measurement
- **Memory Bandwidth**: FP16 uses half the bandwidth of FP32, doubling effective throughput for memory-bound operations
- **Cache Effects**: GEMM Tensor Core operations may evict GEMV's FP16 matrix data from L2 cache
- **L2 Contention**: Both kernels compete for L2 cache capacity and bandwidth
- **Arithmetic Intensity**: GEMV's AI (~1.0 FLOPs/byte for FP16) means it's still memory-bound but more balanced than FP32

## References

- **CUDA Green Contexts Documentation**: [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#green-contexts)
- **Related Experiments**: 
  - `experiments/compute_mem_contention_test/` (mem_copy vs GEMM)
  - `experiments/simple_sm_util_greencontext/` (bandwidth scaling)
- **CUDA Streams and Events**: [CUDA C++ Programming Guide - Streams](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#streams)

## License

This project follows the same license as the parent repository.

---

**Note**: This benchmark requires CUDA 13.0+ and is primarily tested on NVIDIA Hopper (H100) and Ampere (A100) architectures. Performance characteristics may vary on other GPU architectures.

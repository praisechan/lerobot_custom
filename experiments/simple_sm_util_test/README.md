# SM-limited Read Bandwidth Sweep

This microbenchmark sweeps SM counts and measures DRAM read bandwidth. It attempts to use CUDA Green Contexts (execution affinity) to control SM usage. If Green Contexts are not supported by the hardware/driver, it falls back to the primary context and varies grid size instead.

## What are Green Contexts?

Green Contexts allow partitioning GPU resources (SMs and work queues) to control which parts of the GPU different workloads can use. This is useful for:
- Ensuring latency-sensitive work always has available SMs
- Reducing interference between concurrent GPU workloads
- Testing performance with limited SM counts

**Requirements:**
- CUDA 11.4+ (API availability)
- Supported GPU architectures: Hopper (H100), Blackwell, and newer
- Driver support for execution affinity

See [NVIDIA Green Contexts Documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/green-contexts.html)

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

## Green Context APIs

This project uses the **CUDA Driver API** (`cuCtxCreate_v3`) for backward compatibility with CUDA 11.4-12.x. 

**For new projects, NVIDIA recommends the Runtime API** (CUDA 13.1+), which provides:
- Higher-level abstractions
- Better resource management  
- Support for heterogeneous SM partitioning
- Easier workqueue configuration

See `green_context_runtime_api_example.cu` for a complete Runtime API example.

### Driver API approach (current implementation):
```cpp
CUexecAffinityParam params[1];
params[0].type = CU_EXEC_AFFINITY_TYPE_SM_COUNT;
params[0].param.smCount.val = sm_count;
cuCtxCreate_v3(&ctx, params, 1, 0, dev);
```

### Runtime API approach (recommended for CUDA 13.1+):
```cpp
cudaDeviceGetDevResource(device, &resource, cudaDevResourceTypeSm);
cudaDevSmResourceSplit(&result, nbGroups, &resource, &remainder, flags, &groupParams);
cudaDevResourceGenerateDesc(&desc, &result, nbResources);
cudaGreenCtxCreate(&greenCtx, desc, device, 0);
cudaExecutionCtxStreamCreate(&stream, greenCtx, cudaStreamDefault, 0);
```

## Hardware Compatibility

**Supported GPUs:**
- NVIDIA H100 (Hopper architecture)
- NVIDIA Blackwell and newer architectures

**Not Supported (will fall back to primary context):**
- RTX A6000, RTX 3090, RTX 4090 (Ampere/Ada Lovelace)
- A100, A40 (Ampere)
- V100 (Volta)
- Older architectures

Check support programmatically:
```cpp
int supported = 0;
cuDeviceGetExecAffinitySupport(&supported, CU_EXEC_AFFINITY_TYPE_SM_COUNT, device);
```

## Plot

```bash
python3 tools/plot.py --csv ./results.csv --out ./bw_vs_sm.png
```

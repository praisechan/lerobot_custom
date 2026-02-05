# Pi0Inference CUDA MPS Support

## Overview

The `Pi0Inference` class now includes a new `forward_mps()` method that enables **CUDA MPS (Multi-Process Service)** for concurrent execution of encoder and decoder with adjustable compute resource allocation.

## What is CUDA MPS?

CUDA Multi-Process Service allows multiple CUDA applications or streams to share a single GPU, with fine-grained control over SM (Streaming Multiprocessor) allocation via the `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` environment variable.

**Benefits:**
- Fine-grained control over SM resource allocation between concurrent workloads
- Potential for better load balancing than traditional concurrent streams
- Reduced GPU idle time when encoder and decoder have different computational intensity

## Method Signature

```python
def forward_mps(self, observation_images_normalized, observation_state_normalized, diffusion_noise, 
                mps_encoder_percentage=50, mps_decoder_percentage=50, concurrent=False):
    """
    Forward pass using CUDA MPS (Multi-Process Service) for concurrent execution.
    
    Args:
        observation_images_normalized: Input images tensor [num_views, 224, 224, 3]
        observation_state_normalized: Input state tensor [32]
        diffusion_noise: Input noise tensor [chunk_size, 32]
        mps_encoder_percentage: Percentage of SMs allocated to encoder (1-100). Default: 50
        mps_decoder_percentage: Percentage of SMs allocated to decoder (1-100). Default: 50
        concurrent: If True, run encoder and decoder concurrently with MPS. 
                   If False, run sequentially with MPS. Default: False
    
    Returns:
        Output diffusion noise tensor [chunk_size, 32]
    """
```

## Usage Examples

### Example 1: Sequential Execution with MPS

Encoder and decoder run sequentially, each getting full GPU resources:

```python
output = infer.forward_mps(
    input_image, 
    input_state, 
    input_noise, 
    mps_encoder_percentage=100,
    mps_decoder_percentage=100,
    concurrent=False
)
```

### Example 2: Concurrent with Balanced Resources (50-50 Split)

Both encoder and decoder run concurrently, sharing GPU resources equally:

```python
output = infer.forward_mps(
    input_image, 
    input_state, 
    input_noise, 
    mps_encoder_percentage=50,
    mps_decoder_percentage=50,
    concurrent=True
)
```

### Example 3: Concurrent with Encoder-Heavy Allocation (70-30 Split)

Prioritize encoder execution when it's the performance bottleneck:

```python
output = infer.forward_mps(
    input_image, 
    input_state, 
    input_noise, 
    mps_encoder_percentage=70,
    mps_decoder_percentage=30,
    concurrent=True
)
```

### Example 4: Concurrent with Decoder-Heavy Allocation (30-70 Split)

Prioritize decoder execution when it's the performance bottleneck:

```python
output = infer.forward_mps(
    input_image, 
    input_state, 
    input_noise, 
    mps_encoder_percentage=30,
    mps_decoder_percentage=70,
    concurrent=True
)
```

## Implementation Details

### Sequential Mode (`concurrent=False`)

1. **Setup**: Copy input data to GPU buffers
2. **Encoder Phase**:
   - Set `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` to `mps_encoder_percentage`
   - Synchronize GPU
   - Execute encoder model
   - Synchronize GPU
3. **Decoder Phase**:
   - Set `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` to `mps_decoder_percentage`
   - Synchronize GPU
   - Execute decoder model
   - Synchronize GPU
4. **Return**: Output tensor

### Concurrent Mode (`concurrent=True`)

1. **Setup**: Copy input data to GPU buffers, create two CUDA streams
2. **Stream Creation**:
   - `stream_encoder`: For encoder execution
   - `stream_decoder`: For decoder execution
3. **Allocation & Execution**:
   - Set encoder SM percentage and launch encoder on `stream_encoder`
   - Set decoder SM percentage and launch decoder on `stream_decoder`
   - Both streams execute concurrently with MPS scheduling
4. **Synchronization**:
   - Wait for both streams to complete
   - Return output tensor

## Comparison with Original Methods

| Method | Mode | Resource Control | Usage |
|--------|------|------------------|-------|
| `forward()` | Sequential | None | Baseline inference |
| `forward()` | Concurrent (streams) | None | Concurrent without MPS |
| `forward_mps()` | Sequential | Per-stage percentages | Sequential with MPS allocation |
| `forward_mps()` | Concurrent | Per-stage percentages | Concurrent with MPS control |

## Performance Tuning Guide

### When to Use Each Mode

**Sequential Mode (`concurrent=False`)**
- When you want maximum throughput for each component
- When memory bandwidth is the bottleneck
- For baseline performance measurements

**Concurrent Mode (`concurrent=True`)**
- When both encoder and decoder can overlap execution
- When you want to minimize total latency
- When SM allocation can be optimized

### Resource Allocation Strategies

**50-50 Split** (Balanced)
- Good starting point for benchmarking
- Works well when encoder and decoder have similar computational intensity
- Command: `mps_encoder_percentage=50, mps_decoder_percentage=50`

**70-30 Split** (Encoder-Heavy)
- Encoder is often the bottleneck in vision-language models
- More SMs allocated to encoder can improve overall throughput
- Command: `mps_encoder_percentage=70, mps_decoder_percentage=30`

**30-70 Split** (Decoder-Heavy)
- For decoder-intensive workloads
- When decoder has more complex operations
- Command: `mps_encoder_percentage=30, mps_decoder_percentage=70`

**100-100 Sequential**
- Equivalent to `forward_mps(..., concurrent=False)`
- Each component gets full GPU resources sequentially
- Command: `mps_encoder_percentage=100, mps_decoder_percentage=100, concurrent=False`

## Profiling with NSys

To analyze the execution timeline and measure performance improvements:

```bash
# Profile with CUDA MPS (50-50 split, concurrent)
nsys profile -o mps_concurrent_50_50 python pi0_infer_mps_example.py \
    --iterations 10

# View the timeline
nsys-ui mps_concurrent_50_50.nsys-rep
```

Key metrics to observe:
- **Kernel overlap**: How much encoder and decoder kernels overlap
- **SM utilization**: Whether both components are actively using assigned SMs
- **Total latency**: Measure end-to-end latency improvements
- **Memory bandwidth**: Check for memory contention between streams

## Important Notes

1. **Environment Variable Scope**: 
   - `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` is a process-wide setting
   - Changes affect all subsequent GPU operations until the value is changed again
   - GPU synchronization is performed before each allocation change to ensure proper enforcement

2. **MPS Limitations**:
   - MPS requires Maxwell generation (GTX 750+) or newer NVIDIA GPUs
   - Not all CUDA operations are MPS-compatible
   - Actual resource allocation depends on NVIDIA MPS scheduler

3. **Resource Constraints**:
   - Sum of percentages should typically be ≤ 100 for concurrent mode
   - Values > 100 are valid but may not provide additional benefits
   - Minimum value is typically 1 (1% of SMs)

4. **Synchronization**:
   - Explicit `torch.cuda.synchronize()` calls ensure MPS settings take effect
   - Stream synchronization ensures proper ordering of encoder/decoder execution

## Example Output

```
================================================================================
Test 1: Sequential execution with MPS
================================================================================
Sequential MPS execution completed

================================================================================
Test 2: Concurrent MPS with 50-50 resource split
================================================================================
Concurrent MPS 50-50 execution completed

================================================================================
Test 3: Concurrent MPS with encoder-heavy allocation (70-30)
================================================================================
Concurrent MPS 70-30 execution completed

================================================================================
Test 4: Concurrent MPS with decoder-heavy allocation (30-70)
================================================================================
Concurrent MPS 30-70 execution completed

================================================================================
Test 5: Original concurrent streams (no MPS) for comparison
================================================================================
Original concurrent streams execution completed
```

## Troubleshooting

**Issue**: MPS percentage changes don't seem to affect performance
- **Solution**: Ensure `torch.cuda.synchronize()` is called between MPS allocation changes

**Issue**: Concurrent execution is slower than sequential
- **Solution**: Try increasing the percentage for the slower component (70-30 or 80-20)

**Issue**: "CUDA MPS not supported" error
- **Solution**: Check that your GPU is Maxwell generation or newer, and MPS daemon is running

## References

- [NVIDIA CUDA Multi-Process Service Documentation](https://docs.nvidia.com/cuda/mps-user-guide/)
- [Semi-PD Implementation](../Semi-PD/python/sglang/srt/entrypoints/engine.py) for production examples
- [NSys Profiling Guide](https://docs.nvidia.com/nsight-systems/profiling/index.html)

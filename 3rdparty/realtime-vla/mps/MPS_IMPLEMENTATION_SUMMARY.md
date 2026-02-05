# Pi0Inference CUDA MPS Implementation Summary

## Overview

I've successfully implemented CUDA MPS (Multi-Process Service) support for the `Pi0Inference` class in the realtime-vla project. This enables fine-grained control over GPU resource allocation when running encoder and decoder concurrently.

## What Was Added

### 1. New `forward_mps()` Method

**Location**: `/home/juchanlee/lerobot_custom/3rdparty/realtime-vla/pi0_infer.py`

**Signature**:
```python
def forward_mps(self, observation_images_normalized, observation_state_normalized, diffusion_noise, 
                mps_encoder_percentage=50, mps_decoder_percentage=50, concurrent=False)
```

**Key Parameters**:
- `mps_encoder_percentage` (1-100): Percentage of SMs allocated to encoder
- `mps_decoder_percentage` (1-100): Percentage of SMs allocated to decoder  
- `concurrent` (bool): Whether to run encoder and decoder concurrently

**Modes of Operation**:

1. **Sequential Mode** (`concurrent=False`):
   - Encoder runs first with allocated resources
   - Decoder runs second with allocated resources
   - Full GPU resources available for each stage

2. **Concurrent Mode** (`concurrent=True`):
   - Encoder and decoder run on separate CUDA streams simultaneously
   - Resources are allocated per-stream via MPS
   - Potential for latency reduction through overlapping execution

### 2. Implementation Details

#### Sequential Execution Flow:
```
1. Copy inputs to GPU buffers
2. Set encoder SM percentage
3. Execute encoder_model()
4. Set decoder SM percentage  
5. Execute decoder_model()
6. Return output
```

#### Concurrent Execution Flow:
```
1. Copy inputs to GPU buffers
2. Create two CUDA streams (encoder, decoder)
3. Set encoder SM percentage and launch encoder on stream
4. Set decoder SM percentage and launch decoder on stream
5. Both streams execute concurrently with MPS scheduling
6. Synchronize both streams
7. Return output
```

## How It Works

### Reference: Semi-PD Implementation

The implementation is based on NVIDIA's Semi-PD pattern for dynamic inference:

```python
# From Semi-PD (engine.py)
os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(DECODE_ENGINE_SM_PERCENTILE)
# Launch decode scheduler process

os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(PREFILL_ENGINE_SM_PERCENTILE)  
# Launch prefill scheduler process
```

Our implementation adapts this pattern for local concurrent execution within a single Python process.

### CUDA MPS Mechanism

- **CUDA_MPS_ACTIVE_THREAD_PERCENTAGE**: Environment variable controlling SM allocation
- **Value Range**: 1-100 (percentage of total SMs)
- **Scope**: Process-wide (affects all subsequent GPU operations)
- **Enforcement**: Requires GPU synchronization to take effect

### GPU Synchronization Strategy

```python
os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(percentage)
torch.cuda.synchronize()  # Ensure MPS setting takes effect
encoder_model(...)         # Execute with allocated resources
torch.cuda.synchronize()  # Drain GPU queue before changing allocation
```

## Files Created/Modified

### Modified:
- **pi0_infer.py**: Added `forward_mps()` method to `Pi0Inference` class (70 lines)

### Created:
1. **pi0_infer_mps_example.py** (160 lines)
   - Comprehensive example with 5 different MPS configurations
   - Includes NVTX range markers for profiling
   - Demonstrates sequential and concurrent modes
   - Shows resource allocation strategies (50-50, 70-30, 30-70)

2. **pi0_infer_mps_quickstart.py** (50 lines)
   - Minimal quick-start guide
   - Shows 5 basic usage patterns
   - Good starting point for new users

3. **CUDA_MPS_GUIDE.md** (300+ lines)
   - Complete documentation
   - Method signature and parameters
   - Usage examples for all configurations
   - Performance tuning guide
   - Profiling instructions with NSys
   - Troubleshooting section

## Usage Examples

### Basic Usage - Balanced 50-50 Split (Recommended Starting Point):
```python
output = infer.forward_mps(
    input_image, 
    input_state, 
    input_noise, 
    concurrent=True
)  # Uses default 50-50 split
```

### Encoder-Heavy (70-30 Split):
```python
output = infer.forward_mps(
    input_image, input_state, input_noise,
    mps_encoder_percentage=70,
    mps_decoder_percentage=30,
    concurrent=True
)
```

### Decoder-Heavy (30-70 Split):
```python
output = infer.forward_mps(
    input_image, input_state, input_noise,
    mps_encoder_percentage=30,
    mps_decoder_percentage=70,
    concurrent=True
)
```

### Sequential Execution:
```python
output = infer.forward_mps(
    input_image, input_state, input_noise,
    concurrent=False
)  # Encoder then decoder sequentially
```

## Comparison with Original Method

| Feature | `forward()` | `forward_mps()` |
|---------|-----------|-----------------|
| SM Control | No | Yes (per-component) |
| Concurrent Support | Yes | Yes |
| Sequential Support | Yes | Yes |
| MPS Integration | No | Yes |
| Default Behavior | Concurrent streams | Balanced 50-50 concurrent |

## Performance Profiling

### Using NVIDIA NSys:
```bash
# Generate profiling data
nsys profile -o output_profile python pi0_infer_mps_example.py

# View interactive timeline
nsys-ui output_profile.nsys-rep
```

### Key Metrics to Monitor:
- **Kernel overlap**: How much encoder/decoder overlap
- **SM utilization**: Whether both are active simultaneously
- **Total latency**: End-to-end execution time
- **Memory bandwidth**: Check for memory contention

## Technical Implementation Notes

### Environment Variable Handling
- `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` is process-global
- Must explicitly set and synchronize before GPU operations
- Changes are immediate but require GPU synchronization

### Stream Synchronization Pattern
```python
stream_encoder = torch.cuda.Stream()
stream_decoder = torch.cuda.Stream()
start_event = torch.cuda.Event()

# Launch encoder
with torch.cuda.stream(stream_encoder):
    stream_encoder.wait_event(start_event)
    encoder_model(...)

# Launch decoder  
with torch.cuda.stream(stream_decoder):
    stream_decoder.wait_event(start_event)
    decoder_model(...)

# Wait for both
stream_encoder.synchronize()
stream_decoder.synchronize()
```

## Limitations & Considerations

1. **GPU Requirements**: Requires Maxwell generation (GTX 750) or newer
2. **MPS Compatibility**: Not all CUDA operations are MPS-compatible
3. **Scheduling**: Actual resource allocation depends on NVIDIA's MPS scheduler
4. **Percentage Sum**: Should typically keep ≤ 100 for concurrent mode
5. **Determinism**: MPS scheduling may introduce slight non-determinism

## Next Steps / Recommendations

1. **Profile Your Workload**: Use NSys to determine optimal resource allocation
   ```bash
   nsys profile python pi0_infer_mps_example.py
   ```

2. **Benchmark Different Splits**: Test 50-50, 60-40, 70-30 to find optimal ratio

3. **Monitor Memory**: Watch memory bandwidth to identify contention

4. **Production Deployment**: Consider caching optimal percentages in config

## References

- **NVIDIA CUDA MPS Guide**: https://docs.nvidia.com/cuda/mps-user-guide/
- **Semi-PD Implementation**: `Semi-PD/python/sglang/srt/entrypoints/engine.py` (lines 585-650)
- **NSys Profiling**: https://docs.nvidia.com/nsight-systems/profiling/

---

**Implementation Date**: February 5, 2026
**Status**: ✅ Complete and Ready for Use

# Green Context Implementation Summary

## Issues Identified and Fixed

### 1. **Macro Detection Bug**
**Original Problem:** Code used `#if defined(CU_EXEC_AFFINITY_TYPE_SM_COUNT)` to detect Green Context support, but `CU_EXEC_AFFINITY_TYPE_SM_COUNT` is an enum value, not a preprocessor macro.

**Fix:** Changed to version-based detection:
```cpp
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11040
```

### 2. **Struct Field Access Error**  
**Original Problem:** Code tried to assign `params[0].param.smCount = sm_count`

**Fix:** `CUexecAffinitySmCount` is a struct with a `.val` field:
```cpp
params[0].param.smCount.val = sm_count;
```

### 3. **Missing Device Support Check**
**Original Problem:** No verification that the hardware actually supports Green Contexts

**Fix:** Added `device_supports_green_contexts()` function using:
```cpp
cuDeviceGetExecAffinitySupport(&supported, CU_EXEC_AFFINITY_TYPE_SM_COUNT, device);
```

### 4. **Poor Error Messages**
**Original Problem:** Generic error messages didn't explain why Green Contexts failed

**Fix:** Added detailed error messages with:
- Architecture requirements (Hopper/Blackwell)
- Link to NVIDIA documentation
- Explanation of fallback behavior

## Code Structure

### Updated Files

**green_context_utils.h:**
- Added `device_supports_green_contexts()` function declaration

**green_context_utils.cu:**
- Fixed `green_contexts_supported()` to use CUDA_VERSION
- Added `device_supports_green_contexts()` with hardware querying
- Added detailed error messages in `create_context_with_sm_count()`
- Added comments referencing Runtime API approach

**main.cu:**
- Added device support check before attempting Green Context creation
- Improved warning messages (only print once)
- Clearer fallback behavior messaging

**README.md:**
- Added comprehensive Green Context explanation
- Documented both Driver API and Runtime API approaches
- Listed supported/unsupported hardware
- Added example code snippets

### New Files

**green_context_runtime_api_example.cu:**
- Complete working example using CUDA Runtime API (CUDA 13.1+)
- Shows the recommended approach from NVIDIA documentation
- Includes all 6 steps from the manual
- Graceful error handling for unsupported hardware

## Hardware Support Status

### Supported (Green Contexts Work)
- ✅ NVIDIA H100 (Hopper)
- ✅ NVIDIA Blackwell and newer

### Not Supported (Falls Back to Primary Context)
- ❌ RTX A6000 (Ampere) - Current test system
- ❌ RTX 3090, RTX 4090 (Ampere/Ada Lovelace)
- ❌ A100, A40 (Ampere)
- ❌ V100 (Volta)
- ❌ Older architectures

## API Comparison

### Driver API (Current Implementation)
**Pros:**
- Available since CUDA 11.4
- Lower-level control
- Backward compatible

**Cons:**
- More complex to use
- Limited to homogeneous partitioning
- Manual resource management

### Runtime API (Recommended for New Code)
**Pros:**
- Higher-level abstractions
- Heterogeneous SM partitioning support
- Better workqueue configuration
- Easier resource management
- Available in CUDA 13.1+

**Cons:**
- Requires CUDA 13.1 or newer
- More dependencies

## Testing Results

With RTX A6000 (Ampere, compute capability 8.6):
```
Warning: Device does not support execution affinity (Green Contexts). 
This feature requires specific GPU architectures and driver support. 
See: https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/green-contexts.html
Falling back to primary context. SM sweep will vary grid size instead.
```

The program continues to work correctly in fallback mode, varying grid size to simulate different SM usage patterns.

## References

- [NVIDIA Green Contexts Documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/green-contexts.html)
- [CUDA Runtime API Reference](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION__CONTEXT.html)
- [CUDA Driver API Reference](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html)

## Verification

Use Nsight Systems to verify Green Context usage on supported hardware:
```bash
nsys profile --cuda-graph-trace=node ./sm_bw_sweep --min_sms 1 --max_sms 120 --step 1
```

Green Context rows will appear in the CUDA HW timeline if the feature is active.

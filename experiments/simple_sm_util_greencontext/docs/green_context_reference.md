# CUDA Green Contexts Reference Guide

**Source:** [CUDA Driver API v13.0.0 - Green Contexts](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html)

---

## Overview

**Green Contexts** are a lightweight alternative to traditional CUDA contexts that enable **spatial partitioning of GPU resources**. Unlike regular contexts, green contexts allow developers to:
- Explicitly provision and partition GPU resources (primarily SMs - Streaming Multiprocessors)
- Target specific resource partitions while using the same CUDA programming model (streams, kernels, etc.)
- Create multiple isolated execution environments on the same GPU

### Key Concept
Green contexts represent **distinct spatial partitions of the GPU**, allowing you to restrict kernel execution to specific subsets of SMs.

---

## Main Workflow (4 Steps)

### 1. **Get Initial Resources**
Start with the full device resources using `cuDeviceGetDevResource()`
- Currently only `CU_DEV_RESOURCE_TYPE_SM` is supported
- Returns `CUdevResource` containing SM information

### 2. **Partition Resources**
Split resources into partitions using `cuDevSmResourceSplitByCount()`
- Specify minimum SM count per partition
- Respects architecture-specific alignment requirements
- Returns multiple disjoint, symmetrical partitions

### 3. **Generate Descriptor**
Create a resource descriptor using `cuDevResourceGenerateDesc()`
- Encapsulates configured resources
- Required for green context creation

### 4. **Create Green Context**
Instantiate the green context using `cuGreenCtxCreate()`
- Provisions resources according to descriptor
- Returns `CUgreenCtx` handle

---

## Core Data Types

### Resource Types
- **`CUdevResourceType`**: Enum for resource types
  - `CU_DEV_RESOURCE_TYPE_INVALID = 0`
  - `CU_DEV_RESOURCE_TYPE_SM = 1`: Streaming multiprocessor resources

### Key Structures
- **`CUdevResource`**: Represents device resources (SMs)
- **`CUdevSmResource`**: SM-specific resource information
- **`CUdevResourceDesc`**: Opaque descriptor handle encapsulating resources
- **`CUgreenCtx`**: Green context handle

---

## Essential API Functions

### Resource Management

#### `cuDeviceGetDevResource()`
```c
CUresult cuDeviceGetDevResource(
    CUdevice device,
    CUdevResource* resource,
    CUdevResourceType type
)
```
- Gets all available resources for a device
- Starting point for resource partitioning
- Query `minSmPartitionSize` and `smCoscheduledAlignment` fields

#### `cuDevSmResourceSplitByCount()`
```c
CUresult cuDevSmResourceSplitByCount(
    CUdevResource* result,      // Output array of partitions (NULL to query count)
    unsigned int* nbGroups,     // In/out: desired/actual partition count
    const CUdevResource* input, // SM resource to split
    CUdevResource* remaining,   // Remaining SMs (if uneven split)
    unsigned int useFlags,      // Partitioning flags
    unsigned int minCount       // Minimum SMs per partition
)
```

**Important Constraints:**
- Creates **disjoint, symmetrical partitions**
- Must respect minimum SM count and alignment requirements
- Can return fewer groups than requested due to architectural constraints
- **Output partitions cannot be re-split** without creating new green context

**Supported Flags:**
- `CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING`: Finer-grained partitions, sacrifices advanced features
- `CU_DEV_SM_RESOURCE_SPLIT_MAX_POTENTIAL_CLUSTER_SIZE`: For Compute Capability 9.0+, enables maximal thread clusters

#### `cuDevResourceGenerateDesc()`
```c
CUresult cuDevResourceGenerateDesc(
    CUdevResourceDesc* phDesc,
    CUdevResource* resources,
    unsigned int nbResources
)
```
- Combines resources into a descriptor
- All resources must be from the same device
- SM resources must come from the same split API call

### Green Context Lifecycle

#### `cuGreenCtxCreate()`
```c
CUresult cuGreenCtxCreate(
    CUgreenCtx* phCtx,
    CUdevResourceDesc desc,
    CUdevice dev,
    unsigned int flags
)
```
- Creates green context with specified resources
- Retains device's primary context (released on destroy)
- **Required flag:** `CU_GREEN_CTX_DEFAULT_STREAM`
- Does NOT set context current automatically

#### `cuGreenCtxDestroy()`
```c
CUresult cuGreenCtxDestroy(CUgreenCtx hCtx)
```
- Destroys green context and releases resources
- Does NOT destroy streams (must be destroyed manually to avoid leaks)
- Subsequent stream operations return `CUDA_ERROR_CONTEXT_IS_DESTROYED`

### Context Conversion

#### `cuCtxFromGreenCtx()`
```c
CUresult cuCtxFromGreenCtx(
    CUcontext* pContext,
    CUgreenCtx hCtx
)
```
- Converts green context to primary context (`CUcontext`)
- Converted context has resources of the green context
- **Must call this** before using standard CUDA APIs that accept `CUcontext`
- Required before `cuCtxSetCurrent()` or `cuCtxPushCurrent()`

### Stream Management

#### `cuGreenCtxStreamCreate()`
```c
CUresult cuGreenCtxStreamCreate(
    CUstream* phStream,
    CUgreenCtx greenCtx,
    unsigned int flags,
    int priority
)
```
- Creates stream for specific green context
- **Required flag:** `CU_STREAM_NON_BLOCKING`
- Ignores current thread context
- Priority: lower numbers = higher priority

#### `cuStreamGetGreenCtx()`
```c
CUresult cuStreamGetGreenCtx(
    CUstream hStream,
    CUgreenCtx* phCtx
)
```
- Queries which green context a stream belongs to
- Returns NULL if not associated with green context

### Synchronization

#### `cuGreenCtxRecordEvent()`
```c
CUresult cuGreenCtxRecordEvent(
    CUgreenCtx hCtx,
    CUevent hEvent
)
```
- Records event for all green context activities
- Event and context must share same primary context

#### `cuGreenCtxWaitEvent()`
```c
CUresult cuGreenCtxWaitEvent(
    CUgreenCtx hCtx,
    CUevent hEvent
)
```
- Makes green context wait for event completion
- Device-side synchronization (doesn't block CPU)
- Event can be from different context/device

### Helper Functions

#### `cuGreenCtxGetDevResource()`
- Gets resources assigned to a green context

#### `cuGreenCtxGetId()`
- Returns unique ID for green context (lifetime of program)

#### `cuCtxGetDevResource()`
- Gets resources for a regular `CUcontext`

---

## Architecture-Specific Constraints

### Minimum SM Partition Requirements

| Compute Capability | Min SMs | Alignment |
|-------------------|---------|-----------|
| 6.x (Pascal)      | 2       | Multiple of 2 |
| 7.x (Volta/Turing)| 2       | Multiple of 2 |
| 8.x (Ampere)      | 4       | Multiple of 2 |
| 9.0+ (Hopper)     | 8       | Multiple of 8 |

**Recommendation:** Always query `cuDeviceGetDevResource()` for accurate `minSmPartitionSize` and `smCoscheduledAlignment` values.

---

## Important Limitations & Caveats

### Concurrency & Forward Progress
- **No concurrency guarantee:** Even with disjoint SM partitions, kernels may not run concurrently
- **Resource contention:** HW connections (see `CUDA_DEVICE_MAX_CONNECTIONS`) can create dependencies
- **SM overflow:** Workload may use MORE SMs than provisioned (but never less) in certain scenarios:
  - **Volta+ MPS:** When `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` is used
  - **Compute 9.x with CDP:** Dynamic parallelism adds 2 shared SMs

### Thread Safety
- **Single-threaded:** Green context can be current to only ONE thread at a time
- No internal synchronization for multi-threaded access

### Platform Support
- **Not supported on 32-bit platforms**

### Re-partitioning
- Output partitions from `cuDevSmResourceSplitByCount()` **cannot be split again**
- Must create descriptor + green context before further partitioning

### Stream Cleanup
- Streams created via `cuGreenCtxStreamCreate()` are **not destroyed** automatically
- Must explicitly call `cuStreamDestroy()` before `cuGreenCtxDestroy()`
- Failure causes memory leak

---

## Typical Usage Pattern

```c
// 1. Get device resources
CUdevResource deviceResource;
cuDeviceGetDevResource(device, &deviceResource, CU_DEV_RESOURCE_TYPE_SM);

// 2. Split into partitions (e.g., 4 groups with 8 SMs each)
unsigned int numGroups = 4;
CUdevResource partitions[4];
CUdevResource remaining;
cuDevSmResourceSplitByCount(
    partitions,
    &numGroups,
    &deviceResource,
    &remaining,
    0, // flags
    8  // minCount SMs
);

// 3. Generate descriptor for one partition
CUdevResourceDesc desc;
cuDevResourceGenerateDesc(&desc, &partitions[0], 1);

// 4. Create green context
CUgreenCtx greenCtx;
cuGreenCtxCreate(&greenCtx, desc, device, CU_GREEN_CTX_DEFAULT_STREAM);

// 5. Convert to CUcontext and set current
CUcontext ctx;
cuCtxFromGreenCtx(&ctx, greenCtx);
cuCtxSetCurrent(ctx);

// 6. Create stream and launch kernels
CUstream stream;
cuGreenCtxStreamCreate(&stream, greenCtx, CU_STREAM_NON_BLOCKING, 0);
cuLaunchKernel(..., stream, ...);

// 7. Cleanup
cuStreamSynchronize(stream);
cuStreamDestroy(stream);  // MUST destroy before context
cuGreenCtxDestroy(greenCtx);
```

---

## Use Case for Bandwidth Saturation Testing

For **finding minimum SMs to saturate DRAM bandwidth**:

1. **Start with device resources** covering all SMs
2. **Create multiple partitions** with varying SM counts
3. **For each partition:**
   - Create green context
   - Launch bandwidth-intensive kernel
   - Measure achieved bandwidth
4. **Identify threshold** where bandwidth plateaus

### Strategy
- Use `cuDevSmResourceSplitByCount()` to test: 8, 16, 24, 32... SMs
- Launch memory-bound kernels (simple reads/writes)
- Compare bandwidth vs. SM count to find saturation point

---

## Related Documentation
- [CUDA Driver API - Context Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html)
- [CUDA Driver API - Stream Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html)
- [CUDA C Programming Guide - Multi-Process Service](https://docs.nvidia.com/cuda/mps/index.html)

---

## Version Notes
- **Introduced in:** CUDA 13.0
- **Minimum Compute Capability:** 6.0 (Pascal)
- **Platform:** 64-bit only

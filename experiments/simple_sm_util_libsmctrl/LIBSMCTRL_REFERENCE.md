# libsmctrl Reference Guide

## Overview

**libsmctrl** is a library for controlling SM (Streaming Multiprocessor) / TPC (Thread Processing Cluster) masks on NVIDIA CUDA GPU launches. It enables **intra-context hardware compute partitioning** by restricting kernel execution to specific subsets of SMs.

**Key Insight:** libsmctrl exploits preexisting debug logic in the CUDA driver library to inject SM masks into kernel launches without requiring special privileges or GPU MIG (Multi-Instance GPU) mode.

---

## Architecture & GPU Hierarchy

### TPC vs SM
- **TPC (Thread Processing Cluster)**: A hardware grouping unit
- **SM (Streaming Multiprocessor)**: The actual compute unit that executes threads
- **Relationship**: 
  - Pascal and older (< sm_60, except P100): **1 SM per TPC**
  - Volta and newer (≥ sm_60): **2 SMs per TPC**
  - P100 (sm_60): **2 SMs per TPC** (exception)

### Why Mask TPCs, Not SMs?
libsmctrl masks at the **TPC level** because:
1. NVIDIA's QMD (Queue Management Descriptor) / TMD (Task Management Descriptor) structures store TPC masks, not SM masks
2. TPC masking is the granularity exposed by NVIDIA's hardware debug interfaces
3. On modern GPUs (2 SMs/TPC), masking a TPC disables both SMs within it

---

## How libsmctrl Works

### Three Masking Mechanisms

libsmctrl uses three different mechanisms depending on CUDA version:

| CUDA Version | Global Mask | Stream Mask | Next Mask |
|--------------|-------------|-------------|-----------|
| 8.0 - 12.8   | TMD/QMD Hook | stream struct | TMD/QMD Hook |
| 6.5 - 7.5    | TMD/QMD Hook | N/A | TMD/QMD Hook |

### 1. **TMD/QMD Hook Mechanism** (Global & Next Masks)

**Concept:** Intercept the Task/Queue Management Descriptor just before it's uploaded to GPU.

**Implementation:**
```c
// Register a callback that hooks into CUDA's internal launch path
cuGetExportTable(&tbl_base, &callback_funcs_id);
subscribe(&my_hndl, control_callback_v2, NULL);
enable(1, my_hndl, QMD_DOMAIN, QMD_PRE_UPLOAD);
```

**In the callback:**
```c
static void control_callback_v2(void *ukwn, int domain, int cbid, const void *in_params) {
    // Extract pointer to TMD from in_params
    void* tmd = *((void**)in_params + 4);
    
    // Determine TMD version and mask location
    uint8_t tmd_ver = *(uint8_t*)(tmd + 72);
    
    if (tmd_ver >= 0x40) {
        // Hopper+ (supports >64 TPCs via 128-bit mask)
        lower_ptr = tmd + 304;
        upper_ptr = tmd + 308;
    } else if (tmd_ver >= 0x16) {
        // Kepler V2+ (64-bit mask)
        lower_ptr = tmd + 84;
        upper_ptr = tmd + 88;
    }
    
    // Apply mask (next > stream > global precedence)
    if (g_next_sm_mask) {
        *lower_ptr = (uint32_t)g_next_sm_mask;
        *upper_ptr = (uint32_t)(g_next_sm_mask >> 32);
        g_next_sm_mask = 0;  // Clear after use
    } else if (!*lower_ptr && !*upper_ptr) {
        // Only apply global if no stream mask set
        *lower_ptr = (uint32_t)g_sm_mask;
        *upper_ptr = (uint32_t)(g_sm_mask >> 32);
    }
}
```

**Key Points:**
- Hooks into CUDA's debug callback infrastructure (originally for CUPTI)
- Modifies TMD/QMD in-flight before GPU upload
- Works across CUDA 6.5 - 12.8
- Thread-safe via `__thread` storage for next mask

### 2. **Stream Struct Mechanism** (Stream Mask)

**Concept:** Directly modify CUDA's internal stream data structure where it stores the TPC mask.

**Implementation:**
```c
int libsmctrl_set_stream_mask(void* stream, uint64_t mask) {
    char* stream_struct_base = *(char**)stream;
    
    // Offset varies by CUDA version and architecture
    int ver;
    cuDriverGetVersion(&ver);
    
    switch (ver) {
    case 12020:  // CUDA 12.2
        hw_mask_v2 = (void*)(stream_struct_base + 0x4e4);  // x86_64
        break;
    // ... many version-specific offsets ...
    }
    
    // Write mask to stream struct
    if (hw_mask_v2) {  // CUDA 12.0+
        hw_mask_v2->enabled = 0x80000000;
        hw_mask_v2->mask[0] = mask;
        hw_mask_v2->mask[1] = mask >> 32;
        hw_mask_v2->mask[2] = mask >> 64;  // For >64 TPCs
        hw_mask_v2->mask[3] = mask >> 96;
    } else {  // CUDA 8.0 - 11.8
        hw_mask->lower = (uint32_t)mask;
        hw_mask->upper = (uint32_t)(mask >> 32);
    }
}
```

**Key Points:**
- Directly patches CUDA's internal stream structure
- **Fragile:** Requires hardcoded offsets for each CUDA version
- CUDA applies this mask when building the QMD/TMD
- Persists across all launches on that stream until changed
- Not supported on CUDA 6.5-7.5

---

## API Functions

### Core Partitioning Functions

#### `libsmctrl_set_global_mask(uint64_t mask)`
Set default TPC mask for **all** kernel launches (including CUDA-internal ones).
- **Scope:** Process-wide, all streams
- **Supported:** CUDA 6.5 - 12.8
- **Precedence:** Lowest (overridden by stream and next masks)

```c
// Disable all TPCs except TPC 0
libsmctrl_set_global_mask(~0x1ull);
```

#### `libsmctrl_set_stream_mask(void* stream, uint64_t mask)`
Set TPC mask for all kernels launched on a specific stream.
- **Scope:** Per-stream, persistent
- **Supported:** CUDA 8.0 - 12.8
- **Precedence:** Medium (overrides global, overridden by next)

```c
cudaStream_t stream;
cudaStreamCreate(&stream);
// Only use TPCs 2, 3, 4
libsmctrl_set_stream_mask(stream, ~0b00111100ull);
```

#### `libsmctrl_set_next_mask(uint64_t mask)`
Set TPC mask for **only the next** kernel launch from the calling CPU thread.
- **Scope:** Thread-local, single-use
- **Supported:** CUDA 6.5 - 12.8
- **Precedence:** Highest (overrides everything)

```c
// Next kernel uses only TPC 0
libsmctrl_set_next_mask(~0x1ull);
kernel<<<blocks, threads>>>();  // Masked
kernel<<<blocks, threads>>>();  // NOT masked (reset after first use)
```

#### `libsmctrl_set_stream_mask_ext(void* stream, uint128_t mask)`
Extended version supporting GPUs with >64 TPCs (128-bit mask).

```c
uint128_t mask = ~((uint128_t)0x3 << 65);  // Enable TPCs 65-66
libsmctrl_set_stream_mask_ext(stream, mask);
```

### Informational Functions

#### `libsmctrl_get_tpc_info_cuda(uint32_t* num_tpcs, int cuda_dev)`
Get total number of TPCs on a device. **No special requirements.**

```c
uint32_t num_tpcs;
libsmctrl_get_tpc_info_cuda(&num_tpcs, 0);
printf("Device 0 has %u TPCs\n", num_tpcs);
```

#### `libsmctrl_get_gpc_info(uint32_t* num_gpcs, uint64_t** tpcs_for_gpc, int dev)`
Get GPC (Graphics Processing Cluster) topology. **Requires nvdebug kernel module.**

```c
uint32_t num_gpcs;
uint64_t* tpcs_per_gpc;
libsmctrl_get_gpc_info(&num_gpcs, &tpcs_per_gpc, 0);
```

### Helper Functions

#### `libsmctrl_make_mask(uint64_t* result, uint32_t low, uint32_t high_exclusive)`
Create a mask that enables TPCs in range `[low, high_exclusive)`.

```c
uint64_t mask;
// Enable TPCs 5-9 (disable all others)
libsmctrl_make_mask(&mask, 5, 10);
libsmctrl_set_stream_mask(stream, mask);
```

**Equivalent to:**
```c
uint64_t mask = 0;
for (int i = 5; i < 10; i++) {
    mask |= (1ull << i);
}
mask = ~mask;  // Invert: set bit = disabled TPC
```

#### `libsmctrl_validate_stream_mask(void* stream, int low, int high, bool echo)`
Validate that stream mask actually restricts execution to specified TPC range.

```c
libsmctrl_set_stream_mask(stream, mask);
int result = libsmctrl_validate_stream_mask(stream, 5, 10, true);
// result == 0: validation passed
// result == -1: validation failed
```

---

## Understanding Bitmasks

### Critical Concept: **Set Bit = DISABLED TPC**

This is **inverted** from typical intuition. A `1` bit **disables** that TPC.

```c
// Disable TPC 0
libsmctrl_set_next_mask(0x1);

// Enable ONLY TPC 0 (disable all others)
libsmctrl_set_global_mask(~0x1ull);

// Enable TPCs 0-7 (disable 8+)
libsmctrl_set_stream_mask(stream, ~0xFFull);

// Enable TPCs 2, 3, 4, disable all others
//   0b00011100 = 0x1C enables TPCs 2-4
//   ~0x1Cull disables all except 2-4
libsmctrl_set_stream_mask(stream, ~0b00011100ull);
```

### Important: Use 64-bit Literals!

```c
// WRONG - may only affect lower 32 bits
libsmctrl_set_global_mask(~0x1);

// CORRECT - 64-bit literal
libsmctrl_set_global_mask(~0x1ull);
```

### Example: Enable First N TPCs

```c
uint64_t enable_first_n_tpcs(int n) {
    uint64_t mask = (1ull << n) - 1;  // Set bits 0 to n-1
    return ~mask;  // Invert to disable all others
}

// Enable first 16 TPCs
libsmctrl_set_global_mask(enable_first_n_tpcs(16));
```

---

## Hardware Compatibility

### Supported

| GPU Architecture | Compute Capability | SMs per TPC | Note |
|---|---|---|---|
| Kepler V2+ | sm_35 - sm_37 | 1 | First supported architecture |
| Maxwell | sm_50 - sm_53 | 1 | |
| Pascal (except P100) | sm_61 - sm_62 | 1 | |
| Pascal P100 | sm_60 | 2 | Exception! |
| Volta | sm_70 - sm_72 | 2 | |
| Turing | sm_75 | 2 | |
| Ampere | sm_80 - sm_87 | 2 | |
| Ada Lovelace | sm_89 | 2 | |
| Hopper | sm_90 | 2 | 128-bit mask support |

### Not Supported
- Compute capability < 3.5 (Kepler V1 and older)
- Anything without TPC masking support in TMD

### Check Support Programmatically

```c
uint32_t num_tpcs;
int result = libsmctrl_get_tpc_info_cuda(&num_tpcs, 0);
if (result == ENOTSUP) {
    printf("SM masking not supported on this GPU\n");
}
```

---

## Practical Usage Patterns

### Pattern 1: Limit to N TPCs

```c
uint32_t total_tpcs;
libsmctrl_get_tpc_info_cuda(&total_tpcs, 0);

// Use only first 8 TPCs
uint64_t mask;
libsmctrl_make_mask(&mask, 0, 8);
libsmctrl_set_global_mask(mask);
```

### Pattern 2: Sweep SM Count to Find Bandwidth Saturation

```c
cudaStream_t stream;
cudaStreamCreate(&stream);

for (int num_tpcs = 1; num_tpcs <= total_tpcs; num_tpcs++) {
    uint64_t mask;
    libsmctrl_make_mask(&mask, 0, num_tpcs);
    libsmctrl_set_stream_mask(stream, mask);
    
    // Launch bandwidth kernel
    bandwidth_kernel<<<blocks, threads, 0, stream>>>(...);
    cudaStreamSynchronize(stream);
    
    // Measure achieved bandwidth
    float bw = measure_bandwidth();
    printf("TPCs: %d, BW: %.2f GB/s\n", num_tpcs, bw);
}
```

### Pattern 3: Isolate Work on Disjoint TPCs

```c
cudaStream_t stream_a, stream_b;
cudaStreamCreate(&stream_a);
cudaStreamCreate(&stream_b);

// Stream A: TPCs 0-31
libsmctrl_make_mask(&mask_a, 0, 32);
libsmctrl_set_stream_mask(stream_a, mask_a);

// Stream B: TPCs 32-63
libsmctrl_make_mask(&mask_b, 32, 64);
libsmctrl_set_stream_mask(stream_b, mask_b);

// These won't interfere with each other
latency_kernel<<<blocks, threads, 0, stream_a>>>(...);
throughput_kernel<<<blocks, threads, 0, stream_b>>>(...);
```

### Pattern 4: Override for Single Critical Kernel

```c
// All kernels use half the GPU
libsmctrl_set_global_mask(enable_first_n_tpcs(total_tpcs / 2));

// ... many kernel launches ...

// One critical kernel uses full GPU
libsmctrl_set_next_mask(0x0);  // Disable no TPCs = enable all
critical_kernel<<<blocks, threads>>>();

// Back to half GPU for subsequent launches
normal_kernel<<<blocks, threads>>>();
```

---

## Building and Linking

### Compilation Requirements
- CUDA SDK with `nvcc`
- GNU `gcc`
- Link against `-lcuda` (CUDA Driver API)

### Build libsmctrl

```bash
cd /path/to/libsmctrl
make config  # Configure for your system
make build   # Builds libsmctrl.so
```

### Link Against libsmctrl

```bash
nvcc my_app.cu -o my_app \
    -I/path/to/libsmctrl \
    -L/path/to/libsmctrl/build \
    -lsmctrl \
    -lcuda
```

Or in CMakeLists.txt:

```cmake
find_library(LIBSMCTRL smctrl HINTS /path/to/libsmctrl/build)
find_library(CUDA_DRIVER_LIB cuda)

target_include_directories(my_target PRIVATE /path/to/libsmctrl/src)
target_link_libraries(my_target PRIVATE ${LIBSMCTRL} ${CUDA_DRIVER_LIB})
```

---

## Important Limitations

### 1. **Intra-Context Only**
libsmctrl partitions within a single CUDA context. Cannot easily partition across separate processes.

### 2. **No Implicit Synchronization Prevention**
Still subject to CUDA's implicit synchronization. May need additional techniques (e.g., CUPiD^RT) to avoid cross-stream barriers.

### 3. **Version-Specific Offsets (Stream Mask)**
Stream masking relies on hardcoded offsets that differ per CUDA version. Must use correct version.

### 4. **Limited to 128 TPCs**
Even with `_ext` functions, currently limited to 128 TPCs. Some H100 variants have more.

### 5. **TPC Granularity, Not SM**
Cannot individually control SMs within a TPC. On modern GPUs (2 SMs/TPC), minimum granularity is 2 SMs.

### 6. **No Explicit TPC Selection**
Cannot choose *which* specific TPCs, only *how many* (counting from TPC 0). Implementation-defined subset for higher TPC IDs.

---

## Comparison with NVIDIA Green Contexts

| Feature | libsmctrl | Green Contexts |
|---------|-----------|----------------|
| **Introduced** | 2022 (research) | CUDA 11.4 (official) |
| **Privilege** | None | None |
| **API Level** | Driver/Runtime hack | Official Runtime API |
| **Granularity** | TPC-level | SM-level (new API) |
| **Persistence** | Per-launch or per-stream | Context-level |
| **Forward Compat** | Fragile (offsets) | Stable (official API) |
| **Hardware Support** | sm_35+ (most GPUs) | Hopper+ only (H100, Blackwell) |
| **Portability** | High (CUDA 6.5-12.8) | Low (new hardware only) |
| **Recommendation** | Use for older GPUs | Use for H100+ |

**Key Difference:** Green Contexts create **separate CUDA contexts** with dedicated SM pools. libsmctrl works **within a single context**.

---

## Testing and Validation

### Verify Masking Works

libsmctrl includes a validator that launches a kernel and checks which SMs actually executed:

```c
#include <libsmctrl.h>

cudaStream_t stream;
cudaStreamCreate(&stream);

uint64_t mask;
libsmctrl_make_mask(&mask, 5, 10);  // TPCs 5-9
libsmctrl_set_stream_mask(stream, mask);

// This will fail if masking didn't work
int result = libsmctrl_validate_stream_mask(stream, 5, 10, true);
if (result == 0) {
    printf("✓ Validation passed: only TPCs 5-9 were used\n");
} else {
    printf("✗ Validation failed: masking not working correctly\n");
}
```

### Read SM ID in Kernel

```c
__global__ void print_smid() {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int smid;
        asm("mov.u32 %0, %%smid;" : "=r"(smid));
        printf("Running on SM %d\n", smid);
    }
}
```

---

## Troubleshooting

### "Stream masking unsupported on this CUDA version"
Your CUDA version doesn't have a hardcoded offset. Either:
1. Use global or next masking instead (works on all versions)
2. Port the offset for your CUDA version (see README.md "Porting to New Architectures")

### Masking has no effect
1. Verify your GPU supports SM masking (compute capability ≥ 3.5)
2. Check that you're using 64-bit literals (`~0x1ull` not `~0x1`)
3. Validate with `libsmctrl_validate_stream_mask()`
4. Ensure you're not launching with zero blocks (mask only affects scheduling)

### Performance doesn't scale linearly with TPC count
Expected! Reasons:
- Memory bandwidth saturation (what you're trying to measure!)
- Cache interference
- Shared resource contention (L2, DRAM controllers)
- Workload characteristics (memory-bound vs compute-bound)

---

## Summary for Bandwidth Measurement

**Goal:** Find minimum TPC count to saturate DRAM read bandwidth.

**Recommended Approach:**
1. Get total TPC count: `libsmctrl_get_tpc_info_cuda()`
2. Create a bandwidth kernel (streaming reads, minimal compute)
3. Sweep TPC count from 1 to max:
   - Use `libsmctrl_set_stream_mask()` with range `[0, N)`
   - Launch kernel, measure time and compute bandwidth
   - Record results
4. Plot bandwidth vs TPC count
5. Find knee point where bandwidth saturates

**Code sketch:**
```c
for (int n = 1; n <= total_tpcs; n++) {
    uint64_t mask;
    libsmctrl_make_mask(&mask, 0, n);
    libsmctrl_set_stream_mask(stream, mask);
    
    cudaEventRecord(start, stream);
    bandwidth_kernel<<<blocks, threads, 0, stream>>>(data, size);
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    float bw_gbps = (size * sizeof(T)) / (ms * 1e6);
    
    printf("%d,%f\n", n, bw_gbps);
}
```

**Expected Result:** Bandwidth increases with TPC count until memory subsystem saturates, then plateaus.

---

## Citation

If using libsmctrl in research, cite:

> J. Bakita and J. H. Anderson, "Hardware Compute Partitioning on NVIDIA GPUs",  
> Proceedings of the 29th IEEE Real-Time and Embedded Technology and Applications Symposium,  
> pp. 54-66, May 2023.

---

## References

- **Paper:** https://www.cs.unc.edu/~jbakita/rtas23.pdf
- **Original Repository:** http://rtsrv.cs.unc.edu/cgit/cgit.cgi/libsmctrl.git/
- **BulletServe Fork:** https://github.com/zejia-lin/BulletServe (supports CUDA ≤ 12.2)

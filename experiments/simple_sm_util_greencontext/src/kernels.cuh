#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// =============================================================================
// Read Bandwidth Kernel
// =============================================================================
// 
// Design:
// - Each thread reads vectorized data (4x uint32 = 16 bytes per load)
// - Accumulates into a register checksum (XOR + addition mix)
// - Minimal sink write: 1 uint64 per block to prevent DCE
//
// Anti-DCE Strategy:
// - All loads accumulate into checksum register
// - Block-level reduction writes ONE value per block
// - Sink traffic: (num_blocks * 8 bytes) << read traffic
//
// Access Pattern:
// - Streaming: stride = vector width, no reuse
// - Coalesced: consecutive threads access consecutive 16B chunks
//
// =============================================================================

// Vector load type: 16 bytes (4x uint32)
struct alignas(16) Vec4U32 {
    uint32_t x, y, z, w;
};

// Kernel: Read-only bandwidth test with minimal sink write
__global__ void bandwidth_read_kernel(
    const Vec4U32* __restrict__ input,  // Input buffer (read-only)
    uint64_t* __restrict__ sink,         // Sink buffer (1 value per block)
    size_t num_elements,                 // Total Vec4U32 elements to process
    int iters_per_thread                 // Iterations per thread
) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t grid_stride = gridDim.x * blockDim.x;
    
    // Accumulator: prevent DCE by mixing XOR and addition
    uint64_t checksum = 0;
    
    // Each thread processes multiple elements with stride
    #pragma unroll 4
    for (int iter = 0; iter < iters_per_thread; ++iter) {
        size_t idx = tid + iter * grid_stride;
        
        if (idx < num_elements) {
            // Vectorized 16B load
            Vec4U32 data = input[idx];
            
            // Accumulate to checksum (mix XOR and ADD to prevent compiler tricks)
            checksum ^= data.x;
            checksum += data.y;
            checksum ^= data.z;
            checksum += data.w;
        }
    }
    
    // Block-level reduction: accumulate within block
    __shared__ uint64_t block_checksum[256];  // Max 1024 threads per block
    block_checksum[threadIdx.x] = checksum;
    __syncthreads();
    
    // Simple reduction (thread 0 accumulates)
    if (threadIdx.x == 0) {
        uint64_t sum = 0;
        for (int i = 0; i < blockDim.x; ++i) {
            sum ^= block_checksum[i];
        }
        
        // Minimal sink write: 1 uint64 per block
        sink[blockIdx.x] = sum;
    }
}

// =============================================================================
// Kernel Launch Helper
// =============================================================================

inline void launch_bandwidth_kernel(
    cudaStream_t stream,
    const Vec4U32* input,
    uint64_t* sink,
    size_t total_bytes_to_read,
    int threads_per_block,
    int num_sms
) {
    // Calculate grid dimensions
    size_t num_vec4_elements = total_bytes_to_read / sizeof(Vec4U32);
    size_t threads_per_sm = threads_per_block * 2;  // 2 blocks per SM for occupancy
    size_t total_threads = num_sms * threads_per_sm;
    size_t num_blocks = total_threads / threads_per_block;
    
    // Calculate iterations per thread to cover all data
    int iters_per_thread = (num_vec4_elements + total_threads - 1) / total_threads;
    
    // Launch kernel
    bandwidth_read_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        input,
        sink,
        num_vec4_elements,
        iters_per_thread
    );
}

// =============================================================================
// Host Helper: Allocate and initialize buffers
// =============================================================================

inline void allocate_buffers(
    Vec4U32** d_input,
    uint64_t** d_sink,
    size_t total_bytes_to_read,
    size_t max_blocks
) {
    // Allocate input buffer
    size_t input_size = total_bytes_to_read;
    cudaMalloc((void**)d_input, input_size);
    
    // Initialize with dummy pattern (optional, for debugging)
    cudaMemset(*d_input, 0xAB, input_size);
    
    // Allocate sink buffer (1 uint64 per block)
    size_t sink_size = max_blocks * sizeof(uint64_t);
    cudaMalloc((void**)d_sink, sink_size);
    cudaMemset(*d_sink, 0, sink_size);
}

inline void free_buffers(Vec4U32* d_input, uint64_t* d_sink) {
    cudaFree(d_input);
    cudaFree(d_sink);
}

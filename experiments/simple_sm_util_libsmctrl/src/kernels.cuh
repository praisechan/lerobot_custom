#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

// Read-bandwidth kernel with vectorized loads and checksum to prevent DCE
// Each thread reads data in a streaming pattern, accumulates into a checksum,
// and writes one value per block to sink buffer to prevent dead-code elimination.
//
// Key design choices:
// - float4 loads = 16 bytes per load instruction
// - Loop unrolling (4x) for ILP
// - XOR-based checksum mixing to prevent compiler optimization
// - One 8-byte write per block (negligible compared to read traffic)
//
// Parameters:
//   data: input buffer to read from (must be >= total_bytes)
//   sink: output buffer for checksum (size >= gridDim.x * sizeof(uint64_t))
//   total_bytes: total bytes to read across all threads
//   iters: number of iterations per thread (derived from total_bytes)
template <int UNROLL = 4>
__global__ void read_bandwidth_kernel(const float4* __restrict__ data,
                                       uint64_t* __restrict__ sink,
                                       size_t total_bytes,
                                       int iters) {
    // Global thread ID
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;
    
    // Each iteration reads UNROLL vectors of 16 bytes
    const int stride = total_threads;
    
    // Checksum accumulator (prevents DCE)
    uint64_t checksum = 0;
    
    // Unrolled loop for better memory-level parallelism
    #pragma unroll
    for (int iter = 0; iter < iters; ++iter) {
        int idx = tid + iter * stride;
        
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            int offset = idx + u * stride * iters;
            
            // Vectorized 16B load
            float4 val = data[offset];
            
            // Mix into checksum to prevent optimization
            // Use pointer aliasing to convert float4 to uint64_t contributions
            const uint32_t* words = reinterpret_cast<const uint32_t*>(&val);
            checksum ^= (static_cast<uint64_t>(words[0]) << 0)  ^
                        (static_cast<uint64_t>(words[1]) << 16) ^
                        (static_cast<uint64_t>(words[2]) << 32) ^
                        (static_cast<uint64_t>(words[3]) << 48);
        }
    }
    
    // Reduce checksum within block to minimize write traffic
    // Use shared memory reduction
    __shared__ uint64_t block_checksum;
    
    if (threadIdx.x == 0) {
        block_checksum = 0;
    }
    __syncthreads();
    
    // Atomic XOR into block checksum
    atomicXor(reinterpret_cast<unsigned long long*>(&block_checksum),
              static_cast<unsigned long long>(checksum));
    __syncthreads();
    
    // One thread per block writes to sink (8 bytes per block)
    if (threadIdx.x == 0) {
        sink[blockIdx.x] = block_checksum;
    }
}

// Utility kernel to initialize data buffer with non-zero pattern
__global__ void init_data(float4* data, size_t num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        // Simple pattern: use index-based values
        float base = static_cast<float>(idx);
        data[idx] = make_float4(base, base + 0.25f, base + 0.5f, base + 0.75f);
    }
}

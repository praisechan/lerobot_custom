#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// =============================================================================
// Vector Load Type for Memory Copy Kernel
// =============================================================================

struct alignas(16) Vec4U32 {
    uint32_t x, y, z, w;
};

// =============================================================================
// Memory Copy Kernel (Memory-Bound Workload)
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
// GEMM Kernel (Compute-Bound Workload)
// =============================================================================
//
// Simple tiled matrix multiplication: C = A * B
// - A: [M x K], B: [K x N], C: [M x N]
// - Uses shared memory for tile caching
// - FP32 operations for compute intensity
//
// Design:
// - Each thread block computes a TILE_SIZE x TILE_SIZE tile of C
// - Loads tiles from A and B into shared memory
// - Performs accumulation in registers
// - Writes result to global memory
//
// =============================================================================

#define TILE_SIZE 32

__global__ void gemm_kernel(
    const float* __restrict__ A,  // [M x K]
    const float* __restrict__ B,  // [K x N]
    float* __restrict__ C,        // [M x N]
    int M,
    int N,
    int K
) {
    // Block indices
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Thread indices within block
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Compute row and column of C that this thread computes
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    // Shared memory for tiles
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    // Accumulator for the dot product
    float sum = 0.0f;
    
    // Loop over tiles of A and B required to compute C tile
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < numTiles; ++t) {
        // Load tile of A into shared memory
        int aCol = t * TILE_SIZE + tx;
        int aRow = row;
        if (aRow < M && aCol < K) {
            As[ty][tx] = A[aRow * K + aCol];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        // Load tile of B into shared memory
        int bRow = t * TILE_SIZE + ty;
        int bCol = col;
        if (bRow < K && bCol < N) {
            Bs[ty][tx] = B[bRow * N + bCol];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product for this tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write result to C
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// =============================================================================
// Kernel Launch Helpers
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
    
    // Ensure at least some iterations
    if (iters_per_thread < 1) {
        iters_per_thread = 1;
    }
    
    // Launch kernel
    bandwidth_read_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        input,
        sink,
        num_vec4_elements,
        iters_per_thread
    );
}

inline void launch_gemm_kernel(
    cudaStream_t stream,
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K,
    int threads_per_block  // Not used, kept for consistency with interface
) {
    // Calculate grid dimensions for tiled GEMM
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    // Launch kernel
    gemm_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}

// =============================================================================
// Buffer Allocation Helpers
// =============================================================================

inline void allocate_mem_buffers(
    Vec4U32** d_input,
    uint64_t** d_sink,
    size_t total_bytes_to_read,
    size_t max_blocks
) {
    // Allocate input buffer
    size_t input_size = total_bytes_to_read;
    cudaMalloc((void**)d_input, input_size);
    
    // Initialize with dummy pattern
    cudaMemset(*d_input, 0xAB, input_size);
    
    // Allocate sink buffer (1 uint64 per block)
    size_t sink_size = max_blocks * sizeof(uint64_t);
    cudaMalloc((void**)d_sink, sink_size);
    cudaMemset(*d_sink, 0, sink_size);
}

inline void allocate_gemm_buffers(
    float** d_A,
    float** d_B,
    float** d_C,
    int M,
    int N,
    int K
) {
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    cudaMalloc((void**)d_A, size_A);
    cudaMalloc((void**)d_B, size_B);
    cudaMalloc((void**)d_C, size_C);
    
    // Initialize with dummy data
    cudaMemset(*d_A, 0x01, size_A);
    cudaMemset(*d_B, 0x01, size_B);
    cudaMemset(*d_C, 0, size_C);
}

inline void free_mem_buffers(Vec4U32* d_input, uint64_t* d_sink) {
    cudaFree(d_input);
    cudaFree(d_sink);
}

inline void free_gemm_buffers(float* d_A, float* d_B, float* d_C) {
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

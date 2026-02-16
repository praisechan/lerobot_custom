#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>

using namespace nvcuda;

// =============================================================================
// GEMV Kernel (Memory-Bound Workload) - FP16
// =============================================================================
//
// Computes: y = A * x (FP16 precision)
// - A: [M x N] matrix (row-major layout)
// - x: [N] vector
// - y: [M] vector
//
// Design:
// - Each block computes multiple elements of y
// - Memory-bound: reads M*N elements from A, bottlenecked by bandwidth
// - FLOPs: 2*M*N (multiply and add)
// - Arithmetic intensity: ~1 FLOPs/byte (FP16 uses half memory)
//
// =============================================================================

__global__ void gemv_kernel(
    const __half* __restrict__ A,  // [M x N] matrix
    const __half* __restrict__ x,  // [N] vector
    __half* __restrict__ y,        // [M] vector (output)
    int M,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < M) {
        // Each thread computes one element of y
        float sum = 0.0f;
        
        // Compute dot product of row idx with vector x (accumulate in FP32)
        #pragma unroll 8
        for (int j = 0; j < N; ++j) {
            sum += __half2float(A[idx * N + j]) * __half2float(x[j]);
        }
        
        // Write result (convert back to FP16)
        y[idx] = __float2half(sum);
    }
}

// =============================================================================
// GEMM Kernel (Compute-Bound Workload) - FP16 with Tensor Cores (WMMA)
// =============================================================================
//
// Tensor Core matrix multiplication: C = A * B (using WMMA API)
// - A: [M x K], B: [K x N], C: [M x N] (all FP16)
// - Uses Tensor Cores via WMMA (Warp Matrix Multiply-Accumulate)
// - WMMA tile size: 16x16x16 (M x N x K per warp operation)
//
// Design:
// - Each warp computes a 16x16 tile of C using multiple 16x16x16 WMMA ops
// - Tensor Cores perform matrix multiply at high throughput
// - Accumulation in FP16 for maximum Tensor Core utilization
//
// Requirements:
// - Compute Capability 7.0+ (Volta or later)
// - M, N, K must be multiples of 16
//
// =============================================================================

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void gemm_kernel_wmma(
    const __half* __restrict__ A,  // [M x K]
    const __half* __restrict__ B,  // [K x N]
    __half* __restrict__ C,        // [M x N]
    int M,
    int N,
    int K
) {
    // Warp and lane IDs
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
    
    // Bounds check
    if (warpM * WMMA_M >= M || warpN * WMMA_N >= N) return;
    
    // Declare fragments for WMMA operations
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> acc_frag;
    
    // Initialize accumulator to zero
    wmma::fill_fragment(acc_frag, __float2half(0.0f));
    
    // Loop over K dimension in chunks of WMMA_K
    for (int k = 0; k < K; k += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = k;
        int bRow = k;
        int bCol = warpN * WMMA_N;
        
        // Bounds check for partial tiles
        if (aRow < M && aCol < K && bRow < K && bCol < N) {
            // Load tiles into fragments
            wmma::load_matrix_sync(a_frag, A + aRow * K + aCol, K);
            wmma::load_matrix_sync(b_frag, B + bRow * N + bCol, N);
            
            // Perform Tensor Core matrix multiply-accumulate
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }
    
    // Store result (FP16 accumulator to FP16 output)
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;
    
    if (cRow < M && cCol < N) {
        wmma::store_matrix_sync(C + cRow * N + cCol, acc_frag, N, wmma::mem_row_major);
    }
}

// =============================================================================
// Kernel Launch Helpers
// =============================================================================

inline void launch_gemv_kernel(
    cudaStream_t stream,
    const __half* A,
    const __half* x,
    __half* y,
    int M,
    int N,
    int threads_per_block = 256
) {
    // Calculate grid dimensions
    int num_blocks = (M + threads_per_block - 1) / threads_per_block;
    
    // Launch kernel
    gemv_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        A, x, y, M, N
    );
}

inline void launch_gemm_kernel(
    cudaStream_t stream,
    const __half* A,
    const __half* B,
    __half* C,
    int M,
    int N,
    int K,
    int threads_per_block  // Not used directly, using warp-based launch
) {
    // WMMA requires warps (32 threads)
    // Each warp computes one 16x16 tile
    // Block size: 128 threads = 4 warps (2x2 tile arrangement)
    dim3 block(32, 4);  // 32 threads per warp, 4 warps per block
    dim3 grid((M + WMMA_M - 1) / WMMA_M, (N + WMMA_N - 1) / WMMA_N);
    
    // Launch Tensor Core kernel
    gemm_kernel_wmma<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}

// =============================================================================
// Buffer Allocation Helpers (FP16)
// =============================================================================

inline void allocate_gemv_buffers(
    __half** d_A,
    __half** d_x,
    __half** d_y,
    int M,
    int N
) {
    size_t size_A = M * N * sizeof(__half);
    size_t size_x = N * sizeof(__half);
    size_t size_y = M * sizeof(__half);
    
    cudaMalloc((void**)d_A, size_A);
    cudaMalloc((void**)d_x, size_x);
    cudaMalloc((void**)d_y, size_y);
    
    // Initialize with dummy FP16 data (value 1.0)
    __half* h_A = new __half[M * N];
    __half* h_x = new __half[N];
    for (int i = 0; i < M * N; ++i) h_A[i] = __float2half(1.0f);
    for (int i = 0; i < N; ++i) h_x[i] = __float2half(1.0f);
    
    cudaMemcpy(*d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(*d_x, h_x, size_x, cudaMemcpyHostToDevice);
    cudaMemset(*d_y, 0, size_y);
    
    delete[] h_A;
    delete[] h_x;
}

inline void allocate_gemm_buffers(
    __half** d_A,
    __half** d_B,
    __half** d_C,
    int M,
    int N,
    int K
) {
    // Round up to multiples of 16 for WMMA requirements
    int M_padded = ((M + 15) / 16) * 16;
    int N_padded = ((N + 15) / 16) * 16;
    int K_padded = ((K + 15) / 16) * 16;
    
    size_t size_A = M_padded * K_padded * sizeof(__half);
    size_t size_B = K_padded * N_padded * sizeof(__half);
    size_t size_C = M_padded * N_padded * sizeof(__half);
    
    cudaMalloc((void**)d_A, size_A);
    cudaMalloc((void**)d_B, size_B);
    cudaMalloc((void**)d_C, size_C);
    
    // Initialize with dummy FP16 data (value 1.0)
    __half* h_A = new __half[M_padded * K_padded];
    __half* h_B = new __half[K_padded * N_padded];
    for (int i = 0; i < M_padded * K_padded; ++i) h_A[i] = __float2half(1.0f);
    for (int i = 0; i < K_padded * N_padded; ++i) h_B[i] = __float2half(1.0f);
    
    cudaMemcpy(*d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(*d_B, h_B, size_B, cudaMemcpyHostToDevice);
    cudaMemset(*d_C, 0, size_C);
    
    delete[] h_A;
    delete[] h_B;
}

inline void free_gemv_buffers(__half* d_A, __half* d_x, __half* d_y) {
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);
}

inline void free_gemm_buffers(__half* d_A, __half* d_B, __half* d_C) {
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

__global__ void fill_half_kernel(__half* data, size_t count, float value) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < count) {
        data[idx] = __float2half(value);
    }
}

inline void fill_half_buffer(cudaStream_t stream, __half* data, size_t count, float value) {
    const int threads = 256;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    fill_half_kernel<<<blocks, threads, 0, stream>>>(data, count, value);
}

__global__ void gemv_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ x,
    __half* __restrict__ y,
    int M,
    int N
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) {
        return;
    }

    float sum = 0.0f;
    for (int j = 0; j < N; ++j) {
        sum += __half2float(A[static_cast<size_t>(row) * N + j]) * __half2float(x[j]);
    }
    y[row] = __float2half(sum);
}

inline void launch_gemv_kernel(
    cudaStream_t stream,
    const __half* A,
    const __half* x,
    __half* y,
    int M,
    int N,
    int threads_per_block
) {
    const int blocks = (M + threads_per_block - 1) / threads_per_block;
    gemv_kernel<<<blocks, threads_per_block, 0, stream>>>(A, x, y, M, N);
}

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

__global__ void gemm_kernel_wmma(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ C,
    int M,
    int N,
    int K
) {
    const int lane_id = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int warp_m = blockIdx.x * blockDim.y + warp_id;
    const int warp_n = blockIdx.y;

    if (lane_id >= 32) {
        return;
    }

    const int row = warp_m * WMMA_M;
    const int col = warp_n * WMMA_N;

    if (row >= M || col >= N) {
        return;
    }

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> acc_frag;

    wmma::fill_fragment(acc_frag, __float2half(0.0f));

    for (int k = 0; k < K; k += WMMA_K) {
        wmma::load_matrix_sync(a_frag, A + static_cast<size_t>(row) * K + k, K);
        wmma::load_matrix_sync(b_frag, B + static_cast<size_t>(k) * N + col, N);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    if (row + WMMA_M <= M && col + WMMA_N <= N) {
        wmma::store_matrix_sync(C + static_cast<size_t>(row) * N + col, acc_frag, N, wmma::mem_row_major);
    }
}

inline void launch_gemm_kernel(
    cudaStream_t stream,
    const __half* A,
    const __half* B,
    __half* C,
    int M,
    int N,
    int K,
    int /*threads_per_block*/
) {
    dim3 block(32, 4, 1);
    dim3 grid((M + (WMMA_M * block.y) - 1) / (WMMA_M * block.y), (N + WMMA_N - 1) / WMMA_N, 1);
    gemm_kernel_wmma<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}

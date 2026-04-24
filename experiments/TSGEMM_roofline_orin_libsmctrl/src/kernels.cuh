#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdint.h>

using namespace nvcuda;

struct alignas(16) Vec4U32 {
    uint32_t x, y, z, w;
};

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

template <int UNROLL>
__global__ void bandwidth_read_kernel(
    const Vec4U32* __restrict__ input,
    uint64_t* __restrict__ sink,
    size_t num_elements,
    int iters_per_thread
) {
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t grid_stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    uint64_t checksum = 0;

    #pragma unroll
    for (int iter = 0; iter < iters_per_thread; ++iter) {
        size_t idx = tid + static_cast<size_t>(iter) * grid_stride;
        if (idx < num_elements) {
            Vec4U32 data = input[idx];
            checksum ^= data.x;
            checksum += data.y;
            checksum ^= data.z;
            checksum += data.w;
        }
    }

    __shared__ uint64_t block_checksum[256];
    block_checksum[threadIdx.x] = checksum;
    __syncthreads();

    if (threadIdx.x == 0) {
        uint64_t sum = 0;
        for (int i = 0; i < blockDim.x; ++i) {
            sum ^= block_checksum[i];
        }
        sink[blockIdx.x] = sum;
    }
}

inline void launch_bandwidth_kernel(
    cudaStream_t stream,
    const Vec4U32* input,
    uint64_t* sink,
    size_t total_bytes_to_read,
    int threads_per_block,
    int active_sms
) {
    const size_t num_vec4_elements = total_bytes_to_read / sizeof(Vec4U32);
    const size_t threads_per_sm = static_cast<size_t>(threads_per_block) * 2;
    const size_t total_threads = static_cast<size_t>(active_sms) * threads_per_sm;
    const size_t num_blocks = total_threads / threads_per_block;
    int iters_per_thread = static_cast<int>((num_vec4_elements + total_threads - 1) / total_threads);
    if (iters_per_thread < 1) {
        iters_per_thread = 1;
    }
    bandwidth_read_kernel<4><<<num_blocks, threads_per_block, 0, stream>>>(input, sink, num_vec4_elements,
                                                                            iters_per_thread);
}

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

__global__ void tsgemm_real_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ C,
    int M,
    int N,
    int K
) {
    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;
    const int tile_row = (blockIdx.x * blockDim.y + warp_id) * WMMA_M;
    const int tile_col = blockIdx.y * WMMA_N;

    if (lane_id >= 32 || tile_row >= M || tile_col >= N) {
        return;
    }

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> acc_frag;
    wmma::fill_fragment(acc_frag, __float2half(0.0f));

    for (int k = 0; k < K; k += WMMA_K) {
        wmma::load_matrix_sync(a_frag, A + static_cast<size_t>(tile_row) * K + k, K);
        wmma::load_matrix_sync(b_frag, B + static_cast<size_t>(k) * N + tile_col, N);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    if (tile_row + WMMA_M <= M && tile_col + WMMA_N <= N) {
        wmma::store_matrix_sync(C + static_cast<size_t>(tile_row) * N + tile_col, acc_frag, N,
                                wmma::mem_row_major);
    }
}

inline void launch_tsgemm_real_kernel(
    cudaStream_t stream,
    const __half* A,
    const __half* B,
    __half* C,
    int M,
    int N,
    int K
) {
    dim3 block(32, 4, 1);
    dim3 grid((M + WMMA_M * block.y - 1) / (WMMA_M * block.y), (N + WMMA_N - 1) / WMMA_N, 1);
    tsgemm_real_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}

__global__ void tsgemm_compute_roof_kernel(float* sink, int repeats) {
    __shared__ __half As[WMMA_M * WMMA_K];
    __shared__ __half Bs[WMMA_K * WMMA_N];

    const int linear_tid = threadIdx.y * blockDim.x + threadIdx.x;
    for (int i = linear_tid; i < WMMA_M * WMMA_K; i += blockDim.x * blockDim.y) {
        As[i] = __float2half(1.0f);
    }
    for (int i = linear_tid; i < WMMA_K * WMMA_N; i += blockDim.x * blockDim.y) {
        Bs[i] = __float2half(1.0f);
    }
    __syncthreads();

    const int lane_id = threadIdx.x;
    const int warp_id = threadIdx.y;
    if (lane_id >= 32) {
        return;
    }

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> acc_frag;

    wmma::load_matrix_sync(a_frag, As, WMMA_K);
    wmma::load_matrix_sync(b_frag, Bs, WMMA_N);
    wmma::fill_fragment(acc_frag, __float2half(0.0f));

    #pragma unroll 1
    for (int iter = 0; iter < repeats; ++iter) {
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    if (lane_id == 0) {
        sink[blockIdx.x * blockDim.y + warp_id] = __half2float(acc_frag.x[0]);
    }
}

inline void launch_tsgemm_compute_roof_kernel(
    cudaStream_t stream,
    float* sink,
    int active_sms,
    int repeats
) {
    dim3 block(32, 4, 1);
    dim3 grid(active_sms * 4, 1, 1);
    tsgemm_compute_roof_kernel<<<grid, block, 0, stream>>>(sink, repeats);
}

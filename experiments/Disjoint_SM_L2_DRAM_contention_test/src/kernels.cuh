#pragma once

#include <cuda/barrier>
#include <cuda_fp16.h>
#include <cuda/ptx>
#include <cuda_runtime.h>
#include <mma.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void init_stream_arrays_kernel(
    float4* __restrict__ a,
    float4* __restrict__ b,
    float4* __restrict__ out,
    size_t n
) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = gridDim.x * blockDim.x;

    for (size_t i = tid; i < n; i += stride) {
        float x = static_cast<float>((i & 1023) + 1) * 0.001f;
        a[i] = make_float4(x, x + 1.0f, x + 2.0f, x + 3.0f);
        b[i] = make_float4(x + 4.0f, x + 5.0f, x + 6.0f, x + 7.0f);
        out[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
}

__global__ void streaming_triad_kernel(
    const float4* __restrict__ a,
    const float4* __restrict__ b,
    float4* __restrict__ out,
    size_t n,
    float alpha
) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = gridDim.x * blockDim.x;

    for (size_t i = tid; i < n; i += stride) {
        float4 av = a[i];
        float4 bv = b[i];
        out[i] = make_float4(
            av.x + alpha * bv.x,
            av.y + alpha * bv.y,
            av.z + alpha * bv.z,
            av.w + alpha * bv.w
        );
    }
}

__global__ void l2_flush_kernel(float4* __restrict__ buf, size_t n) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = gridDim.x * blockDim.x;

    for (size_t i = tid; i < n; i += stride) {
        float4 v = buf[i];
        v.x += 1.0f;
        buf[i] = v;
    }
}

__device__ inline bool tma_elected_warp0_thread() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    if (threadIdx.x >= 32) {
        return false;
    }

    unsigned int leader_lane = 0;
    int is_leader = 0;
    asm volatile(
        "{\n\t"
        " .reg .pred p;\n\t"
        " elect.sync %0|p, %2;\n\t"
        " @p mov.s32 %1, 1;\n\t"
        "}"
        : "+r"(leader_lane), "+r"(is_leader)
        : "r"(0xffffffffu));
    return is_leader != 0;
#else
    return threadIdx.x == 0;
#endif
}

__global__ void tma_streaming_triad_kernel(
    const unsigned char* __restrict__ a_bytes,
    const unsigned char* __restrict__ b_bytes,
    float4* __restrict__ out,
    size_t n,
    size_t tile_bytes,
    int tiles_per_block,
    float alpha
) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    using barrier_t = cuda::barrier<cuda::thread_scope_block>;
    namespace ptx = cuda::ptx;

    extern __shared__ __align__(128) unsigned char smem[];

    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier_t bar;
    if (threadIdx.x == 0) {
        init(&bar, blockDim.x);
    }
    __syncthreads();

    unsigned char* smem_a = smem;
    unsigned char* smem_b = smem + tile_bytes;
    const size_t total_bytes = n * sizeof(float4);
    const size_t rounded_total_bytes = total_bytes & ~static_cast<size_t>(15);

    for (int iter = 0; iter < tiles_per_block; ++iter) {
        const size_t tile_idx = blockIdx.x + static_cast<size_t>(iter) * gridDim.x;
        const size_t offset = tile_idx * tile_bytes;
        if (offset >= rounded_total_bytes) {
            break;
        }

        size_t bytes_this_tile = tile_bytes;
        if (offset + bytes_this_tile > rounded_total_bytes) {
            bytes_this_tile = rounded_total_bytes - offset;
        }
        bytes_this_tile &= ~static_cast<size_t>(15);
        if (bytes_this_tile == 0) {
            continue;
        }

        if (tma_elected_warp0_thread()) {
            const uint32_t tx_bytes = static_cast<uint32_t>(bytes_this_tile);
            ptx::cp_async_bulk(
                ptx::space_shared,
                ptx::space_global,
                smem_a,
                a_bytes + offset,
                tx_bytes,
                cuda::device::barrier_native_handle(bar));
            cuda::device::barrier_expect_tx(bar, tx_bytes);
            ptx::cp_async_bulk(
                ptx::space_shared,
                ptx::space_global,
                smem_b,
                b_bytes + offset,
                tx_bytes,
                cuda::device::barrier_native_handle(bar));
            cuda::device::barrier_expect_tx(bar, tx_bytes);
        }

        barrier_t::arrival_token token = bar.arrive();
        bar.wait(std::move(token));

        const float4* tile_a = reinterpret_cast<const float4*>(smem_a);
        const float4* tile_b = reinterpret_cast<const float4*>(smem_b);
        const size_t base_element = offset / sizeof(float4);
        const size_t elements_this_tile = bytes_this_tile / sizeof(float4);
        for (size_t local = threadIdx.x; local < elements_this_tile; local += blockDim.x) {
            float4 av = tile_a[local];
            float4 bv = tile_b[local];
            out[base_element + local] = make_float4(
                av.x + alpha * bv.x,
                av.y + alpha * bv.y,
                av.z + alpha * bv.z,
                av.w + alpha * bv.w
            );
        }

        // Keep the next TMA writes from racing the shared-memory reads above.
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        (&bar)->~barrier_t();
    }
#else
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = gridDim.x * blockDim.x;
    const float4* a = reinterpret_cast<const float4*>(a_bytes);
    const float4* b = reinterpret_cast<const float4*>(b_bytes);
    for (size_t i = tid; i < n; i += stride) {
        float4 av = a[i];
        float4 bv = b[i];
        out[i] = make_float4(
            av.x + alpha * bv.x,
            av.y + alpha * bv.y,
            av.z + alpha * bv.z,
            av.w + alpha * bv.w
        );
    }
#endif
}

__global__ void init_gemm_kernel(
    __half* __restrict__ a,
    __half* __restrict__ b,
    float* __restrict__ c,
    int n
) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(n) * static_cast<size_t>(n);
    const size_t stride = gridDim.x * blockDim.x;

    for (size_t i = tid; i < total; i += stride) {
        float av = static_cast<float>((i % 17) + 1) * 0.001f;
        float bv = static_cast<float>((i % 13) + 1) * 0.001f;
        a[i] = __float2half(av);
        b[i] = __float2half(bv);
        c[i] = 0.0f;
    }
}

__global__ void wmma_compute_kernel(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    float* __restrict__ c,
    int n,
    int mma_repeats
) {
    const int tile_n = blockIdx.x;
    const int tile_m = blockIdx.y * blockDim.y + threadIdx.y;

    if (tile_m * WMMA_M >= n || tile_n * WMMA_N >= n) {
        return;
    }

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    for (int k = 0; k < n; k += WMMA_K) {
        const __half* a_tile = a + (tile_m * WMMA_M) * n + k;
        const __half* b_tile = b + k * n + tile_n * WMMA_N;

        wmma::load_matrix_sync(a_frag, a_tile, n);
        wmma::load_matrix_sync(b_frag, b_tile, n);

        #pragma unroll
        for (int r = 0; r < 16; ++r) {
            if (r < mma_repeats) {
                wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            }
        }
    }

    float* c_tile = c + (tile_m * WMMA_M) * n + tile_n * WMMA_N;
    wmma::store_matrix_sync(c_tile, acc_frag, n, wmma::mem_row_major);
}

inline void launch_stream_init(
    cudaStream_t stream,
    float4* a,
    float4* b,
    float4* out,
    size_t n
) {
    constexpr int tpb = 256;
    int blocks = 1024;
    init_stream_arrays_kernel<<<blocks, tpb, 0, stream>>>(a, b, out, n);
}

inline void launch_streaming_kernel(
    cudaStream_t stream,
    const float4* a,
    const float4* b,
    float4* out,
    size_t n,
    int sm_count,
    int threads_per_block,
    int blocks_per_sm,
    float alpha
) {
    int blocks = std::max(1, sm_count * blocks_per_sm);
    streaming_triad_kernel<<<blocks, threads_per_block, 0, stream>>>(a, b, out, n, alpha);
}

inline void launch_tma_memory_kernel(
    cudaStream_t stream,
    const float4* a,
    const float4* b,
    float4* out,
    size_t n,
    int sm_count,
    int threads_per_block,
    size_t tile_bytes,
    int blocks_per_sm,
    float alpha
) {
    int blocks = std::max(1, sm_count * blocks_per_sm);
    size_t total_bytes = n * sizeof(float4);
    size_t rounded_total_bytes = total_bytes & ~static_cast<size_t>(15);
    size_t num_tiles = (rounded_total_bytes + tile_bytes - 1) / tile_bytes;
    int tiles_per_block = static_cast<int>((num_tiles + blocks - 1) / blocks);
    size_t dynamic_smem_bytes = 2 * tile_bytes;

    tma_streaming_triad_kernel<<<blocks, threads_per_block, dynamic_smem_bytes, stream>>>(
        reinterpret_cast<const unsigned char*>(a),
        reinterpret_cast<const unsigned char*>(b),
        out,
        n,
        tile_bytes,
        tiles_per_block,
        alpha
    );
}

inline cudaError_t configure_tma_memory_kernel(size_t tile_bytes) {
    return cudaFuncSetAttribute(
        tma_streaming_triad_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(2 * tile_bytes));
}

inline void launch_l2_flush(
    cudaStream_t stream,
    float4* buf,
    size_t n,
    int sm_count,
    int threads_per_block,
    int blocks_per_sm
) {
    int blocks = std::max(1, sm_count * blocks_per_sm);
    l2_flush_kernel<<<blocks, threads_per_block, 0, stream>>>(buf, n);
}

inline void launch_gemm_init(
    cudaStream_t stream,
    __half* a,
    __half* b,
    float* c,
    int n
) {
    constexpr int tpb = 256;
    int blocks = std::max(1, (n * n + tpb - 1) / tpb);
    init_gemm_kernel<<<blocks, tpb, 0, stream>>>(a, b, c, n);
}

inline void launch_wmma_compute_kernel(
    cudaStream_t stream,
    const __half* a,
    const __half* b,
    float* c,
    int n,
    int mma_repeats
) {
    dim3 block(32, 8);
    dim3 grid((n + WMMA_N - 1) / WMMA_N, ((n + WMMA_M - 1) / WMMA_M + block.y - 1) / block.y);
    wmma_compute_kernel<<<grid, block, 0, stream>>>(a, b, c, n, mma_repeats);
}

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

constexpr int kVecBytes = 16;
constexpr int kUnroll = 4;

using VecType = uint4;

__global__ void read_bw_kernel(const VecType* __restrict__ src,
                               size_t num_vec,
                               uint64_t* __restrict__ sink,
                               int iters) {
  const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
  uint64_t sum = 0;

  for (int iter = 0; iter < iters; ++iter) {
    for (size_t i = tid; i < num_vec; i += stride * kUnroll) {
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        const size_t idx = i + static_cast<size_t>(u) * stride;
        if (idx < num_vec) {
          const VecType v = src[idx];
          sum += static_cast<uint64_t>(v.x) + v.y + v.z + v.w;
        }
      }
    }
  }

  if (tid < stride) {
    sink[tid] = sum;
  }
}

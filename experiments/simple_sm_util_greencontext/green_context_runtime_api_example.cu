/**
 * Green Context Example using CUDA Runtime API (CUDA 13.1+)
 * 
 * This example demonstrates the recommended approach from NVIDIA documentation:
 * https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/green-contexts.html
 * 
 * The Runtime API is higher-level and easier to use than the Driver API (cuCtxCreate_v3).
 * It provides better resource management and supports heterogeneous SM partitioning.
 * 
 * Requirements:
 * - CUDA 13.1 or later
 * - GPU with Green Context support (Hopper H100, Blackwell, or newer architectures)
 * - Driver version that supports execution affinity
 * 
 * Compile with:
 *   nvcc -o green_ctx_runtime green_context_runtime_api_example.cu -lcuda
 */

#include <cuda_runtime.h>
#include <iostream>

#define CHECK_CUDA(call) \
  do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                << cudaGetErrorString(err) << std::endl; \
      exit(EXIT_FAILURE); \
    } \
  } while (0)

__global__ void dummy_kernel(int *data) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  data[idx] = idx;
}

int main() {
  int device = 0;
  CHECK_CUDA(cudaSetDevice(device));
  
  std::cout << "=== Green Context Creation using CUDA Runtime API ===" << std::endl;
  
  // Step 1: Get available GPU SM resources
  std::cout << "\nStep 1: Getting available GPU resources..." << std::endl;
  cudaDevResource initial_SM_resources = {};
  CHECK_CUDA(cudaDeviceGetDevResource(device, &initial_SM_resources, cudaDevResourceTypeSm));
  
  std::cout << "  Total SMs available: " << initial_SM_resources.sm.smCount << std::endl;
  std::cout << "  Min SM partition size: " << initial_SM_resources.sm.minSmPartitionSize << std::endl;
  std::cout << "  SM co-scheduled alignment: " << initial_SM_resources.sm.smCoscheduledAlignment << std::endl;
  
  // Step 2: Partition SM resources into multiple groups
  std::cout << "\nStep 2: Partitioning SM resources..." << std::endl;
  
  // Example: Create 2 heterogeneous partitions (16 SMs and 8 SMs)
  // using cudaDevSmResourceSplit
  int nbGroups = 2;
  cudaDevResource split_result[2] = {{}, {}};
  cudaDevResource remainder = {};
  
  cudaDevSmResourceGroupParams group_params[2] = {
    {.smCount=16, .coscheduledSmCount=0, .preferredCoscheduledSmCount=0, .flags=0},
    {.smCount=8,  .coscheduledSmCount=0, .preferredCoscheduledSmCount=0, .flags=0}
  };
  
  cudaError_t split_err = cudaDevSmResourceSplit(
    &split_result[0],         // result array
    nbGroups,                 // number of groups
    &initial_SM_resources,    // input resource to split
    &remainder,               // remaining partition (can be nullptr)
    0,                        // flags (0 = default)
    &group_params[0]          // group parameters
  );
  
  if (split_err != cudaSuccess) {
    std::cout << "  Resource split failed: " << cudaGetErrorString(split_err) << std::endl;
    std::cout << "  This is expected if your GPU doesn't support Green Contexts." << std::endl;
    std::cout << "  Supported architectures: Hopper (H100), Blackwell and newer." << std::endl;
    return 0;  // Exit gracefully
  }
  
  std::cout << "  Created " << nbGroups << " partitions:" << std::endl;
  std::cout << "    Partition 1: " << split_result[0].sm.smCount << " SMs" << std::endl;
  std::cout << "    Partition 2: " << split_result[1].sm.smCount << " SMs" << std::endl;
  std::cout << "    Remainder: " << remainder.sm.smCount << " SMs" << std::endl;
  
  // Step 3: Create resource descriptors
  std::cout << "\nStep 3: Creating resource descriptors..." << std::endl;
  
  cudaDevResourceDesc_t resource_desc1 = {};
  cudaDevResourceDesc_t resource_desc2 = {};
  
  CHECK_CUDA(cudaDevResourceGenerateDesc(&resource_desc1, &split_result[0], 1));
  CHECK_CUDA(cudaDevResourceGenerateDesc(&resource_desc2, &split_result[1], 1));
  
  std::cout << "  Resource descriptors created successfully." << std::endl;
  
  // Step 4: Create green contexts
  std::cout << "\nStep 4: Creating green contexts..." << std::endl;
  
  cudaExecutionContext_t green_ctx1 = {};
  cudaExecutionContext_t green_ctx2 = {};
  
  CHECK_CUDA(cudaGreenCtxCreate(&green_ctx1, resource_desc1, device, 0));
  CHECK_CUDA(cudaGreenCtxCreate(&green_ctx2, resource_desc2, device, 0));
  
  std::cout << "  Green contexts created successfully." << std::endl;
  
  // Step 5: Create streams for each green context
  std::cout << "\nStep 5: Creating streams for green contexts..." << std::endl;
  
  cudaStream_t stream1, stream2;
  int priority = 0;
  
  CHECK_CUDA(cudaExecutionCtxStreamCreate(&stream1, green_ctx1, cudaStreamDefault, priority));
  CHECK_CUDA(cudaExecutionCtxStreamCreate(&stream2, green_ctx2, cudaStreamDefault, priority));
  
  std::cout << "  Streams created successfully." << std::endl;
  
  // Step 6: Launch kernels on green context streams
  std::cout << "\nStep 6: Launching kernels on green context streams..." << std::endl;
  
  int *d_data1, *d_data2;
  size_t data_size = 1024 * sizeof(int);
  
  CHECK_CUDA(cudaMalloc(&d_data1, data_size));
  CHECK_CUDA(cudaMalloc(&d_data2, data_size));
  
  // Launch kernel on green_ctx1 stream (will use 16 SMs)
  dummy_kernel<<<16, 64, 0, stream1>>>(d_data1);
  
  // Launch kernel on green_ctx2 stream (will use 8 SMs)
  dummy_kernel<<<8, 64, 0, stream2>>>(d_data2);
  
  CHECK_CUDA(cudaStreamSynchronize(stream1));
  CHECK_CUDA(cudaStreamSynchronize(stream2));
  
  std::cout << "  Kernels executed successfully." << std::endl;
  
  // Cleanup
  std::cout << "\nCleaning up..." << std::endl;
  
  CHECK_CUDA(cudaFree(d_data1));
  CHECK_CUDA(cudaFree(d_data2));
  CHECK_CUDA(cudaStreamDestroy(stream1));
  CHECK_CUDA(cudaStreamDestroy(stream2));
  CHECK_CUDA(cudaExecutionCtxDestroy(green_ctx1));
  CHECK_CUDA(cudaExecutionCtxDestroy(green_ctx2));
  
  std::cout << "\n=== Green Context example completed successfully ===" << std::endl;
  std::cout << "\nNote: Use Nsight Systems to verify green context usage:" << std::endl;
  std::cout << "  nsys profile --cuda-graph-trace=node ./green_ctx_runtime" << std::endl;
  
  return 0;
}

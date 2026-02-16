#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

#include "kernels.cuh"

// =============================================================================
// Error Checking Macros
// =============================================================================

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CHECK_CU(call) \
    do { \
        CUresult err = call; \
        if (err != CUDA_SUCCESS) { \
            const char* errStr; \
            cuGetErrorString(err, &errStr); \
            fprintf(stderr, "CUDA Driver Error at %s:%d - %s\n", __FILE__, __LINE__, errStr); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// =============================================================================
// Configuration Structure
// =============================================================================

struct BenchmarkConfig {
    int min_sms;
    int max_sms;
    int gemv_M;
    int gemv_N;
    int gemm_size;
    int num_iters;  // Unified iteration count for both kernels
    int tpb_gemv;
    int tpb_gemm;
    int num_repeats;
    bool perfect_sync;  // Use device sync for perfect alignment (slower but guaranteed)
    std::string csv_path;
    
    // Defaults
    BenchmarkConfig() :
        min_sms(48),
        max_sms(48),  // Default to all SMs on DGX Spark
        gemv_M(32768),  // Tuned for ~68ms per iteration (FP16)
        gemv_N(32768),
        gemm_size(4096),  // Increased to ~233ms per iteration (FP16 with Tensor Cores)
        num_iters(8),  // Unified iteration count for both GEMV and GEMM
        tpb_gemv(256),
        tpb_gemm(256),  // Not used directly, WMMA uses warp-based execution
        num_repeats(5),
        perfect_sync(false),  // Use event-based sync by default (faster, slight misalignment possible)
        csv_path("./results.csv")
    {}
};

// =============================================================================
// Argument Parsing
// =============================================================================

void print_usage(const char* prog_name) {
    printf("Usage: %s [options]\n", prog_name);
    printf("Options:\n");
    printf("  --min_sms <N>       Minimum SM count (default: 8)\n");
    printf("  --max_sms <N>       Maximum SM count (default: device max)\n");
    printf("  --gemv_M <N>        GEMV matrix rows (default: 16384)\n");
    printf("  --gemv_N <N>        GEMV matrix cols (default: 16384)\n");
    printf("  --gemm_size <N>     Matrix dimension for GEMM (default: 2048)\n");
    printf("  --num_iters <N>     Number of iterations for both kernels (default: 8)\n");
    printf("  --tpb_gemv <N>      Threads per block for GEMV (default: 256)\n");
    printf("  --tpb_gemm <N>      Threads per block for GEMM (default: 256)\n");
    printf("  --repeats <N>       Measurement repeats (default: 5)\n");
    printf("  --perfect_sync      Use device sync for perfect kernel alignment (slower)\n");
    printf("  --csv <path>        Output CSV path (default: ./results.csv)\n");
    printf("  --help              Print usage\n");
}

BenchmarkConfig parse_args(int argc, char** argv) {
    BenchmarkConfig config;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help") {
            print_usage(argv[0]);
            exit(0);
        } else if (arg == "--min_sms" && i + 1 < argc) {
            config.min_sms = std::atoi(argv[++i]);
        } else if (arg == "--max_sms" && i + 1 < argc) {
            config.max_sms = std::atoi(argv[++i]);
        } else if (arg == "--gemv_M" && i + 1 < argc) {
            config.gemv_M = std::atoi(argv[++i]);
        } else if (arg == "--gemv_N" && i + 1 < argc) {
            config.gemv_N = std::atoi(argv[++i]);
        } else if (arg == "--gemm_size" && i + 1 < argc) {
            config.gemm_size = std::atoi(argv[++i]);
        } else if (arg == "--num_iters" && i + 1 < argc) {
            config.num_iters = std::atoi(argv[++i]);
        } else if (arg == "--tpb_gemv" && i + 1 < argc) {
            config.tpb_gemv = std::atoi(argv[++i]);
        } else if (arg == "--tpb_gemm" && i + 1 < argc) {
            config.tpb_gemm = std::atoi(argv[++i]);
        } else if (arg == "--repeats" && i + 1 < argc) {
            config.num_repeats = std::atoi(argv[++i]);
        } else if (arg == "--perfect_sync") {
            config.perfect_sync = true;
        } else if (arg == "--csv" && i + 1 < argc) {
            config.csv_path = argv[++i];
        } else {
            fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            exit(1);
        }
    }
    
    return config;
}

// =============================================================================
// Green Context Helper
// =============================================================================

struct GreenContextHandle {
    CUdevResource partition;
    CUdevResourceDesc desc;
    CUgreenCtx green_ctx;
    CUcontext ctx;
    CUstream stream_gemv;
    CUstream stream_gemm;
    
    GreenContextHandle() : desc(nullptr), green_ctx(nullptr), ctx(nullptr), 
                           stream_gemv(nullptr), stream_gemm(nullptr) {}
};

void create_green_context_for_sms(
    CUdevice device,
    const CUdevResource& device_resource,
    int target_sm_count,
    GreenContextHandle& handle
) {
    // Split resources to get partition with target_sm_count SMs
    unsigned int num_groups = 1;
    CUdevResource remaining;
    
    CHECK_CU(cuDevSmResourceSplitByCount(
        &handle.partition,
        &num_groups,
        &device_resource,
        &remaining,
        0,  // flags
        target_sm_count
    ));
    
    if (num_groups == 0) {
        fprintf(stderr, "Failed to create partition with %d SMs\n", target_sm_count);
        exit(1);
    }
    
    // Generate descriptor
    CHECK_CU(cuDevResourceGenerateDesc(&handle.desc, &handle.partition, 1));
    
    // Create green context
    CHECK_CU(cuGreenCtxCreate(&handle.green_ctx, handle.desc, device, CU_GREEN_CTX_DEFAULT_STREAM));
    
    // Convert to CUcontext
    CHECK_CU(cuCtxFromGreenCtx(&handle.ctx, handle.green_ctx));
    
    // Create TWO streams within the same green context (for concurrent execution)
    CHECK_CU(cuGreenCtxStreamCreate(&handle.stream_gemv, handle.green_ctx, CU_STREAM_NON_BLOCKING, 0));
    CHECK_CU(cuGreenCtxStreamCreate(&handle.stream_gemm, handle.green_ctx, CU_STREAM_NON_BLOCKING, 0));
}

void destroy_green_context(GreenContextHandle& handle) {
    if (handle.stream_gemv) {
        cuStreamSynchronize(handle.stream_gemv);
        cuStreamDestroy(handle.stream_gemv);
    }
    if (handle.stream_gemm) {
        cuStreamSynchronize(handle.stream_gemm);
        cuStreamDestroy(handle.stream_gemm);
    }
    if (handle.green_ctx) {
        cuGreenCtxDestroy(handle.green_ctx);
    }
    // desc and partition are automatically cleaned up
}

// =============================================================================
// Result Structures
// =============================================================================

struct GemvResult {
    double gflops;
    double time_ms;
};

struct GemmResult {
    double gflops;
    double time_ms;
};

struct ConcurrentResult {
    GemvResult gemv;
    GemmResult gemm;
    double wall_time_ms;
    double gemv_slowdown;
    double gemm_slowdown;
    double overlap_pct;
};

// =============================================================================
// Measurement Functions
// =============================================================================

GemvResult measure_gemv_only(
    CUstream stream,
    __half* d_A,
    __half* d_x,
    __half* d_y,
    int M,
    int N,
    int num_iters,
    int tpb_gemv
) {
    // Warm-up
    for (int i = 0; i < 3; ++i) {
        for (int iter = 0; iter < num_iters; ++iter) {
            launch_gemv_kernel(stream, d_A, d_x, d_y, M, N, tpb_gemv);
        }
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
    
    // Timed measurement - measure each iteration individually and average
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    float total_time_ms = 0.0f;
    for (int iter = 0; iter < num_iters; ++iter) {
        CHECK_CUDA(cudaEventRecord(start, stream));
        launch_gemv_kernel(stream, d_A, d_x, d_y, M, N, tpb_gemv);
        CHECK_CUDA(cudaEventRecord(stop, stream));
        CHECK_CUDA(cudaEventSynchronize(stop));
        
        float iter_time_ms = 0;
        CHECK_CUDA(cudaEventElapsedTime(&iter_time_ms, start, stop));
        total_time_ms += iter_time_ms;
    }
    
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    float avg_time_ms = total_time_ms / num_iters;
    
    // Calculate GFLOPS (2*M*N FLOPs per iteration)
    double flops_per_iter = 2.0 * M * N;
    double gflops = flops_per_iter / (avg_time_ms * 1e6);
    
    GemvResult result;
    result.gflops = gflops;
    result.time_ms = avg_time_ms;
    
    return result;
}

GemmResult measure_gemm_only(
    CUstream stream,
    __half* d_A,
    __half* d_B,
    __half* d_C,
    int gemm_size,
    int num_iters,
    int tpb_gemm
) {
    int M = gemm_size;
    int N = gemm_size;
    int K = gemm_size;
    
    // Warm-up
    for (int i = 0; i < 3; ++i) {
        for (int iter = 0; iter < num_iters; ++iter) {
            launch_gemm_kernel(stream, d_A, d_B, d_C, M, N, K, tpb_gemm);
        }
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
    
    // Timed measurement - measure each iteration individually and average
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    float total_time_ms = 0.0f;
    for (int iter = 0; iter < num_iters; ++iter) {
        CHECK_CUDA(cudaEventRecord(start, stream));
        launch_gemm_kernel(stream, d_A, d_B, d_C, M, N, K, tpb_gemm);
        CHECK_CUDA(cudaEventRecord(stop, stream));
        CHECK_CUDA(cudaEventSynchronize(stop));
        
        float iter_time_ms = 0;
        CHECK_CUDA(cudaEventElapsedTime(&iter_time_ms, start, stop));
        total_time_ms += iter_time_ms;
    }
    
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    float avg_time_ms = total_time_ms / num_iters;
    
    // Calculate GFLOPS (2*M*N*K FLOPs per iteration)
    double flops_per_iter = 2.0 * M * N * K;
    double gflops = flops_per_iter / (avg_time_ms * 1e6);
    
    GemmResult result;
    result.gflops = gflops;
    result.time_ms = avg_time_ms;
    
    return result;
}

ConcurrentResult measure_concurrent(
    CUstream stream_gemv,
    CUstream stream_gemm,
    __half* d_A_gemv,
    __half* d_x_gemv,
    __half* d_y_gemv,
    __half* d_A_gemm,
    __half* d_B_gemm,
    __half* d_C_gemm,
    int gemv_M,
    int gemv_N,
    int num_iters,
    int gemm_size,
    int tpb_gemv,
    int tpb_gemm,
    bool perfect_sync,
    const GemvResult& isolated_gemv,
    const GemmResult& isolated_gemm
) {
    int M = gemm_size;
    int N = gemm_size;
    int K = gemm_size;
    
    // // Warm-up: run both kernels concurrently a few times
    // for (int i = 0; i < 3; ++i) {
    //     for (int iter = 0; iter < num_iters; ++iter) {
    //         if (perfect_sync) {
    //             // Perfect alignment mode: use device synchronization (slower but guaranteed alignment)
    //             CHECK_CUDA(cudaDeviceSynchronize());
    //             launch_gemv_kernel(stream_gemv, d_A_gemv, d_x_gemv, d_y_gemv, gemv_M, gemv_N, tpb_gemv);
    //             launch_gemm_kernel(stream_gemm, d_A_gemm, d_B_gemm, d_C_gemm, M, N, K, tpb_gemm);
    //         } else {
    //             // Event-based mode: faster but may have slight misalignment
    //             cudaEvent_t sync_event;
    //             CHECK_CUDA(cudaEventCreate(&sync_event));
    //             CHECK_CUDA(cudaEventRecord(sync_event, 0));
                
    //             CHECK_CUDA(cudaStreamWaitEvent(stream_gemv, sync_event, 0));
    //             CHECK_CUDA(cudaStreamWaitEvent(stream_gemm, sync_event, 0));
                
    //             launch_gemv_kernel(stream_gemv, d_A_gemv, d_x_gemv, d_y_gemv, gemv_M, gemv_N, tpb_gemv);
    //             launch_gemm_kernel(stream_gemm, d_A_gemm, d_B_gemm, d_C_gemm, M, N, K, tpb_gemm);
                
    //             CHECK_CUDA(cudaEventDestroy(sync_event));
    //         }
    //     }
        
    //     CHECK_CUDA(cudaStreamSynchronize(stream_gemv));
    //     CHECK_CUDA(cudaStreamSynchronize(stream_gemm));
    // }
    
    // Timed concurrent measurement - measure each iteration individually
    cudaEvent_t gemv_start, gemv_end, gemm_start, gemm_end;
    CHECK_CUDA(cudaEventCreate(&gemv_start));
    CHECK_CUDA(cudaEventCreate(&gemv_end));
    CHECK_CUDA(cudaEventCreate(&gemm_start));
    CHECK_CUDA(cudaEventCreate(&gemm_end));
    
    // Pre-allocate synchronization events for all iterations (avoids in-loop overhead)
    std::vector<cudaEvent_t> sync_barriers(num_iters);
    std::vector<cudaEvent_t> gemv_iter_done(num_iters);
    std::vector<cudaEvent_t> gemm_iter_done(num_iters);
    for (int iter = 0; iter < num_iters; ++iter) {
        CHECK_CUDA(cudaEventCreate(&sync_barriers[iter]));
        CHECK_CUDA(cudaEventCreate(&gemv_iter_done[iter]));
        CHECK_CUDA(cudaEventCreate(&gemm_iter_done[iter]));
    }
    
    // Timing accumulation
    float total_gemv_time_ms = 0.0f;
    float total_gemm_time_ms = 0.0f;
    float max_wall_time_ms = 0.0f;
    
    // Launch both kernels iteratively with strict lockstep synchronization and per-iteration timing
    for (int iter = 0; iter < num_iters; ++iter) {
        if (perfect_sync) {
            // ===== PERFECT SYNC MODE =====
            // Use device synchronization for guaranteed perfect alignment
            // This ensures both kernels start at EXACTLY the same time in Nsight Systems
            // Trade-off: ~10-20% slower due to CPU-GPU sync overhead
            
            CHECK_CUDA(cudaDeviceSynchronize());  // Wait for all previous work to complete
            
            // Record start times
            CHECK_CUDA(cudaEventRecord(gemv_start, stream_gemv));
            CHECK_CUDA(cudaEventRecord(gemm_start, stream_gemm));
            
            // Launch both kernels immediately after sync (minimal timing skew)
            launch_gemv_kernel(stream_gemv, d_A_gemv, d_x_gemv, d_y_gemv, gemv_M, gemv_N, tpb_gemv);
            launch_gemm_kernel(stream_gemm, d_A_gemm, d_B_gemm, d_C_gemm, M, N, K, tpb_gemm);
            
            // Record end times
            CHECK_CUDA(cudaEventRecord(gemv_end, stream_gemv));
            CHECK_CUDA(cudaEventRecord(gemm_end, stream_gemm));
            
        } else {
            // ===== EVENT-BASED MODE (Default) =====
            // Faster but may show slight misalignment in Nsight Systems due to:
            // - Asynchronous event system
            // - GPU scheduler decisions
            // - Different kernel runtimes (GEMV ~220ms vs GEMM ~228ms)
            
            // Record sync barrier on default stream
            CHECK_CUDA(cudaEventRecord(sync_barriers[iter], 0));
            
            // Both streams wait for the barrier
            CHECK_CUDA(cudaStreamWaitEvent(stream_gemv, sync_barriers[iter], 0));
            CHECK_CUDA(cudaStreamWaitEvent(stream_gemm, sync_barriers[iter], 0));
            
            // Cross-stream synchronization - wait for OTHER stream's previous iteration
            if (iter > 0) {
                CHECK_CUDA(cudaStreamWaitEvent(stream_gemv, gemm_iter_done[iter-1], 0));
                CHECK_CUDA(cudaStreamWaitEvent(stream_gemm, gemv_iter_done[iter-1], 0));
            }
            
            // Record start times
            CHECK_CUDA(cudaEventRecord(gemv_start, stream_gemv));
            CHECK_CUDA(cudaEventRecord(gemm_start, stream_gemm));
            
            // Launch both kernels
            launch_gemv_kernel(stream_gemv, d_A_gemv, d_x_gemv, d_y_gemv, gemv_M, gemv_N, tpb_gemv);
            launch_gemm_kernel(stream_gemm, d_A_gemm, d_B_gemm, d_C_gemm, M, N, K, tpb_gemm);
            
            // Record end times and completion for next iteration's cross-sync
            CHECK_CUDA(cudaEventRecord(gemv_end, stream_gemv));
            CHECK_CUDA(cudaEventRecord(gemm_end, stream_gemm));
            CHECK_CUDA(cudaEventRecord(gemv_iter_done[iter], stream_gemv));
            CHECK_CUDA(cudaEventRecord(gemm_iter_done[iter], stream_gemm));
        }
        
        // Synchronize and calculate times for this iteration
        CHECK_CUDA(cudaEventSynchronize(gemv_end));
        CHECK_CUDA(cudaEventSynchronize(gemm_end));
        
        float gemv_iter_time, gemm_iter_time;
        CHECK_CUDA(cudaEventElapsedTime(&gemv_iter_time, gemv_start, gemv_end));
        CHECK_CUDA(cudaEventElapsedTime(&gemm_iter_time, gemm_start, gemm_end));
        
        total_gemv_time_ms += gemv_iter_time;
        total_gemm_time_ms += gemm_iter_time;
        max_wall_time_ms = std::max(max_wall_time_ms, std::max(gemv_iter_time, gemm_iter_time));
    }
    
    // Cleanup synchronization events
    for (int iter = 0; iter < num_iters; ++iter) {
        CHECK_CUDA(cudaEventDestroy(sync_barriers[iter]));
        CHECK_CUDA(cudaEventDestroy(gemv_iter_done[iter]));
        CHECK_CUDA(cudaEventDestroy(gemm_iter_done[iter]));
    }
    
    // Calculate average times per iteration
    float avg_gemv_time_ms = total_gemv_time_ms / num_iters;
    float avg_gemm_time_ms = total_gemm_time_ms / num_iters;
    float avg_wall_time_ms = max_wall_time_ms;  // Use max from any iteration
    
    // Calculate metrics per iteration
    double gemv_flops_per_iter = 2.0 * gemv_M * gemv_N;
    double gemv_gflops = gemv_flops_per_iter / (avg_gemv_time_ms * 1e6);
    
    double gemm_flops_per_iter = 2.0 * M * N * K;
    double gemm_gflops = gemm_flops_per_iter / (avg_gemm_time_ms * 1e6);
    
    // Calculate slowdowns and overlap
    double gemv_slowdown = avg_gemv_time_ms / isolated_gemv.time_ms;
    double gemm_slowdown = avg_gemm_time_ms / isolated_gemm.time_ms;
    double overlap_pct = 100.0 * (1.0 - avg_wall_time_ms / (isolated_gemv.time_ms + isolated_gemm.time_ms));
    
    // Cleanup events
    CHECK_CUDA(cudaEventDestroy(gemv_start));
    CHECK_CUDA(cudaEventDestroy(gemv_end));
    CHECK_CUDA(cudaEventDestroy(gemm_start));
    CHECK_CUDA(cudaEventDestroy(gemm_end));
    
    ConcurrentResult result;
    result.gemv.gflops = gemv_gflops;
    result.gemv.time_ms = avg_gemv_time_ms;
    result.gemm.gflops = gemm_gflops;
    result.gemm.time_ms = avg_gemm_time_ms;
    result.wall_time_ms = avg_wall_time_ms;
    result.gemv_slowdown = gemv_slowdown;
    result.gemm_slowdown = gemm_slowdown;
    result.overlap_pct = overlap_pct;
    
    return result;
}

// =============================================================================
// Full Benchmark for One SM Count
// =============================================================================

struct FullBenchmarkResult {
    int num_sms;
    int gemv_M;
    int gemv_N;
    int gemm_size;
    GemvResult isolated_gemv;
    GemmResult isolated_gemm;
    ConcurrentResult concurrent;
};

FullBenchmarkResult benchmark_sm_count(
    CUdevice device,
    const CUdevResource& device_resource,
    int target_sm_count,
    const BenchmarkConfig& config
) {
    // Create green context for this SM count
    GreenContextHandle handle;
    create_green_context_for_sms(device, device_resource, target_sm_count, handle);
    
    // Set context current
    CHECK_CU(cuCtxSetCurrent(handle.ctx));
    
    // Allocate buffers (FP16)
    __half* d_A_gemv;
    __half* d_x_gemv;
    __half* d_y_gemv;
    __half* d_A_gemm;
    __half* d_B_gemm;
    __half* d_C_gemm;
    
    allocate_gemv_buffers(&d_A_gemv, &d_x_gemv, &d_y_gemv, config.gemv_M, config.gemv_N);
    allocate_gemm_buffers(&d_A_gemm, &d_B_gemm, &d_C_gemm, config.gemm_size, config.gemm_size, config.gemm_size);
    
    // Run measurements multiple times and average
    std::vector<GemvResult> gemv_results;
    std::vector<GemmResult> gemm_results;
    std::vector<ConcurrentResult> concurrent_results;
    
    for (int rep = 0; rep < config.num_repeats; ++rep) {
        // Measure isolated GEMV
        GemvResult gemv = measure_gemv_only(
            handle.stream_gemv,
            d_A_gemv,
            d_x_gemv,
            d_y_gemv,
            config.gemv_M,
            config.gemv_N,
            config.num_iters,
            config.tpb_gemv
        );
        gemv_results.push_back(gemv);
        
        // Measure isolated GEMM
        GemmResult gemm = measure_gemm_only(
            handle.stream_gemm,
            d_A_gemm,
            d_B_gemm,
            d_C_gemm,
            config.gemm_size,
            config.num_iters,
            config.tpb_gemm
        );
        gemm_results.push_back(gemm);
        
        // Measure concurrent execution
        ConcurrentResult concurrent = measure_concurrent(
            handle.stream_gemv,
            handle.stream_gemm,
            d_A_gemv,
            d_x_gemv,
            d_y_gemv,
            d_A_gemm,
            d_B_gemm,
            d_C_gemm,
            config.gemv_M,
            config.gemv_N,
            config.num_iters,
            config.gemm_size,
            config.tpb_gemv,
            config.tpb_gemm,
            config.perfect_sync,
            gemv,
            gemm
        );
        concurrent_results.push_back(concurrent);
    }
    
    // Average results
    GemvResult avg_gemv = {0, 0};
    GemmResult avg_gemm = {0, 0};
    ConcurrentResult avg_concurrent = {0};
    
    for (const auto& r : gemv_results) {
        avg_gemv.gflops += r.gflops;
        avg_gemv.time_ms += r.time_ms;
    }
    avg_gemv.gflops /= gemv_results.size();
    avg_gemv.time_ms /= gemv_results.size();
    
    for (const auto& r : gemm_results) {
        avg_gemm.gflops += r.gflops;
        avg_gemm.time_ms += r.time_ms;
    }
    avg_gemm.gflops /= gemm_results.size();
    avg_gemm.time_ms /= gemm_results.size();
    
    for (const auto& r : concurrent_results) {
        avg_concurrent.gemv.gflops += r.gemv.gflops;
        avg_concurrent.gemv.time_ms += r.gemv.time_ms;
        avg_concurrent.gemm.gflops += r.gemm.gflops;
        avg_concurrent.gemm.time_ms += r.gemm.time_ms;
        avg_concurrent.wall_time_ms += r.wall_time_ms;
        avg_concurrent.gemv_slowdown += r.gemv_slowdown;
        avg_concurrent.gemm_slowdown += r.gemm_slowdown;
        avg_concurrent.overlap_pct += r.overlap_pct;
    }
    avg_concurrent.gemv.gflops /= concurrent_results.size();
    avg_concurrent.gemv.time_ms /= concurrent_results.size();
    avg_concurrent.gemm.gflops /= concurrent_results.size();
    avg_concurrent.gemm.time_ms /= concurrent_results.size();
    avg_concurrent.wall_time_ms /= concurrent_results.size();
    avg_concurrent.gemv_slowdown /= concurrent_results.size();
    avg_concurrent.gemm_slowdown /= concurrent_results.size();
    avg_concurrent.overlap_pct /= concurrent_results.size();
    
    // Cleanup
    free_gemv_buffers(d_A_gemv, d_x_gemv, d_y_gemv);
    free_gemm_buffers(d_A_gemm, d_B_gemm, d_C_gemm);
    destroy_green_context(handle);
    
    FullBenchmarkResult result;
    result.num_sms = target_sm_count;
    result.gemv_M = config.gemv_M;
    result.gemv_N = config.gemv_N;
    result.gemm_size = config.gemm_size;
    result.isolated_gemv = avg_gemv;
    result.isolated_gemm = avg_gemm;
    result.concurrent = avg_concurrent;
    
    return result;
}

// =============================================================================
// CSV Output
// =============================================================================

void write_csv(const std::string& path, const std::vector<FullBenchmarkResult>& results) {
    FILE* f = fopen(path.c_str(), "w");
    if (!f) {
        fprintf(stderr, "Failed to open CSV file: %s\n", path.c_str());
        return;
    }
    
    fprintf(f, "num_sms,gemv_M,gemv_N,gemm_size,mode,gemv_gflops,gemm_gflops,gemv_time_ms,gemm_time_ms,gemv_slowdown,gemm_slowdown,overlap_pct\n");
    
    for (const auto& r : results) {
        // Isolated GEMV
        fprintf(f, "%d,%d,%d,%d,isolated_gemv,%.3f,0.0,%.3f,0.0,1.0,0.0,0.0\n",
                r.num_sms, r.gemv_M, r.gemv_N, r.gemm_size,
                r.isolated_gemv.gflops, r.isolated_gemv.time_ms);
        
        // Isolated GEMM
        fprintf(f, "%d,%d,%d,%d,isolated_gemm,0.0,%.3f,0.0,%.3f,0.0,1.0,0.0\n",
                r.num_sms, r.gemv_M, r.gemv_N, r.gemm_size,
                r.isolated_gemm.gflops, r.isolated_gemm.time_ms);
        
        // Concurrent
        fprintf(f, "%d,%d,%d,%d,concurrent,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
                r.num_sms, r.gemv_M, r.gemv_N, r.gemm_size,
                r.concurrent.gemv.gflops,
                r.concurrent.gemm.gflops,
                r.concurrent.gemv.time_ms,
                r.concurrent.gemm.time_ms,
                r.concurrent.gemv_slowdown,
                r.concurrent.gemm_slowdown,
                r.concurrent.overlap_pct);
    }
    
    fclose(f);
    printf("Results written to: %s\n", path.c_str());
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    // Parse arguments
    BenchmarkConfig config = parse_args(argc, argv);
    
    // Initialize CUDA Driver API
    CHECK_CU(cuInit(0));
    
    // Get device
    CUdevice device;
    CHECK_CU(cuDeviceGet(&device, 0));
    
    // Query device properties
    char device_name[256];
    CHECK_CU(cuDeviceGetName(device_name, sizeof(device_name), device));
    
    int total_sms;
    CHECK_CU(cuDeviceGetAttribute(&total_sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));
    
    int compute_major, compute_minor;
    CHECK_CU(cuDeviceGetAttribute(&compute_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
    CHECK_CU(cuDeviceGetAttribute(&compute_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
    
    printf("=============================================================\n");
    printf("GEMV-GEMM Contention Test (FP16 with Tensor Cores)\n");
    printf("=============================================================\n");
    printf("Device: %s\n", device_name);
    printf("Compute Capability: %d.%d\n", compute_major, compute_minor);
    printf("Total SMs: %d\n", total_sms);
    printf("Configuration:\n");
    printf("  SM Range: %d - %d\n", config.min_sms, 
           config.max_sms > 0 ? config.max_sms : total_sms);
    printf("  Unified Iterations: %d\n", config.num_iters);
    printf("  Data Type: FP16 (__half)\n");
    printf("  GEMV: [%d x %d] matrix × vector (FP16)\n", 
           config.gemv_M, config.gemv_N);
    printf("  GEMV Expected FLOPs (total): %.2f GFLOPs\n",
           2.0 * config.gemv_M * config.gemv_N * config.num_iters / 1e9);
    printf("  GEMM Size: %d x %d x %d (FP16 with WMMA Tensor Cores)\n", 
           config.gemm_size, config.gemm_size, config.gemm_size);
    printf("  GEMM Expected FLOPs (total): %.2f GFLOPs\n",
           2.0 * config.gemm_size * config.gemm_size * config.gemm_size * config.num_iters / 1e9);
    printf("  Threads per block (GEMV): %d\n", config.tpb_gemv);
    printf("  Repeats: %d\n", config.num_repeats);
    printf("  CSV output: %s\n", config.csv_path.c_str());
    printf("=============================================================\n\n");
    
    // Set max_sms if not specified
    if (config.max_sms <= 0) {
        config.max_sms = total_sms;
    }
    
    // Get device SM resources
    CUdevResource device_resource;
    CHECK_CU(cuDeviceGetDevResource(device, &device_resource, CU_DEV_RESOURCE_TYPE_SM));
    
    // Query partition constraints
    printf("Device SM Resource Info:\n");
    printf("  Total SMs: %u\n", device_resource.sm.smCount);
    printf("  Min SM Partition Size: %u\n", device_resource.sm.minSmPartitionSize);
    printf("  SM Co-scheduled Alignment: %u\n", device_resource.sm.smCoscheduledAlignment);
    printf("\n");
    
    // Determine SM counts to test (respecting alignment)
    int alignment = device_resource.sm.smCoscheduledAlignment;
    int min_partition = device_resource.sm.minSmPartitionSize;
    
    std::vector<int> sm_counts;
    for (int sm = std::max(config.min_sms, (int)min_partition); 
         sm <= config.max_sms; 
         sm += alignment) {
        sm_counts.push_back(sm);
    }
    
    if (sm_counts.empty()) {
        fprintf(stderr, "Error: No valid SM counts in range [%d, %d] with alignment %d\n",
                config.min_sms, config.max_sms, alignment);
        return 1;
    }
    
    printf("Testing %zu SM configurations...\n\n", sm_counts.size());
    
    // Run benchmark for each SM count
    std::vector<FullBenchmarkResult> results;
    for (int sm_count : sm_counts) {
        printf("Testing with %d SMs:\n", sm_count);
        
        FullBenchmarkResult result = benchmark_sm_count(
            device,
            device_resource,
            sm_count,
            config
        );
        
        results.push_back(result);
        
        // Print detailed results
        printf("  [GEMV Only]  Throughput: %.2f GFLOPS, Time: %.2f ms\n",
               result.isolated_gemv.gflops, result.isolated_gemv.time_ms);
        printf("  [GEMM Only] Throughput: %.2f GFLOPS, Time: %.2f ms\n",
               result.isolated_gemm.gflops, result.isolated_gemm.time_ms);
        printf("  [Concurrent] Wall Time: %.2f ms (overlap: %.1f%%)\n",
               result.concurrent.wall_time_ms, result.concurrent.overlap_pct);
        printf("               GEMV: %.2f GFLOPS (%.1f%% retained), Time: %.2f ms, Slowdown: %.2fx\n",
               result.concurrent.gemv.gflops,
               100.0 * result.concurrent.gemv.gflops / result.isolated_gemv.gflops,
               result.concurrent.gemv.time_ms,
               result.concurrent.gemv_slowdown);
        printf("               GEMM: %.2f GFLOPS (%.1f%% retained), Time: %.2f ms, Slowdown: %.2fx\n",
               result.concurrent.gemm.gflops,
               100.0 * result.concurrent.gemm.gflops / result.isolated_gemm.gflops,
               result.concurrent.gemm.time_ms,
               result.concurrent.gemm_slowdown);
        printf("\n");
    }
    
    // Write CSV
    write_csv(config.csv_path, results);
    
    printf("\n=============================================================\n");
    printf("Benchmark Complete!\n");
    printf("=============================================================\n");
    
    return 0;
}

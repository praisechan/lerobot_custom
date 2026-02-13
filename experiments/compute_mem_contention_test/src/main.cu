#include <cuda.h>
#include <cuda_runtime.h>
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
    size_t mem_bytes;
    int gemm_size;
    int tpb_mem;
    int tpb_gemm;
    int num_repeats;
    std::string csv_path;
    
    // Defaults
    BenchmarkConfig() :
        min_sms(8),
        max_sms(0),  // Will be set to device max
        mem_bytes(2ULL * 1024 * 1024 * 1024),  // 2 GiB
        gemm_size(2048),
        tpb_mem(256),
        tpb_gemm(256),  // Not used, GEMM uses fixed TILE_SIZE
        num_repeats(5),
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
    printf("  --mem_bytes <N>     Bytes for mem copy kernel (default: 2147483648)\n");
    printf("  --gemm_size <N>     Matrix dimension for GEMM (default: 2048)\n");
    printf("  --tpb_mem <N>       Threads per block for mem copy (default: 256)\n");
    printf("  --tpb_gemm <N>      Threads per block for GEMM (default: 256)\n");
    printf("  --repeats <N>       Measurement repeats (default: 5)\n");
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
        } else if (arg == "--mem_bytes" && i + 1 < argc) {
            config.mem_bytes = std::atoll(argv[++i]);
        } else if (arg == "--gemm_size" && i + 1 < argc) {
            config.gemm_size = std::atoi(argv[++i]);
        } else if (arg == "--tpb_mem" && i + 1 < argc) {
            config.tpb_mem = std::atoi(argv[++i]);
        } else if (arg == "--tpb_gemm" && i + 1 < argc) {
            config.tpb_gemm = std::atoi(argv[++i]);
        } else if (arg == "--repeats" && i + 1 < argc) {
            config.num_repeats = std::atoi(argv[++i]);
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
    CUstream stream_mem;
    CUstream stream_gemm;
    
    GreenContextHandle() : desc(nullptr), green_ctx(nullptr), ctx(nullptr), 
                           stream_mem(nullptr), stream_gemm(nullptr) {}
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
    CHECK_CU(cuGreenCtxStreamCreate(&handle.stream_mem, handle.green_ctx, CU_STREAM_NON_BLOCKING, 0));
    CHECK_CU(cuGreenCtxStreamCreate(&handle.stream_gemm, handle.green_ctx, CU_STREAM_NON_BLOCKING, 0));
}

void destroy_green_context(GreenContextHandle& handle) {
    if (handle.stream_mem) {
        cuStreamSynchronize(handle.stream_mem);
        cuStreamDestroy(handle.stream_mem);
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

struct MemResult {
    double bw_gb_s;
    double time_ms;
};

struct GemmResult {
    double gflops;
    double time_ms;
};

struct ConcurrentResult {
    MemResult mem;
    GemmResult gemm;
    double wall_time_ms;
    double mem_slowdown;
    double gemm_slowdown;
    double overlap_pct;
};

// =============================================================================
// Measurement Functions
// =============================================================================

MemResult measure_mem_only(
    CUstream stream,
    Vec4U32* d_input,
    uint64_t* d_sink,
    size_t mem_bytes,
    int tpb_mem,
    int num_sms
) {
    // Warm-up
    for (int i = 0; i < 3; ++i) {
        launch_bandwidth_kernel(stream, d_input, d_sink, mem_bytes, tpb_mem, num_sms);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
    
    // Timed measurement
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    CHECK_CUDA(cudaEventRecord(start, stream));
    launch_bandwidth_kernel(stream, d_input, d_sink, mem_bytes, tpb_mem, num_sms);
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float elapsed_ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
    
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    // Calculate bandwidth
    double elapsed_s = elapsed_ms / 1000.0;
    double bw_gb_s = (mem_bytes / (1024.0 * 1024.0 * 1024.0)) / elapsed_s;
    
    MemResult result;
    result.bw_gb_s = bw_gb_s;
    result.time_ms = elapsed_ms;
    
    return result;
}

GemmResult measure_gemm_only(
    CUstream stream,
    float* d_A,
    float* d_B,
    float* d_C,
    int gemm_size,
    int tpb_gemm
) {
    int M = gemm_size;
    int N = gemm_size;
    int K = gemm_size;
    
    // Warm-up
    for (int i = 0; i < 3; ++i) {
        launch_gemm_kernel(stream, d_A, d_B, d_C, M, N, K, tpb_gemm);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
    
    // Timed measurement
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    CHECK_CUDA(cudaEventRecord(start, stream));
    launch_gemm_kernel(stream, d_A, d_B, d_C, M, N, K, tpb_gemm);
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float elapsed_ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
    
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    // Calculate GFLOPS (2*M*N*K FLOPs for GEMM)
    double flops = 2.0 * M * N * K;
    double gflops = flops / (elapsed_ms * 1e6);
    
    GemmResult result;
    result.gflops = gflops;
    result.time_ms = elapsed_ms;
    
    return result;
}

ConcurrentResult measure_concurrent(
    CUstream stream_mem,
    CUstream stream_gemm,
    Vec4U32* d_input,
    uint64_t* d_sink,
    float* d_A,
    float* d_B,
    float* d_C,
    size_t mem_bytes,
    int gemm_size,
    int tpb_mem,
    int tpb_gemm,
    int num_sms,
    const MemResult& isolated_mem,
    const GemmResult& isolated_gemm
) {
    int M = gemm_size;
    int N = gemm_size;
    int K = gemm_size;
    
    // Warm-up: run both kernels concurrently a few times
    for (int i = 0; i < 3; ++i) {
        // Create sync event
        cudaEvent_t sync_event;
        CHECK_CUDA(cudaEventCreate(&sync_event));
        CHECK_CUDA(cudaEventRecord(sync_event, 0));
        
        // Make both streams wait for sync event
        CHECK_CUDA(cudaStreamWaitEvent(stream_mem, sync_event, 0));
        CHECK_CUDA(cudaStreamWaitEvent(stream_gemm, sync_event, 0));
        
        // Launch both kernels
        launch_bandwidth_kernel(stream_mem, d_input, d_sink, mem_bytes, tpb_mem, num_sms);
        launch_gemm_kernel(stream_gemm, d_A, d_B, d_C, M, N, K, tpb_gemm);
        
        // Synchronize both streams
        CHECK_CUDA(cudaStreamSynchronize(stream_mem));
        CHECK_CUDA(cudaStreamSynchronize(stream_gemm));
        
        CHECK_CUDA(cudaEventDestroy(sync_event));
    }
    
    // Timed concurrent measurement
    cudaEvent_t start_sync, mem_start, mem_end, gemm_start, gemm_end;
    CHECK_CUDA(cudaEventCreate(&start_sync));
    CHECK_CUDA(cudaEventCreate(&mem_start));
    CHECK_CUDA(cudaEventCreate(&mem_end));
    CHECK_CUDA(cudaEventCreate(&gemm_start));
    CHECK_CUDA(cudaEventCreate(&gemm_end));
    
    // Record sync event on default stream
    CHECK_CUDA(cudaEventRecord(start_sync, 0));
    
    // Both streams wait for sync event (ensures simultaneous start)
    CHECK_CUDA(cudaStreamWaitEvent(stream_mem, start_sync, 0));
    CHECK_CUDA(cudaStreamWaitEvent(stream_gemm, start_sync, 0));
    
    // Launch with timing on each stream
    CHECK_CUDA(cudaEventRecord(mem_start, stream_mem));
    launch_bandwidth_kernel(stream_mem, d_input, d_sink, mem_bytes, tpb_mem, num_sms);
    CHECK_CUDA(cudaEventRecord(mem_end, stream_mem));
    
    CHECK_CUDA(cudaEventRecord(gemm_start, stream_gemm));
    launch_gemm_kernel(stream_gemm, d_A, d_B, d_C, M, N, K, tpb_gemm);
    CHECK_CUDA(cudaEventRecord(gemm_end, stream_gemm));
    
    // Synchronize both streams
    CHECK_CUDA(cudaStreamSynchronize(stream_mem));
    CHECK_CUDA(cudaStreamSynchronize(stream_gemm));
    
    // Calculate individual kernel times
    float mem_time_ms, gemm_time_ms;
    CHECK_CUDA(cudaEventElapsedTime(&mem_time_ms, mem_start, mem_end));
    CHECK_CUDA(cudaEventElapsedTime(&gemm_time_ms, gemm_start, gemm_end));
    
    // Calculate wall-clock time (max of both kernels for concurrent execution)
    float wall_time_ms = std::max(mem_time_ms, gemm_time_ms);
    
    // Calculate metrics
    double mem_bw_gb_s = (mem_bytes / (1024.0 * 1024.0 * 1024.0)) / (mem_time_ms / 1000.0);
    double gemm_flops = 2.0 * M * N * K;
    double gemm_gflops = gemm_flops / (gemm_time_ms * 1e6);
    
    // Calculate slowdowns and overlap
    double mem_slowdown = mem_time_ms / isolated_mem.time_ms;
    double gemm_slowdown = gemm_time_ms / isolated_gemm.time_ms;
    double overlap_pct = 100.0 * (1.0 - wall_time_ms / (isolated_mem.time_ms + isolated_gemm.time_ms));
    
    // Cleanup events
    CHECK_CUDA(cudaEventDestroy(start_sync));
    CHECK_CUDA(cudaEventDestroy(mem_start));
    CHECK_CUDA(cudaEventDestroy(mem_end));
    CHECK_CUDA(cudaEventDestroy(gemm_start));
    CHECK_CUDA(cudaEventDestroy(gemm_end));
    
    ConcurrentResult result;
    result.mem.bw_gb_s = mem_bw_gb_s;
    result.mem.time_ms = mem_time_ms;
    result.gemm.gflops = gemm_gflops;
    result.gemm.time_ms = gemm_time_ms;
    result.wall_time_ms = wall_time_ms;
    result.mem_slowdown = mem_slowdown;
    result.gemm_slowdown = gemm_slowdown;
    result.overlap_pct = overlap_pct;
    
    return result;
}

// =============================================================================
// Full Benchmark for One SM Count
// =============================================================================

struct FullBenchmarkResult {
    int num_sms;
    MemResult isolated_mem;
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
    
    // Allocate buffers
    Vec4U32* d_input;
    uint64_t* d_sink;
    float* d_A;
    float* d_B;
    float* d_C;
    
    size_t threads_per_sm = config.tpb_mem * 2;
    size_t max_blocks = (target_sm_count * threads_per_sm) / config.tpb_mem;
    
    allocate_mem_buffers(&d_input, &d_sink, config.mem_bytes, max_blocks);
    allocate_gemm_buffers(&d_A, &d_B, &d_C, config.gemm_size, config.gemm_size, config.gemm_size);
    
    // Run measurements multiple times and average
    std::vector<MemResult> mem_results;
    std::vector<GemmResult> gemm_results;
    std::vector<ConcurrentResult> concurrent_results;
    
    for (int rep = 0; rep < config.num_repeats; ++rep) {
        // Measure isolated mem copy
        MemResult mem = measure_mem_only(
            handle.stream_mem,
            d_input,
            d_sink,
            config.mem_bytes,
            config.tpb_mem,
            target_sm_count
        );
        mem_results.push_back(mem);
        
        // Measure isolated GEMM
        GemmResult gemm = measure_gemm_only(
            handle.stream_gemm,
            d_A,
            d_B,
            d_C,
            config.gemm_size,
            config.tpb_gemm
        );
        gemm_results.push_back(gemm);
        
        // Measure concurrent execution
        ConcurrentResult concurrent = measure_concurrent(
            handle.stream_mem,
            handle.stream_gemm,
            d_input,
            d_sink,
            d_A,
            d_B,
            d_C,
            config.mem_bytes,
            config.gemm_size,
            config.tpb_mem,
            config.tpb_gemm,
            target_sm_count,
            mem,
            gemm
        );
        concurrent_results.push_back(concurrent);
    }
    
    // Average results
    MemResult avg_mem = {0, 0};
    GemmResult avg_gemm = {0, 0};
    ConcurrentResult avg_concurrent = {0};
    
    for (const auto& r : mem_results) {
        avg_mem.bw_gb_s += r.bw_gb_s;
        avg_mem.time_ms += r.time_ms;
    }
    avg_mem.bw_gb_s /= mem_results.size();
    avg_mem.time_ms /= mem_results.size();
    
    for (const auto& r : gemm_results) {
        avg_gemm.gflops += r.gflops;
        avg_gemm.time_ms += r.time_ms;
    }
    avg_gemm.gflops /= gemm_results.size();
    avg_gemm.time_ms /= gemm_results.size();
    
    for (const auto& r : concurrent_results) {
        avg_concurrent.mem.bw_gb_s += r.mem.bw_gb_s;
        avg_concurrent.mem.time_ms += r.mem.time_ms;
        avg_concurrent.gemm.gflops += r.gemm.gflops;
        avg_concurrent.gemm.time_ms += r.gemm.time_ms;
        avg_concurrent.wall_time_ms += r.wall_time_ms;
        avg_concurrent.mem_slowdown += r.mem_slowdown;
        avg_concurrent.gemm_slowdown += r.gemm_slowdown;
        avg_concurrent.overlap_pct += r.overlap_pct;
    }
    avg_concurrent.mem.bw_gb_s /= concurrent_results.size();
    avg_concurrent.mem.time_ms /= concurrent_results.size();
    avg_concurrent.gemm.gflops /= concurrent_results.size();
    avg_concurrent.gemm.time_ms /= concurrent_results.size();
    avg_concurrent.wall_time_ms /= concurrent_results.size();
    avg_concurrent.mem_slowdown /= concurrent_results.size();
    avg_concurrent.gemm_slowdown /= concurrent_results.size();
    avg_concurrent.overlap_pct /= concurrent_results.size();
    
    // Cleanup
    free_mem_buffers(d_input, d_sink);
    free_gemm_buffers(d_A, d_B, d_C);
    destroy_green_context(handle);
    
    FullBenchmarkResult result;
    result.num_sms = target_sm_count;
    result.isolated_mem = avg_mem;
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
    
    fprintf(f, "num_sms,mode,mem_bw_gb_s,gemm_gflops,mem_time_ms,gemm_time_ms,mem_slowdown,gemm_slowdown,overlap_pct\n");
    
    for (const auto& r : results) {
        // Isolated mem
        fprintf(f, "%d,isolated_mem,%.3f,0.0,%.3f,0.0,1.0,0.0,0.0\n",
                r.num_sms, r.isolated_mem.bw_gb_s, r.isolated_mem.time_ms);
        
        // Isolated GEMM
        fprintf(f, "%d,isolated_gemm,0.0,%.3f,0.0,%.3f,0.0,1.0,0.0\n",
                r.num_sms, r.isolated_gemm.gflops, r.isolated_gemm.time_ms);
        
        // Concurrent
        fprintf(f, "%d,concurrent,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
                r.num_sms,
                r.concurrent.mem.bw_gb_s,
                r.concurrent.gemm.gflops,
                r.concurrent.mem.time_ms,
                r.concurrent.gemm.time_ms,
                r.concurrent.mem_slowdown,
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
    printf("Compute-Memory Contention Test using CUDA Green Contexts\n");
    printf("=============================================================\n");
    printf("Device: %s\n", device_name);
    printf("Compute Capability: %d.%d\n", compute_major, compute_minor);
    printf("Total SMs: %d\n", total_sms);
    printf("Configuration:\n");
    printf("  SM Range: %d - %d\n", config.min_sms, 
           config.max_sms > 0 ? config.max_sms : total_sms);
    printf("  Mem Copy Bytes: %lu (%.2f GiB)\n", 
           config.mem_bytes,
           config.mem_bytes / (1024.0 * 1024.0 * 1024.0));
    printf("  GEMM Size: %d x %d x %d\n", 
           config.gemm_size, config.gemm_size, config.gemm_size);
    printf("  Expected GEMM FLOPs: %.2f GFLOPs\n",
           2.0 * config.gemm_size * config.gemm_size * config.gemm_size / 1e9);
    printf("  Threads per block (mem): %d\n", config.tpb_mem);
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
        printf("  [Mem Only]  BW: %.2f GB/s, Time: %.2f ms\n",
               result.isolated_mem.bw_gb_s, result.isolated_mem.time_ms);
        printf("  [GEMM Only] Throughput: %.2f GFLOPS, Time: %.2f ms\n",
               result.isolated_gemm.gflops, result.isolated_gemm.time_ms);
        printf("  [Concurrent] Wall Time: %.2f ms (overlap: %.1f%%)\n",
               result.concurrent.wall_time_ms, result.concurrent.overlap_pct);
        printf("               Mem: %.2f GB/s (%.1f%% retained), Time: %.2f ms, Slowdown: %.2fx\n",
               result.concurrent.mem.bw_gb_s,
               100.0 * result.concurrent.mem.bw_gb_s / result.isolated_mem.bw_gb_s,
               result.concurrent.mem.time_ms,
               result.concurrent.mem_slowdown);
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

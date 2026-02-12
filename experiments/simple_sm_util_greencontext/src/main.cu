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
    size_t total_bytes_to_read;
    int threads_per_block;
    int iters_override;
    int num_repeats;
    std::string csv_path;
    
    // Defaults
    BenchmarkConfig() :
        min_sms(8),
        max_sms(0),  // Will be set to device max
        total_bytes_to_read(1073741824),  // 1 GiB
        threads_per_block(256),
        iters_override(-1),  // No override
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
    printf("  --bytes <N>         Total bytes to read per measurement (default: 1073741824)\n");
    printf("  --tpb <N>           Threads per block (default: 256)\n");
    printf("  --iters <N>         Override iterations (default: auto-calculated)\n");
    printf("  --repeats <N>       Number of measurement repeats (default: 5)\n");
    printf("  --csv <path>        Output CSV path (default: ./results.csv)\n");
    printf("  --help              Print this help\n");
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
        } else if (arg == "--bytes" && i + 1 < argc) {
            config.total_bytes_to_read = std::atoll(argv[++i]);
        } else if (arg == "--tpb" && i + 1 < argc) {
            config.threads_per_block = std::atoi(argv[++i]);
        } else if (arg == "--iters" && i + 1 < argc) {
            config.iters_override = std::atoi(argv[++i]);
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
    CUstream stream;
    
    GreenContextHandle() : desc(nullptr), green_ctx(nullptr), ctx(nullptr), stream(nullptr) {}
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
    
    // Create stream
    CHECK_CU(cuGreenCtxStreamCreate(&handle.stream, handle.green_ctx, CU_STREAM_NON_BLOCKING, 0));
}

void destroy_green_context(GreenContextHandle& handle) {
    if (handle.stream) {
        cuStreamSynchronize(handle.stream);
        cuStreamDestroy(handle.stream);
    }
    if (handle.green_ctx) {
        cuGreenCtxDestroy(handle.green_ctx);
    }
    // desc and partition are automatically cleaned up
}

// =============================================================================
// Bandwidth Measurement
// =============================================================================

struct BandwidthResult {
    int num_sms;
    double mean_bw_gb_s;
    double stdev_bw_gb_s;
};

double measure_bandwidth_once(
    CUstream stream,
    const Vec4U32* d_input,
    uint64_t* d_sink,
    size_t total_bytes,
    int threads_per_block,
    int num_sms,
    int warmup_iters
) {
    // Calculate kernel launch parameters
    size_t num_vec4_elements = total_bytes / sizeof(Vec4U32);
    size_t threads_per_sm = threads_per_block * 2;
    size_t total_threads = num_sms * threads_per_sm;
    size_t num_blocks = total_threads / threads_per_block;
    int iters_per_thread = (num_vec4_elements + total_threads - 1) / total_threads;
    
    // Warm-up
    for (int i = 0; i < warmup_iters; ++i) {
        bandwidth_read_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
            d_input, d_sink, num_vec4_elements, iters_per_thread
        );
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
    
    // Timed measurement
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    CHECK_CUDA(cudaEventRecord(start, stream));
    bandwidth_read_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        d_input, d_sink, num_vec4_elements, iters_per_thread
    );
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float elapsed_ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
    
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    // Calculate bandwidth in GB/s
    double elapsed_s = elapsed_ms / 1000.0;
    double bw_gb_s = (total_bytes / (1024.0 * 1024.0 * 1024.0)) / elapsed_s;
    
    return bw_gb_s;
}

BandwidthResult measure_bandwidth_for_sms(
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
    size_t threads_per_sm = config.threads_per_block * 2;
    size_t max_blocks = (target_sm_count * threads_per_sm) / config.threads_per_block;
    allocate_buffers(&d_input, &d_sink, config.total_bytes_to_read, max_blocks);
    
    // Measure multiple times
    std::vector<double> measurements;
    for (int rep = 0; rep < config.num_repeats; ++rep) {
        double bw = measure_bandwidth_once(
            handle.stream,
            d_input,
            d_sink,
            config.total_bytes_to_read,
            config.threads_per_block,
            target_sm_count,
            3  // warmup iterations
        );
        measurements.push_back(bw);
    }
    
    // Calculate statistics
    double sum = 0;
    for (double bw : measurements) {
        sum += bw;
    }
    double mean = sum / measurements.size();
    
    double variance = 0;
    for (double bw : measurements) {
        variance += (bw - mean) * (bw - mean);
    }
    double stdev = std::sqrt(variance / measurements.size());
    
    // Cleanup
    free_buffers(d_input, d_sink);
    destroy_green_context(handle);
    
    BandwidthResult result;
    result.num_sms = target_sm_count;
    result.mean_bw_gb_s = mean;
    result.stdev_bw_gb_s = stdev;
    
    return result;
}

// =============================================================================
// CSV Output
// =============================================================================

void write_csv(const std::string& path, const std::vector<BandwidthResult>& results) {
    FILE* f = fopen(path.c_str(), "w");
    if (!f) {
        fprintf(stderr, "Failed to open CSV file: %s\n", path.c_str());
        return;
    }
    
    fprintf(f, "num_sms,mean_bw_gb_s,stdev_bw_gb_s\n");
    for (const auto& r : results) {
        fprintf(f, "%d,%.3f,%.3f\n", r.num_sms, r.mean_bw_gb_s, r.stdev_bw_gb_s);
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
    printf("SM Bandwidth Sweep using CUDA Green Contexts\n");
    printf("=============================================================\n");
    printf("Device: %s\n", device_name);
    printf("Compute Capability: %d.%d\n", compute_major, compute_minor);
    printf("Total SMs: %d\n", total_sms);
    printf("Configuration:\n");
    printf("  SM Range: %d - %d\n", config.min_sms, 
           config.max_sms > 0 ? config.max_sms : total_sms);
    printf("  Bytes per measurement: %lu (%.2f GiB)\n", 
           config.total_bytes_to_read,
           config.total_bytes_to_read / (1024.0 * 1024.0 * 1024.0));
    printf("  Threads per block: %d\n", config.threads_per_block);
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
    std::vector<BandwidthResult> results;
    for (int sm_count : sm_counts) {
        printf("Testing with %d SMs... ", sm_count);
        fflush(stdout);
        
        BandwidthResult result = measure_bandwidth_for_sms(
            device,
            device_resource,
            sm_count,
            config
        );
        
        results.push_back(result);
        printf("%.2f ± %.2f GB/s\n", result.mean_bw_gb_s, result.stdev_bw_gb_s);
    }
    
    printf("\n");
    
    // Write CSV
    write_csv(config.csv_path, results);
    
    printf("\n=============================================================\n");
    printf("Benchmark Complete!\n");
    printf("=============================================================\n");
    
    return 0;
}

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

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
            const char* errStr = nullptr; \
            cuGetErrorString(err, &errStr); \
            fprintf(stderr, "CUDA Driver Error at %s:%d - %s\n", __FILE__, __LINE__, \
                    errStr ? errStr : "unknown"); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// =============================================================================
// Configuration
// =============================================================================

enum class BenchmarkMode {
    Global,
    Tma,
    Both,
};

const char* mode_to_string(BenchmarkMode mode) {
    switch (mode) {
        case BenchmarkMode::Global: return "global";
        case BenchmarkMode::Tma: return "tma";
        case BenchmarkMode::Both: return "both";
    }
    return "unknown";
}

BenchmarkMode parse_mode_string(const std::string& value) {
    if (value == "global") {
        return BenchmarkMode::Global;
    }
    if (value == "tma") {
        return BenchmarkMode::Tma;
    }
    if (value == "both") {
        return BenchmarkMode::Both;
    }
    fprintf(stderr, "Invalid --mode '%s'. Expected global, tma, or both.\n", value.c_str());
    exit(1);
}

struct BenchmarkConfig {
    BenchmarkMode mode;
    int min_sms;
    int max_sms;
    size_t total_bytes_to_read;
    int threads_per_block;
    int iters_override;
    int num_repeats;
    std::string csv_path;
    size_t tma_tile_bytes;
    int tma_blocks_per_sm;
    
    BenchmarkConfig() :
        mode(BenchmarkMode::Both),
        min_sms(8),
        max_sms(0),
        total_bytes_to_read(1073741824),
        threads_per_block(256),
        iters_override(-1),
        num_repeats(5),
        csv_path("./results.csv"),
        tma_tile_bytes(32768),
        tma_blocks_per_sm(2)
    {}
};

void print_usage(const char* prog_name) {
    printf("Usage: %s [options]\n", prog_name);
    printf("Options:\n");
    printf("  --mode <global|tma|both>      Benchmark mode (default: both)\n");
    printf("  --min_sms <N>                 Minimum SM count (default: 8)\n");
    printf("  --max_sms <N>                 Maximum SM count (default: device max)\n");
    printf("  --bytes <N>                   Total bytes to read/copy per measurement (default: 1073741824)\n");
    printf("  --tpb <N>                     Threads per block (default: 256)\n");
    printf("  --iters <N>                   Override global-load iterations (default: auto-calculated)\n");
    printf("  --repeats <N>                 Number of measurement repeats (default: 5)\n");
    printf("  --csv <path>                  Output CSV path (default: ./results.csv)\n");
    printf("  --tma_tile_bytes <N>          TMA global->shared bytes per block tile (default: 32768)\n");
    printf("  --tma_blocks_per_sm <N>       TMA blocks launched per green-context SM (default: 2)\n");
    printf("  --help                        Print this help\n");
}

BenchmarkConfig parse_args(int argc, char** argv) {
    BenchmarkConfig config;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help") {
            print_usage(argv[0]);
            exit(0);
        } else if (arg == "--mode" && i + 1 < argc) {
            config.mode = parse_mode_string(argv[++i]);
        } else if (arg == "--min_sms" && i + 1 < argc) {
            config.min_sms = std::atoi(argv[++i]);
        } else if (arg == "--max_sms" && i + 1 < argc) {
            config.max_sms = std::atoi(argv[++i]);
        } else if (arg == "--bytes" && i + 1 < argc) {
            config.total_bytes_to_read = std::strtoull(argv[++i], nullptr, 10);
        } else if (arg == "--tpb" && i + 1 < argc) {
            config.threads_per_block = std::atoi(argv[++i]);
        } else if (arg == "--iters" && i + 1 < argc) {
            config.iters_override = std::atoi(argv[++i]);
        } else if (arg == "--repeats" && i + 1 < argc) {
            config.num_repeats = std::atoi(argv[++i]);
        } else if (arg == "--csv" && i + 1 < argc) {
            config.csv_path = argv[++i];
        } else if (arg == "--tma_tile_bytes" && i + 1 < argc) {
            config.tma_tile_bytes = std::strtoull(argv[++i], nullptr, 10);
        } else if (arg == "--tma_blocks_per_sm" && i + 1 < argc) {
            config.tma_blocks_per_sm = std::atoi(argv[++i]);
        } else {
            fprintf(stderr, "Unknown or incomplete argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            exit(1);
        }
    }

    if (config.threads_per_block <= 0 || config.threads_per_block > 256) {
        fprintf(stderr, "--tpb must be in [1, 256]; the baseline kernel has a 256-entry shared checksum array.\n");
        exit(1);
    }
    if (config.num_repeats <= 0) {
        fprintf(stderr, "--repeats must be positive.\n");
        exit(1);
    }
    if (config.tma_blocks_per_sm <= 0) {
        fprintf(stderr, "--tma_blocks_per_sm must be positive.\n");
        exit(1);
    }
    if (config.tma_tile_bytes < 16 || (config.tma_tile_bytes % 16) != 0) {
        fprintf(stderr, "--tma_tile_bytes must be at least 16 and a multiple of 16.\n");
        exit(1);
    }
    if (config.tma_tile_bytes > ((1u << 20) - 1)) {
        fprintf(stderr, "--tma_tile_bytes must be <= 1048575 because mbarrier transaction counts are 20 bits.\n");
        exit(1);
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
    unsigned int num_groups = 1;
    CUdevResource remaining;
    
    CHECK_CU(cuDevSmResourceSplitByCount(
        &handle.partition,
        &num_groups,
        &device_resource,
        &remaining,
        0,
        target_sm_count
    ));
    
    if (num_groups == 0) {
        fprintf(stderr, "Failed to create partition with %d SMs\n", target_sm_count);
        exit(1);
    }
    
    CHECK_CU(cuDevResourceGenerateDesc(&handle.desc, &handle.partition, 1));
    CHECK_CU(cuGreenCtxCreate(&handle.green_ctx, handle.desc, device, CU_GREEN_CTX_DEFAULT_STREAM));
    CHECK_CU(cuCtxFromGreenCtx(&handle.ctx, handle.green_ctx));
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
}

// =============================================================================
// Bandwidth Measurement
// =============================================================================

struct BandwidthResult {
    std::string mode;
    int num_sms;
    double mean_bw_gb_s;
    double stdev_bw_gb_s;
};

double gb_per_second(size_t bytes, float elapsed_ms) {
    const double elapsed_s = elapsed_ms / 1000.0;
    return (bytes / (1024.0 * 1024.0 * 1024.0)) / elapsed_s;
}

void record_timed_events(cudaEvent_t* start, cudaEvent_t* stop) {
    CHECK_CUDA(cudaEventCreate(start));
    CHECK_CUDA(cudaEventCreate(stop));
}

double measure_global_bandwidth_once(
    CUstream stream,
    const Vec4U32* d_input,
    uint64_t* d_sink,
    size_t total_bytes,
    int threads_per_block,
    int num_sms,
    int iters_override,
    int warmup_iters
) {
    const size_t num_vec4_elements = total_bytes / sizeof(Vec4U32);
    const size_t threads_per_sm = threads_per_block * 2;
    const size_t total_threads = num_sms * threads_per_sm;
    const size_t num_blocks = total_threads / threads_per_block;
    int iters_per_thread = (num_vec4_elements + total_threads - 1) / total_threads;
    if (iters_override > 0) {
        iters_per_thread = iters_override;
    }
    
    for (int i = 0; i < warmup_iters; ++i) {
        bandwidth_read_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
            d_input, d_sink, num_vec4_elements, iters_per_thread
        );
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
    
    cudaEvent_t start, stop;
    record_timed_events(&start, &stop);
    
    CHECK_CUDA(cudaEventRecord(start, stream));
    bandwidth_read_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        d_input, d_sink, num_vec4_elements, iters_per_thread
    );
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    return gb_per_second(total_bytes, elapsed_ms);
}

double measure_tma_bandwidth_once(
    CUstream stream,
    const Vec4U32* d_input,
    uint64_t* d_sink,
    size_t total_bytes,
    int threads_per_block,
    int num_sms,
    size_t tile_bytes,
    int blocks_per_sm,
    int warmup_iters
) {
    const size_t rounded_total_bytes = total_bytes & ~static_cast<size_t>(15);
    const size_t num_blocks = static_cast<size_t>(num_sms) * blocks_per_sm;
    const size_t num_tiles = (rounded_total_bytes + tile_bytes - 1) / tile_bytes;
    const int tiles_per_block = static_cast<int>((num_tiles + num_blocks - 1) / num_blocks);
    const size_t dynamic_smem_bytes = tile_bytes;

    CHECK_CUDA(cudaFuncSetAttribute(
        tma_bulk_read_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(dynamic_smem_bytes)));
    
    for (int i = 0; i < warmup_iters; ++i) {
        tma_bulk_read_kernel<<<num_blocks, threads_per_block, dynamic_smem_bytes, stream>>>(
            reinterpret_cast<const unsigned char*>(d_input),
            d_sink,
            rounded_total_bytes,
            tile_bytes,
            tiles_per_block
        );
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
    
    cudaEvent_t start, stop;
    record_timed_events(&start, &stop);
    
    CHECK_CUDA(cudaEventRecord(start, stream));
    tma_bulk_read_kernel<<<num_blocks, threads_per_block, dynamic_smem_bytes, stream>>>(
        reinterpret_cast<const unsigned char*>(d_input),
        d_sink,
        rounded_total_bytes,
        tile_bytes,
        tiles_per_block
    );
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    return gb_per_second(rounded_total_bytes, elapsed_ms);
}

BandwidthResult summarize_measurements(
    const std::string& mode,
    int target_sm_count,
    const std::vector<double>& measurements
) {
    double sum = 0.0;
    for (double bw : measurements) {
        sum += bw;
    }
    const double mean = sum / measurements.size();
    
    double variance = 0.0;
    for (double bw : measurements) {
        variance += (bw - mean) * (bw - mean);
    }
    const double stdev = std::sqrt(variance / measurements.size());
    
    return BandwidthResult{mode, target_sm_count, mean, stdev};
}

BandwidthResult measure_bandwidth_for_sms(
    CUdevice device,
    const CUdevResource& device_resource,
    int target_sm_count,
    const BenchmarkConfig& config,
    BenchmarkMode mode
) {
    GreenContextHandle handle;
    create_green_context_for_sms(device, device_resource, target_sm_count, handle);
    CHECK_CU(cuCtxSetCurrent(handle.ctx));
    
    Vec4U32* d_input = nullptr;
    uint64_t* d_sink = nullptr;
    size_t max_blocks = 0;
    if (mode == BenchmarkMode::Global) {
        const size_t threads_per_sm = config.threads_per_block * 2;
        max_blocks = (target_sm_count * threads_per_sm) / config.threads_per_block;
    } else {
        max_blocks = static_cast<size_t>(target_sm_count) * config.tma_blocks_per_sm;
    }
    allocate_buffers(&d_input, &d_sink, config.total_bytes_to_read, max_blocks);
    
    std::vector<double> measurements;
    for (int rep = 0; rep < config.num_repeats; ++rep) {
        double bw = 0.0;
        if (mode == BenchmarkMode::Global) {
            bw = measure_global_bandwidth_once(
                handle.stream,
                d_input,
                d_sink,
                config.total_bytes_to_read,
                config.threads_per_block,
                target_sm_count,
                config.iters_override,
                3
            );
        } else {
            bw = measure_tma_bandwidth_once(
                handle.stream,
                d_input,
                d_sink,
                config.total_bytes_to_read,
                config.threads_per_block,
                target_sm_count,
                config.tma_tile_bytes,
                config.tma_blocks_per_sm,
                3
            );
        }
        measurements.push_back(bw);
    }
    
    free_buffers(d_input, d_sink);
    destroy_green_context(handle);
    
    return summarize_measurements(mode_to_string(mode), target_sm_count, measurements);
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
    
    fprintf(f, "mode,num_sms,mean_bw_gb_s,stdev_bw_gb_s\n");
    for (const auto& r : results) {
        fprintf(f, "%s,%d,%.3f,%.3f\n", r.mode.c_str(), r.num_sms, r.mean_bw_gb_s, r.stdev_bw_gb_s);
    }
    
    fclose(f);
    printf("Results written to: %s\n", path.c_str());
}

std::vector<BenchmarkMode> selected_modes(BenchmarkMode mode) {
    if (mode == BenchmarkMode::Both) {
        return {BenchmarkMode::Global, BenchmarkMode::Tma};
    }
    return {mode};
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    BenchmarkConfig config = parse_args(argc, argv);
    
    CHECK_CU(cuInit(0));
    
    CUdevice device;
    CHECK_CU(cuDeviceGet(&device, 0));
    
    char device_name[256];
    CHECK_CU(cuDeviceGetName(device_name, sizeof(device_name), device));
    
    int total_sms = 0;
    CHECK_CU(cuDeviceGetAttribute(&total_sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));
    
    int compute_major = 0;
    int compute_minor = 0;
    CHECK_CU(cuDeviceGetAttribute(&compute_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
    CHECK_CU(cuDeviceGetAttribute(&compute_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));

    if ((config.mode == BenchmarkMode::Tma || config.mode == BenchmarkMode::Both) && compute_major < 9) {
        fprintf(stderr, "TMA mode requires Hopper+ compute capability 9.0 or newer; device is %d.%d.\n",
                compute_major, compute_minor);
        return 1;
    }
    
    if (config.max_sms <= 0) {
        config.max_sms = total_sms;
    }
    
    printf("=============================================================\n");
    printf("SM Bandwidth Sweep using CUDA Green Contexts + TMA\n");
    printf("=============================================================\n");
    printf("Device: %s\n", device_name);
    printf("Compute Capability: %d.%d\n", compute_major, compute_minor);
    printf("Total SMs: %d\n", total_sms);
    printf("Configuration:\n");
    printf("  Mode: %s\n", mode_to_string(config.mode));
    printf("  SM Range: %d - %d\n", config.min_sms, config.max_sms);
    printf("  Bytes per measurement: %lu (%.2f GiB)\n", 
           config.total_bytes_to_read,
           config.total_bytes_to_read / (1024.0 * 1024.0 * 1024.0));
    printf("  Threads per block: %d\n", config.threads_per_block);
    printf("  Repeats: %d\n", config.num_repeats);
    printf("  TMA tile bytes: %lu\n", config.tma_tile_bytes);
    printf("  TMA blocks per SM: %d\n", config.tma_blocks_per_sm);
    printf("  CSV output: %s\n", config.csv_path.c_str());
    printf("=============================================================\n\n");
    
    CUdevResource device_resource;
    CHECK_CU(cuDeviceGetDevResource(device, &device_resource, CU_DEV_RESOURCE_TYPE_SM));
    
    printf("Device SM Resource Info:\n");
    printf("  Total SMs: %u\n", device_resource.sm.smCount);
    printf("  Min SM Partition Size: %u\n", device_resource.sm.minSmPartitionSize);
    printf("  SM Co-scheduled Alignment: %u\n", device_resource.sm.smCoscheduledAlignment);
    printf("\n");
    
    const int alignment = device_resource.sm.smCoscheduledAlignment;
    const int min_partition = device_resource.sm.minSmPartitionSize;
    
    std::vector<int> sm_counts;
    for (int sm = std::max(config.min_sms, min_partition); sm <= config.max_sms; sm += alignment) {
        sm_counts.push_back(sm);
    }
    
    if (sm_counts.empty()) {
        fprintf(stderr, "Error: No valid SM counts in range [%d, %d] with alignment %d\n",
                config.min_sms, config.max_sms, alignment);
        return 1;
    }
    
    const std::vector<BenchmarkMode> modes = selected_modes(config.mode);
    printf("Testing %zu SM configurations across %zu mode(s)...\n\n", sm_counts.size(), modes.size());
    
    std::vector<BandwidthResult> results;
    for (BenchmarkMode mode : modes) {
        for (int sm_count : sm_counts) {
            printf("Testing mode=%s with %d SMs... ", mode_to_string(mode), sm_count);
            fflush(stdout);
            
            BandwidthResult result = measure_bandwidth_for_sms(
                device,
                device_resource,
                sm_count,
                config,
                mode
            );
            
            results.push_back(result);
            printf("%.2f +/- %.2f GB/s\n", result.mean_bw_gb_s, result.stdev_bw_gb_s);
        }
        printf("\n");
    }
    
    write_csv(config.csv_path, results);
    
    printf("\n=============================================================\n");
    printf("Benchmark Complete!\n");
    printf("=============================================================\n");
    
    return 0;
}

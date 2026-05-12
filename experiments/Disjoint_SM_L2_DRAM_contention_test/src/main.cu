#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include "kernels.cuh"

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CHECK_CU(call) \
    do { \
        CUresult err = call; \
        if (err != CUDA_SUCCESS) { \
            const char* err_str = nullptr; \
            cuGetErrorString(err, &err_str); \
            fprintf(stderr, "CUDA Driver Error at %s:%d - %s\n", __FILE__, __LINE__, err_str ? err_str : "unknown"); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

enum class MemoryMode {
    Streaming,
    Tma,
};

enum class ExperimentMode {
    DurationMatched,
    ComputeUnderMemoryPressure,
};

const char* memory_mode_to_string(MemoryMode mode) {
    switch (mode) {
        case MemoryMode::Streaming: return "streaming";
        case MemoryMode::Tma: return "tma";
    }
    return "unknown";
}

const char* experiment_mode_to_string(ExperimentMode mode) {
    switch (mode) {
        case ExperimentMode::DurationMatched: return "duration_matched";
        case ExperimentMode::ComputeUnderMemoryPressure: return "compute_under_memory_pressure";
    }
    return "unknown";
}

MemoryMode parse_memory_mode(const std::string& value) {
    if (value == "streaming") {
        return MemoryMode::Streaming;
    }
    if (value == "tma") {
        return MemoryMode::Tma;
    }
    fprintf(stderr, "Invalid --mem_mode '%s'. Expected streaming or tma.\n", value.c_str());
    exit(1);
}

ExperimentMode parse_experiment_mode(const std::string& value) {
    if (value == "duration_matched") {
        return ExperimentMode::DurationMatched;
    }
    if (value == "compute_under_memory_pressure") {
        return ExperimentMode::ComputeUnderMemoryPressure;
    }
    fprintf(stderr, "Invalid --experiment '%s'. Expected duration_matched or compute_under_memory_pressure.\n",
            value.c_str());
    exit(1);
}

struct BenchmarkConfig {
    ExperimentMode experiment = ExperimentMode::DurationMatched;
    MemoryMode mem_mode = MemoryMode::Streaming;
    int mem_sms = 8;
    int compute_sms = 8;
    size_t mem_working_set_mib = 1024;
    int compute_size = 1536;
    int iterations = 5;
    int repeats = 5;
    int threads_per_block = 256;
    int blocks_per_sm = 8;
    int mma_repeats = 8;
    size_t tma_tile_bytes = 32768;
    int tma_blocks_per_sm = 2;
    bool flush_l2 = false;
    size_t l2_flush_mib = 256;
    std::string csv_path = "./results.csv";
};

struct GreenContextHandle {
    std::vector<CUdevResource> resources;
    CUdevResourceDesc desc = nullptr;
    CUgreenCtx green_ctx = nullptr;
    CUcontext ctx = nullptr;
    CUstream stream = nullptr;
    unsigned long long id = 0;
    int sm_count = 0;
};

struct MemoryBuffers {
    float4* a = nullptr;
    float4* b = nullptr;
    float4* out = nullptr;
    size_t elements = 0;
};

struct ComputeBuffers {
    __half* a = nullptr;
    __half* b = nullptr;
    float* c = nullptr;
};

struct FlushBuffer {
    float4* buf = nullptr;
    size_t elements = 0;
};

struct MemoryResult {
    double time_ms = 0.0;
    double bandwidth_gib_s = 0.0;
};

struct ComputeResult {
    double time_ms = 0.0;
    double tflops = 0.0;
};

struct ConcurrentResult {
    MemoryResult mem;
    ComputeResult compute;
    double wall_time_ms = 0.0;
    double mem_slowdown = 0.0;
    double compute_slowdown = 0.0;
    double overlap_pct = 0.0;
};

struct ContinuousPressureResult {
    ComputeResult compute;
    double compute_wall_time_ms = 0.0;
    double pressure_wall_time_ms = 0.0;
    double overlap_wall_time_ms = 0.0;
    double overlap_pct = 0.0;
    double compute_slowdown = 0.0;
    double compute_retention_pct = 0.0;
    unsigned long long memory_launches_started = 0;
    unsigned long long memory_launches_completed = 0;
    unsigned long long memory_launches_completed_before_compute_done = 0;
};

static size_t mib_to_bytes(size_t mib) {
    return mib * 1024ULL * 1024ULL;
}

static size_t round_down(size_t value, size_t multiple) {
    return (value / multiple) * multiple;
}

void print_usage(const char* prog_name) {
    printf("Usage: %s [options]\n", prog_name);
    printf("Options:\n");
    printf("  --experiment <duration_matched|compute_under_memory_pressure>\n");
    printf("                             Experiment behavior (default: duration_matched)\n");
    printf("  --mem_mode <streaming|tma> Memory workload implementation (default: streaming)\n");
    printf("  --mem_sms <N>             SMs for memory kernel (default: 8)\n");
    printf("  --compute_sms <N>         SMs for WMMA compute kernel (default: 8)\n");
    printf("  --mem_mib <N>             Total memory working set MiB across a,b,out (default: 1024)\n");
    printf("  --compute_size <N>        Square GEMM dimension, rounded to multiple of 16 (default: 1536)\n");
    printf("  --iterations <N>          Timed kernel launches per repeat (default: 5)\n");
    printf("  --repeats <N>             Measurement repeats (default: 5)\n");
    printf("  --tpb <N>                 Threads per block for memory/flush kernels (default: 256)\n");
    printf("  --blocks_per_sm <N>       Memory/flush blocks per assigned SM (default: 8)\n");
    printf("  --mma_repeats <N>         WMMA ops per loaded tile, 1..16 (default: 8)\n");
    printf("  --tma_tile_bytes <N>      TMA global-to-shared bytes per tile (default: 32768)\n");
    printf("  --tma_blocks_per_sm <N>   TMA blocks launched per assigned SM (default: 2)\n");
    printf("  --flush_l2                Flush with a streaming kernel before timed launches\n");
    printf("  --l2_flush_mib <N>        L2 flush buffer size MiB (default: 256)\n");
    printf("  --csv <path>              Output CSV path (default: ./results.csv)\n");
    printf("  --help                    Print usage\n");
}

BenchmarkConfig parse_args(int argc, char** argv) {
    BenchmarkConfig cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help") {
            print_usage(argv[0]);
            exit(0);
        } else if (arg == "--mem_mode" && i + 1 < argc) {
            cfg.mem_mode = parse_memory_mode(argv[++i]);
        } else if (arg == "--experiment" && i + 1 < argc) {
            cfg.experiment = parse_experiment_mode(argv[++i]);
        } else if (arg == "--mem_sms" && i + 1 < argc) {
            cfg.mem_sms = std::atoi(argv[++i]);
        } else if (arg == "--compute_sms" && i + 1 < argc) {
            cfg.compute_sms = std::atoi(argv[++i]);
        } else if (arg == "--mem_mib" && i + 1 < argc) {
            cfg.mem_working_set_mib = std::strtoull(argv[++i], nullptr, 10);
        } else if (arg == "--compute_size" && i + 1 < argc) {
            cfg.compute_size = std::atoi(argv[++i]);
        } else if (arg == "--iterations" && i + 1 < argc) {
            cfg.iterations = std::atoi(argv[++i]);
        } else if (arg == "--repeats" && i + 1 < argc) {
            cfg.repeats = std::atoi(argv[++i]);
        } else if (arg == "--tpb" && i + 1 < argc) {
            cfg.threads_per_block = std::atoi(argv[++i]);
        } else if (arg == "--blocks_per_sm" && i + 1 < argc) {
            cfg.blocks_per_sm = std::atoi(argv[++i]);
        } else if (arg == "--mma_repeats" && i + 1 < argc) {
            cfg.mma_repeats = std::atoi(argv[++i]);
        } else if (arg == "--tma_tile_bytes" && i + 1 < argc) {
            cfg.tma_tile_bytes = std::strtoull(argv[++i], nullptr, 10);
        } else if (arg == "--tma_blocks_per_sm" && i + 1 < argc) {
            cfg.tma_blocks_per_sm = std::atoi(argv[++i]);
        } else if (arg == "--flush_l2") {
            cfg.flush_l2 = true;
        } else if (arg == "--l2_flush_mib" && i + 1 < argc) {
            cfg.l2_flush_mib = std::strtoull(argv[++i], nullptr, 10);
        } else if (arg == "--csv" && i + 1 < argc) {
            cfg.csv_path = argv[++i];
        } else {
            fprintf(stderr, "Unknown or incomplete argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            exit(1);
        }
    }

    cfg.compute_size = ((cfg.compute_size + 15) / 16) * 16;
    cfg.mma_repeats = std::max(1, std::min(16, cfg.mma_repeats));
    if (cfg.tma_tile_bytes < 16 || (cfg.tma_tile_bytes % 16) != 0) {
        fprintf(stderr, "--tma_tile_bytes must be at least 16 and a multiple of 16.\n");
        exit(1);
    }
    const size_t max_tma_tile_bytes = (((1u << 20) - 1) / 2 / 16) * 16;
    if (cfg.tma_tile_bytes > max_tma_tile_bytes) {
        fprintf(stderr, "--tma_tile_bytes must be <= %zu because TMA mode copies both input arrays before one mbarrier wait.\n",
                max_tma_tile_bytes);
        exit(1);
    }
    if (cfg.tma_blocks_per_sm <= 0) {
        fprintf(stderr, "--tma_blocks_per_sm must be positive.\n");
        exit(1);
    }
    return cfg;
}

int gcd_int(int a, int b) {
    while (b != 0) {
        int t = a % b;
        a = b;
        b = t;
    }
    return std::abs(a);
}

void validate_config(const BenchmarkConfig& cfg, const CUdevResource& device_resource) {
    const int min_part = static_cast<int>(device_resource.sm.minSmPartitionSize);
    const int alignment = static_cast<int>(device_resource.sm.smCoscheduledAlignment);
    const int total_sms = static_cast<int>(device_resource.sm.smCount);

    auto validate_sms = [&](const char* name, int sms) {
        if (sms < min_part || sms > total_sms || sms % alignment != 0) {
            fprintf(stderr, "%s=%d is invalid. It must be between %d and %d and aligned to %d SMs.\n",
                    name, sms, min_part, total_sms, alignment);
            exit(1);
        }
    };

    validate_sms("--mem_sms", cfg.mem_sms);
    validate_sms("--compute_sms", cfg.compute_sms);

    if (cfg.mem_sms + cfg.compute_sms > total_sms) {
        fprintf(stderr, "Requested %d + %d SMs, but device resource has %d SMs.\n",
                cfg.mem_sms, cfg.compute_sms, total_sms);
        exit(1);
    }

    int base = gcd_int(cfg.mem_sms, cfg.compute_sms);
    if (base < min_part || base % alignment != 0) {
        fprintf(stderr, "Cannot form both partitions from one symmetric split: gcd(%d,%d)=%d, min=%d, alignment=%d.\n",
                cfg.mem_sms, cfg.compute_sms, base, min_part, alignment);
        exit(1);
    }

    if (cfg.iterations <= 0 || cfg.repeats <= 0 || cfg.threads_per_block <= 0 ||
        cfg.threads_per_block > 1024 || cfg.blocks_per_sm <= 0) {
        fprintf(stderr, "Invalid launch/repeat configuration.\n");
        exit(1);
    }
}

void create_disjoint_green_contexts(
    CUdevice device,
    const CUdevResource& device_resource,
    int mem_sms,
    int compute_sms,
    GreenContextHandle& mem_ctx,
    GreenContextHandle& compute_ctx
) {
    const int base_sms = gcd_int(mem_sms, compute_sms);
    const unsigned int mem_groups = static_cast<unsigned int>(mem_sms / base_sms);
    const unsigned int compute_groups = static_cast<unsigned int>(compute_sms / base_sms);
    unsigned int requested_groups = mem_groups + compute_groups;
    std::vector<CUdevResource> partitions(requested_groups);
    CUdevResource remaining;

    CHECK_CU(cuDevSmResourceSplitByCount(
        partitions.data(),
        &requested_groups,
        &device_resource,
        &remaining,
        0,
        static_cast<unsigned int>(base_sms)
    ));

    if (requested_groups < mem_groups + compute_groups) {
        fprintf(stderr, "CUDA created only %u partitions of %d SMs; need %u.\n",
                requested_groups, base_sms, mem_groups + compute_groups);
        exit(1);
    }

    mem_ctx.resources.assign(partitions.begin(), partitions.begin() + mem_groups);
    compute_ctx.resources.assign(partitions.begin() + mem_groups, partitions.begin() + mem_groups + compute_groups);
    mem_ctx.sm_count = mem_sms;
    compute_ctx.sm_count = compute_sms;

    CHECK_CU(cuDevResourceGenerateDesc(&mem_ctx.desc, mem_ctx.resources.data(), mem_groups));
    CHECK_CU(cuDevResourceGenerateDesc(&compute_ctx.desc, compute_ctx.resources.data(), compute_groups));

    CHECK_CU(cuGreenCtxCreate(&mem_ctx.green_ctx, mem_ctx.desc, device, CU_GREEN_CTX_DEFAULT_STREAM));
    CHECK_CU(cuGreenCtxCreate(&compute_ctx.green_ctx, compute_ctx.desc, device, CU_GREEN_CTX_DEFAULT_STREAM));
    CHECK_CU(cuCtxFromGreenCtx(&mem_ctx.ctx, mem_ctx.green_ctx));
    CHECK_CU(cuCtxFromGreenCtx(&compute_ctx.ctx, compute_ctx.green_ctx));
    CHECK_CU(cuGreenCtxGetId(mem_ctx.green_ctx, &mem_ctx.id));
    CHECK_CU(cuGreenCtxGetId(compute_ctx.green_ctx, &compute_ctx.id));

    CHECK_CU(cuGreenCtxStreamCreate(&mem_ctx.stream, mem_ctx.green_ctx, CU_STREAM_NON_BLOCKING, 0));
    CHECK_CU(cuGreenCtxStreamCreate(&compute_ctx.stream, compute_ctx.green_ctx, CU_STREAM_NON_BLOCKING, 0));
}

void destroy_green_context(GreenContextHandle& ctx) {
    if (ctx.stream) {
        cuStreamSynchronize(ctx.stream);
        cuStreamDestroy(ctx.stream);
        ctx.stream = nullptr;
    }
    if (ctx.green_ctx) {
        cuGreenCtxDestroy(ctx.green_ctx);
        ctx.green_ctx = nullptr;
    }
}

MemoryBuffers allocate_memory_buffers(CUcontext ctx, size_t working_set_mib) {
    CHECK_CU(cuCtxSetCurrent(ctx));
    size_t total_bytes = round_down(mib_to_bytes(working_set_mib), 3 * sizeof(float4));
    size_t per_array_bytes = total_bytes / 3;
    per_array_bytes = round_down(per_array_bytes, sizeof(float4));
    MemoryBuffers bufs;
    bufs.elements = per_array_bytes / sizeof(float4);
    CHECK_CUDA(cudaMalloc(&bufs.a, per_array_bytes));
    CHECK_CUDA(cudaMalloc(&bufs.b, per_array_bytes));
    CHECK_CUDA(cudaMalloc(&bufs.out, per_array_bytes));
    launch_stream_init(0, bufs.a, bufs.b, bufs.out, bufs.elements);
    CHECK_CUDA(cudaDeviceSynchronize());
    return bufs;
}

ComputeBuffers allocate_compute_buffers(CUcontext ctx, int n) {
    CHECK_CU(cuCtxSetCurrent(ctx));
    size_t elems = static_cast<size_t>(n) * static_cast<size_t>(n);
    ComputeBuffers bufs;
    CHECK_CUDA(cudaMalloc(&bufs.a, elems * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&bufs.b, elems * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&bufs.c, elems * sizeof(float)));
    launch_gemm_init(0, bufs.a, bufs.b, bufs.c, n);
    CHECK_CUDA(cudaDeviceSynchronize());
    return bufs;
}

FlushBuffer allocate_flush_buffer(CUcontext ctx, size_t flush_mib) {
    CHECK_CU(cuCtxSetCurrent(ctx));
    FlushBuffer flush;
    size_t bytes = round_down(mib_to_bytes(flush_mib), sizeof(float4));
    flush.elements = std::max<size_t>(1, bytes / sizeof(float4));
    CHECK_CUDA(cudaMalloc(&flush.buf, flush.elements * sizeof(float4)));
    CHECK_CUDA(cudaMemset(flush.buf, 0x5a, flush.elements * sizeof(float4)));
    return flush;
}

void free_memory_buffers(CUcontext ctx, MemoryBuffers& bufs) {
    CHECK_CU(cuCtxSetCurrent(ctx));
    if (bufs.a) CHECK_CUDA(cudaFree(bufs.a));
    if (bufs.b) CHECK_CUDA(cudaFree(bufs.b));
    if (bufs.out) CHECK_CUDA(cudaFree(bufs.out));
}

void free_compute_buffers(CUcontext ctx, ComputeBuffers& bufs) {
    CHECK_CU(cuCtxSetCurrent(ctx));
    if (bufs.a) CHECK_CUDA(cudaFree(bufs.a));
    if (bufs.b) CHECK_CUDA(cudaFree(bufs.b));
    if (bufs.c) CHECK_CUDA(cudaFree(bufs.c));
}

void free_flush_buffer(CUcontext ctx, FlushBuffer& flush) {
    CHECK_CU(cuCtxSetCurrent(ctx));
    if (flush.buf) CHECK_CUDA(cudaFree(flush.buf));
}

void run_flush(CUcontext ctx, CUstream stream, const FlushBuffer& flush, const BenchmarkConfig& cfg, int sm_count) {
    if (!cfg.flush_l2 || !flush.buf) return;
    CHECK_CU(cuCtxSetCurrent(ctx));
    launch_l2_flush(stream, flush.buf, flush.elements, sm_count, cfg.threads_per_block, cfg.blocks_per_sm);
    CHECK_CUDA(cudaStreamSynchronize(stream));
}

void launch_memory_workload(
    const GreenContextHandle& ctx,
    const MemoryBuffers& bufs,
    const BenchmarkConfig& cfg,
    float alpha
) {
    if (cfg.mem_mode == MemoryMode::Streaming) {
        launch_streaming_kernel(ctx.stream, bufs.a, bufs.b, bufs.out, bufs.elements, ctx.sm_count,
                                cfg.threads_per_block, cfg.blocks_per_sm, alpha);
    } else {
        launch_tma_memory_kernel(ctx.stream, bufs.a, bufs.b, bufs.out, bufs.elements, ctx.sm_count,
                                 cfg.threads_per_block, cfg.tma_tile_bytes, cfg.tma_blocks_per_sm,
                                 alpha);
    }
}

MemoryResult measure_memory_once(
    const GreenContextHandle& ctx,
    const MemoryBuffers& bufs,
    const FlushBuffer& flush,
    const BenchmarkConfig& cfg
) {
    CHECK_CU(cuCtxSetCurrent(ctx.ctx));
    for (int i = 0; i < 2; ++i) {
        launch_memory_workload(ctx, bufs, cfg, 1.001f);
    }
    CHECK_CUDA(cudaStreamSynchronize(ctx.stream));
    run_flush(ctx.ctx, ctx.stream, flush, cfg, ctx.sm_count);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start, ctx.stream));
    for (int i = 0; i < cfg.iterations; ++i) {
        launch_memory_workload(ctx, bufs, cfg, 1.001f + i * 0.0001f);
    }
    CHECK_CUDA(cudaEventRecord(stop, ctx.stream));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    const double bytes = static_cast<double>(bufs.elements) * sizeof(float4) * 3.0 * cfg.iterations;
    MemoryResult r;
    r.time_ms = elapsed_ms / cfg.iterations;
    r.bandwidth_gib_s = (bytes / (1024.0 * 1024.0 * 1024.0)) / (elapsed_ms / 1000.0);
    return r;
}

ComputeResult measure_compute_once(
    const GreenContextHandle& ctx,
    const ComputeBuffers& bufs,
    const FlushBuffer& flush,
    const BenchmarkConfig& cfg
) {
    CHECK_CU(cuCtxSetCurrent(ctx.ctx));
    for (int i = 0; i < 2; ++i) {
        launch_wmma_compute_kernel(ctx.stream, bufs.a, bufs.b, bufs.c, cfg.compute_size, cfg.mma_repeats);
    }
    CHECK_CUDA(cudaStreamSynchronize(ctx.stream));
    run_flush(ctx.ctx, ctx.stream, flush, cfg, ctx.sm_count);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start, ctx.stream));
    for (int i = 0; i < cfg.iterations; ++i) {
        launch_wmma_compute_kernel(ctx.stream, bufs.a, bufs.b, bufs.c, cfg.compute_size, cfg.mma_repeats);
    }
    CHECK_CUDA(cudaEventRecord(stop, ctx.stream));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    const double flops = 2.0 * cfg.compute_size * cfg.compute_size * cfg.compute_size *
                         cfg.mma_repeats * cfg.iterations;
    ComputeResult r;
    r.time_ms = elapsed_ms / cfg.iterations;
    r.tflops = flops / (elapsed_ms / 1000.0) / 1.0e12;
    return r;
}

ConcurrentResult measure_concurrent_once(
    const GreenContextHandle& mem_ctx,
    const GreenContextHandle& compute_ctx,
    const MemoryBuffers& mem_bufs,
    const ComputeBuffers& compute_bufs,
    const FlushBuffer& mem_flush,
    const FlushBuffer& compute_flush,
    const BenchmarkConfig& cfg,
    const MemoryResult& isolated_mem,
    const ComputeResult& isolated_compute
) {
    run_flush(mem_ctx.ctx, mem_ctx.stream, mem_flush, cfg, mem_ctx.sm_count);
    run_flush(compute_ctx.ctx, compute_ctx.stream, compute_flush, cfg, compute_ctx.sm_count);

    std::atomic<bool> go(false);
    std::atomic<int> ready(0);
    MemoryResult mem_result;
    ComputeResult compute_result;

    auto mem_worker = [&]() {
        CHECK_CU(cuCtxSetCurrent(mem_ctx.ctx));
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        ready.fetch_add(1);
        while (!go.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        CHECK_CUDA(cudaEventRecord(start, mem_ctx.stream));
        for (int i = 0; i < cfg.iterations; ++i) {
            launch_memory_workload(mem_ctx, mem_bufs, cfg, 1.001f + i * 0.0001f);
        }
        CHECK_CUDA(cudaEventRecord(stop, mem_ctx.stream));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float elapsed_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        const double bytes = static_cast<double>(mem_bufs.elements) * sizeof(float4) * 3.0 * cfg.iterations;
        mem_result.time_ms = elapsed_ms / cfg.iterations;
        mem_result.bandwidth_gib_s = (bytes / (1024.0 * 1024.0 * 1024.0)) / (elapsed_ms / 1000.0);
    };

    auto compute_worker = [&]() {
        CHECK_CU(cuCtxSetCurrent(compute_ctx.ctx));
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        ready.fetch_add(1);
        while (!go.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        CHECK_CUDA(cudaEventRecord(start, compute_ctx.stream));
        for (int i = 0; i < cfg.iterations; ++i) {
            launch_wmma_compute_kernel(compute_ctx.stream, compute_bufs.a, compute_bufs.b, compute_bufs.c,
                                       cfg.compute_size, cfg.mma_repeats);
        }
        CHECK_CUDA(cudaEventRecord(stop, compute_ctx.stream));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float elapsed_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        const double flops = 2.0 * cfg.compute_size * cfg.compute_size * cfg.compute_size *
                             cfg.mma_repeats * cfg.iterations;
        compute_result.time_ms = elapsed_ms / cfg.iterations;
        compute_result.tflops = flops / (elapsed_ms / 1000.0) / 1.0e12;
    };

    std::thread mem_thread(mem_worker);
    std::thread compute_thread(compute_worker);
    while (ready.load(std::memory_order_acquire) < 2) {
        std::this_thread::yield();
    }

    auto wall_start = std::chrono::steady_clock::now();
    go.store(true, std::memory_order_release);
    mem_thread.join();
    compute_thread.join();
    auto wall_end = std::chrono::steady_clock::now();

    double wall_ms_total = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
    double mem_total_ms = mem_result.time_ms * cfg.iterations;
    double compute_total_ms = compute_result.time_ms * cfg.iterations;
    double overlap_ms = std::max(0.0, mem_total_ms + compute_total_ms - wall_ms_total);

    ConcurrentResult r;
    r.mem = mem_result;
    r.compute = compute_result;
    r.wall_time_ms = wall_ms_total / cfg.iterations;
    r.mem_slowdown = mem_result.time_ms / isolated_mem.time_ms;
    r.compute_slowdown = compute_result.time_ms / isolated_compute.time_ms;
    r.overlap_pct = 100.0 * overlap_ms / std::max(mem_total_ms, compute_total_ms);
    return r;
}

ContinuousPressureResult measure_compute_under_memory_pressure_once(
    const GreenContextHandle& mem_ctx,
    const GreenContextHandle& compute_ctx,
    const MemoryBuffers& mem_bufs,
    const ComputeBuffers& compute_bufs,
    const FlushBuffer& mem_flush,
    const FlushBuffer& compute_flush,
    const BenchmarkConfig& cfg,
    const ComputeResult& isolated_compute
) {
    run_flush(mem_ctx.ctx, mem_ctx.stream, mem_flush, cfg, mem_ctx.sm_count);
    run_flush(compute_ctx.ctx, compute_ctx.stream, compute_flush, cfg, compute_ctx.sm_count);

    std::atomic<bool> go(false);
    std::atomic<bool> compute_running(true);
    std::atomic<int> ready(0);
    std::atomic<unsigned long long> memory_launches_started_atomic(0);

    ContinuousPressureResult result;
    auto zero = std::chrono::steady_clock::now();
    auto memory_start_wall = zero;
    auto memory_end_wall = zero;
    auto compute_start_wall = zero;
    auto compute_end_wall = zero;

    auto mem_worker = [&]() {
        CHECK_CU(cuCtxSetCurrent(mem_ctx.ctx));
        unsigned long long started = 0;
        unsigned long long completed = 0;
        unsigned long long completed_before_compute_done = 0;

        ready.fetch_add(1);
        while (!go.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }

        memory_start_wall = std::chrono::steady_clock::now();
        while (compute_running.load(std::memory_order_acquire)) {
            launch_memory_workload(mem_ctx, mem_bufs, cfg, 1.001f + static_cast<float>(started % 997) * 0.0001f);
            CHECK_CUDA(cudaPeekAtLastError());
            ++started;
            memory_launches_started_atomic.store(started, std::memory_order_release);

            CHECK_CUDA(cudaStreamSynchronize(mem_ctx.stream));
            ++completed;
            if (compute_running.load(std::memory_order_acquire)) {
                ++completed_before_compute_done;
            }
        }
        CHECK_CUDA(cudaStreamSynchronize(mem_ctx.stream));
        memory_end_wall = std::chrono::steady_clock::now();

        result.memory_launches_started = started;
        result.memory_launches_completed = completed;
        result.memory_launches_completed_before_compute_done = completed_before_compute_done;
    };

    auto compute_worker = [&]() {
        CHECK_CU(cuCtxSetCurrent(compute_ctx.ctx));
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        ready.fetch_add(1);
        while (!go.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        while (memory_launches_started_atomic.load(std::memory_order_acquire) == 0) {
            std::this_thread::yield();
        }

        compute_start_wall = std::chrono::steady_clock::now();
        CHECK_CUDA(cudaEventRecord(start, compute_ctx.stream));
        for (int i = 0; i < cfg.iterations; ++i) {
            launch_wmma_compute_kernel(compute_ctx.stream, compute_bufs.a, compute_bufs.b, compute_bufs.c,
                                       cfg.compute_size, cfg.mma_repeats);
        }
        CHECK_CUDA(cudaEventRecord(stop, compute_ctx.stream));
        CHECK_CUDA(cudaEventSynchronize(stop));
        compute_end_wall = std::chrono::steady_clock::now();
        compute_running.store(false, std::memory_order_release);

        float elapsed_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));

        const double flops = 2.0 * cfg.compute_size * cfg.compute_size * cfg.compute_size *
                             cfg.mma_repeats * cfg.iterations;
        result.compute.time_ms = elapsed_ms / cfg.iterations;
        result.compute.tflops = flops / (elapsed_ms / 1000.0) / 1.0e12;
    };

    std::thread mem_thread(mem_worker);
    std::thread compute_thread(compute_worker);
    while (ready.load(std::memory_order_acquire) < 2) {
        std::this_thread::yield();
    }

    go.store(true, std::memory_order_release);
    compute_thread.join();
    mem_thread.join();

    result.compute_wall_time_ms = std::chrono::duration<double, std::milli>(
        compute_end_wall - compute_start_wall).count();
    result.pressure_wall_time_ms = std::chrono::duration<double, std::milli>(
        memory_end_wall - memory_start_wall).count();

    auto overlap_start = std::max(memory_start_wall, compute_start_wall);
    auto overlap_end = std::min(memory_end_wall, compute_end_wall);
    if (overlap_end > overlap_start) {
        result.overlap_wall_time_ms = std::chrono::duration<double, std::milli>(
            overlap_end - overlap_start).count();
    }
    result.overlap_pct = result.compute_wall_time_ms > 0.0
        ? 100.0 * result.overlap_wall_time_ms / result.compute_wall_time_ms
        : 0.0;
    result.compute_slowdown = result.compute.time_ms / isolated_compute.time_ms;
    result.compute_retention_pct = 100.0 * result.compute.tflops / isolated_compute.tflops;
    return result;
}

void write_csv(
    const std::string& path,
    const BenchmarkConfig& cfg,
    size_t actual_mem_working_set_bytes,
    const MemoryResult& mem_iso,
    const ComputeResult& compute_iso,
    const ConcurrentResult& concurrent
) {
    FILE* f = fopen(path.c_str(), "w");
    if (!f) {
        fprintf(stderr, "Failed to open CSV file: %s\n", path.c_str());
        return;
    }

    fprintf(f, "mode,mem_mode,mem_sms,compute_sms,mem_working_set_mib,compute_size,iterations,repeats,mma_repeats,flush_l2,mem_bandwidth_gib_s,compute_tflops,mem_time_ms,compute_time_ms,wall_time_ms,mem_slowdown,compute_slowdown,overlap_pct\n");
    double actual_mib = actual_mem_working_set_bytes / (1024.0 * 1024.0);
    fprintf(f, "isolated_memory,%s,%d,%d,%.3f,%d,%d,%d,%d,%d,%.6f,0.0,%.6f,0.0,0.0,1.0,0.0,0.0\n",
            memory_mode_to_string(cfg.mem_mode), cfg.mem_sms, cfg.compute_sms, actual_mib, cfg.compute_size, cfg.iterations, cfg.repeats,
            cfg.mma_repeats, cfg.flush_l2 ? 1 : 0, mem_iso.bandwidth_gib_s, mem_iso.time_ms);
    fprintf(f, "isolated_compute,%s,%d,%d,%.3f,%d,%d,%d,%d,%d,0.0,%.6f,0.0,%.6f,0.0,0.0,1.0,0.0\n",
            memory_mode_to_string(cfg.mem_mode), cfg.mem_sms, cfg.compute_sms, actual_mib, cfg.compute_size, cfg.iterations, cfg.repeats,
            cfg.mma_repeats, cfg.flush_l2 ? 1 : 0, compute_iso.tflops, compute_iso.time_ms);
    fprintf(f, "concurrent,%s,%d,%d,%.3f,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.3f\n",
            memory_mode_to_string(cfg.mem_mode), cfg.mem_sms, cfg.compute_sms, actual_mib, cfg.compute_size, cfg.iterations, cfg.repeats,
            cfg.mma_repeats, cfg.flush_l2 ? 1 : 0, concurrent.mem.bandwidth_gib_s,
            concurrent.compute.tflops, concurrent.mem.time_ms, concurrent.compute.time_ms,
            concurrent.wall_time_ms, concurrent.mem_slowdown, concurrent.compute_slowdown,
            concurrent.overlap_pct);

    fclose(f);
    printf("Results written to: %s\n", path.c_str());
}

void write_continuous_pressure_csv(
    const std::string& path,
    const BenchmarkConfig& cfg,
    size_t actual_mem_working_set_bytes,
    const ComputeResult& compute_iso,
    const ContinuousPressureResult& pressure
) {
    FILE* f = fopen(path.c_str(), "w");
    if (!f) {
        fprintf(stderr, "Failed to open CSV file: %s\n", path.c_str());
        return;
    }

    double actual_mib = actual_mem_working_set_bytes / (1024.0 * 1024.0);
    fprintf(f, "mode,experiment,mem_mode,mem_sms,compute_sms,mem_working_set_mib,compute_size,iterations,repeats,mma_repeats,flush_l2,compute_time_ms,compute_tflops,compute_slowdown,compute_retention_pct,memory_launches_started,memory_launches_completed,memory_launches_completed_before_compute_done,compute_wall_time_ms,pressure_wall_time_ms,overlap_wall_time_ms,overlap_pct\n");
    fprintf(f, "isolated_compute,%s,%s,%d,%d,%.3f,%d,%d,%d,%d,%d,%.6f,%.6f,1.0,100.0,0,0,0,0.0,0.0,0.0,0.0\n",
            experiment_mode_to_string(cfg.experiment), memory_mode_to_string(cfg.mem_mode),
            cfg.mem_sms, cfg.compute_sms, actual_mib, cfg.compute_size, cfg.iterations, cfg.repeats,
            cfg.mma_repeats, cfg.flush_l2 ? 1 : 0, compute_iso.time_ms, compute_iso.tflops);
    fprintf(f, "compute_under_memory_pressure,%s,%s,%d,%d,%.3f,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.3f,%llu,%llu,%llu,%.6f,%.6f,%.6f,%.3f\n",
            experiment_mode_to_string(cfg.experiment), memory_mode_to_string(cfg.mem_mode),
            cfg.mem_sms, cfg.compute_sms, actual_mib, cfg.compute_size, cfg.iterations, cfg.repeats,
            cfg.mma_repeats, cfg.flush_l2 ? 1 : 0, pressure.compute.time_ms, pressure.compute.tflops,
            pressure.compute_slowdown, pressure.compute_retention_pct,
            pressure.memory_launches_started, pressure.memory_launches_completed,
            pressure.memory_launches_completed_before_compute_done, pressure.compute_wall_time_ms,
            pressure.pressure_wall_time_ms, pressure.overlap_wall_time_ms, pressure.overlap_pct);

    fclose(f);
    printf("Results written to: %s\n", path.c_str());
}

int main(int argc, char** argv) {
    BenchmarkConfig cfg = parse_args(argc, argv);

    CHECK_CU(cuInit(0));
    CUdevice device;
    CHECK_CU(cuDeviceGet(&device, 0));

    char device_name[256];
    CHECK_CU(cuDeviceGetName(device_name, sizeof(device_name), device));

    int cc_major = 0;
    int cc_minor = 0;
    CHECK_CU(cuDeviceGetAttribute(&cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
    CHECK_CU(cuDeviceGetAttribute(&cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));

    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

    CUdevResource device_resource;
    CHECK_CU(cuDeviceGetDevResource(device, &device_resource, CU_DEV_RESOURCE_TYPE_SM));
    validate_config(cfg, device_resource);
    if (cfg.mem_mode == MemoryMode::Tma && cc_major < 9) {
        fprintf(stderr, "TMA memory mode requires Hopper+ compute capability 9.0 or newer; device is %d.%d.\n",
                cc_major, cc_minor);
        return 1;
    }

    printf("=============================================================\n");
    printf("Disjoint SM L2/DRAM Contention Test\n");
    printf("=============================================================\n");
    printf("Device: %s, compute capability %d.%d\n", device_name, cc_major, cc_minor);
    printf("Device SM resource: %u SMs, min partition %u, co-scheduled alignment %u\n",
           device_resource.sm.smCount,
           device_resource.sm.minSmPartitionSize,
           device_resource.sm.smCoscheduledAlignment);
    printf("Device L2 cache size reported by runtime: %.2f MiB\n", prop.l2CacheSize / (1024.0 * 1024.0));
    printf("Configuration:\n");
    printf("  Experiment:        %s\n", experiment_mode_to_string(cfg.experiment));
    printf("  Memory mode:       %s\n", memory_mode_to_string(cfg.mem_mode));
    printf("  Memory partition:  %d SMs\n", cfg.mem_sms);
    printf("  Compute partition: %d SMs\n", cfg.compute_sms);
    printf("  Memory working set: %zu MiB total across two read arrays and one write array\n",
           cfg.mem_working_set_mib);
    printf("  Compute size: %d x %d x %d, WMMA repeats per loaded tile: %d\n",
           cfg.compute_size, cfg.compute_size, cfg.compute_size, cfg.mma_repeats);
    printf("  Iterations/repeats: %d / %d\n", cfg.iterations, cfg.repeats);
    if (cfg.mem_mode == MemoryMode::Tma) {
        printf("  TMA tile bytes / blocks per SM: %zu / %d\n", cfg.tma_tile_bytes, cfg.tma_blocks_per_sm);
    }
    printf("  L2 flush: %s", cfg.flush_l2 ? "enabled" : "disabled");
    if (cfg.flush_l2) printf(" (%zu MiB per context)", cfg.l2_flush_mib);
    printf("\n");
    printf("=============================================================\n\n");

    GreenContextHandle mem_ctx;
    GreenContextHandle compute_ctx;
    create_disjoint_green_contexts(device, device_resource, cfg.mem_sms, cfg.compute_sms, mem_ctx, compute_ctx);
    printf("Created disjoint Green Contexts: memory id=%llu, compute id=%llu\n", mem_ctx.id, compute_ctx.id);

    if (cfg.mem_mode == MemoryMode::Tma) {
        CHECK_CU(cuCtxSetCurrent(mem_ctx.ctx));
        CHECK_CUDA(configure_tma_memory_kernel(cfg.tma_tile_bytes));
    }

    MemoryBuffers mem_bufs = allocate_memory_buffers(mem_ctx.ctx, cfg.mem_working_set_mib);
    ComputeBuffers compute_bufs = allocate_compute_buffers(compute_ctx.ctx, cfg.compute_size);
    FlushBuffer mem_flush;
    FlushBuffer compute_flush;
    if (cfg.flush_l2) {
        mem_flush = allocate_flush_buffer(mem_ctx.ctx, cfg.l2_flush_mib);
        compute_flush = allocate_flush_buffer(compute_ctx.ctx, cfg.l2_flush_mib);
    }

    size_t actual_mem_working_set_bytes = mem_bufs.elements * sizeof(float4) * 3;
    printf("Actual %s working set: %.3f MiB\n\n",
           memory_mode_to_string(cfg.mem_mode),
           actual_mem_working_set_bytes / (1024.0 * 1024.0));

    if (cfg.experiment == ExperimentMode::ComputeUnderMemoryPressure) {
        std::vector<ComputeResult> compute_isolated_results;
        std::vector<ContinuousPressureResult> pressure_results;

        for (int rep = 0; rep < cfg.repeats; ++rep) {
            printf("Repeat %d/%d\n", rep + 1, cfg.repeats);
            ComputeResult compute = measure_compute_once(compute_ctx, compute_bufs, compute_flush, cfg);
            ContinuousPressureResult pressure = measure_compute_under_memory_pressure_once(
                mem_ctx, compute_ctx, mem_bufs, compute_bufs, mem_flush, compute_flush,
                cfg, compute);
            compute_isolated_results.push_back(compute);
            pressure_results.push_back(pressure);

            printf("  [Compute only]     %.2f TFLOP/s, %.3f ms/iter\n", compute.tflops, compute.time_ms);
            printf("  [Under pressure]   %.2f TFLOP/s (%.1f%% retained, %.2fx), %.3f ms/iter\n",
                   pressure.compute.tflops, pressure.compute_retention_pct,
                   pressure.compute_slowdown, pressure.compute.time_ms);
            printf("  [Memory pressure]  launches started/completed %llu/%llu, completed before compute done %llu, overlap %.1f%%\n\n",
                   pressure.memory_launches_started, pressure.memory_launches_completed,
                   pressure.memory_launches_completed_before_compute_done, pressure.overlap_pct);
        }

        ComputeResult avg_compute;
        ContinuousPressureResult avg_pressure;
        unsigned long long total_started = 0;
        unsigned long long total_completed = 0;
        unsigned long long total_completed_before = 0;
        for (const auto& r : compute_isolated_results) {
            avg_compute.time_ms += r.time_ms;
            avg_compute.tflops += r.tflops;
        }
        for (const auto& r : pressure_results) {
            avg_pressure.compute.time_ms += r.compute.time_ms;
            avg_pressure.compute.tflops += r.compute.tflops;
            avg_pressure.compute_wall_time_ms += r.compute_wall_time_ms;
            avg_pressure.pressure_wall_time_ms += r.pressure_wall_time_ms;
            avg_pressure.overlap_wall_time_ms += r.overlap_wall_time_ms;
            avg_pressure.overlap_pct += r.overlap_pct;
            avg_pressure.compute_slowdown += r.compute_slowdown;
            avg_pressure.compute_retention_pct += r.compute_retention_pct;
            total_started += r.memory_launches_started;
            total_completed += r.memory_launches_completed;
            total_completed_before += r.memory_launches_completed_before_compute_done;
        }

        double denom = static_cast<double>(cfg.repeats);
        avg_compute.time_ms /= denom;
        avg_compute.tflops /= denom;
        avg_pressure.compute.time_ms /= denom;
        avg_pressure.compute.tflops /= denom;
        avg_pressure.compute_wall_time_ms /= denom;
        avg_pressure.pressure_wall_time_ms /= denom;
        avg_pressure.overlap_wall_time_ms /= denom;
        avg_pressure.overlap_pct /= denom;
        avg_pressure.compute_slowdown /= denom;
        avg_pressure.compute_retention_pct /= denom;
        avg_pressure.memory_launches_started = static_cast<unsigned long long>(std::llround(total_started / denom));
        avg_pressure.memory_launches_completed = static_cast<unsigned long long>(std::llround(total_completed / denom));
        avg_pressure.memory_launches_completed_before_compute_done =
            static_cast<unsigned long long>(std::llround(total_completed_before / denom));

        printf("=============================================================\n");
        printf("Averages\n");
        printf("=============================================================\n");
        printf("Compute isolated:        %.2f TFLOP/s, %.3f ms/iter\n",
               avg_compute.tflops, avg_compute.time_ms);
        printf("Compute under pressure:  %.2f TFLOP/s, %.3f ms/iter, slowdown %.2fx, retention %.1f%%\n",
               avg_pressure.compute.tflops, avg_pressure.compute.time_ms,
               avg_pressure.compute_slowdown, avg_pressure.compute_retention_pct);
        printf("Memory pressure launches: started/completed %llu/%llu, completed before compute done %llu\n",
               avg_pressure.memory_launches_started, avg_pressure.memory_launches_completed,
               avg_pressure.memory_launches_completed_before_compute_done);
        printf("Pressure wall: %.3f ms, compute wall %.3f ms, wall overlap %.1f%%\n\n",
               avg_pressure.pressure_wall_time_ms, avg_pressure.compute_wall_time_ms,
               avg_pressure.overlap_pct);

        write_continuous_pressure_csv(cfg.csv_path, cfg, actual_mem_working_set_bytes,
                                      avg_compute, avg_pressure);

        if (cfg.flush_l2) {
            free_flush_buffer(mem_ctx.ctx, mem_flush);
            free_flush_buffer(compute_ctx.ctx, compute_flush);
        }
        free_memory_buffers(mem_ctx.ctx, mem_bufs);
        free_compute_buffers(compute_ctx.ctx, compute_bufs);
        destroy_green_context(mem_ctx);
        destroy_green_context(compute_ctx);

        return 0;
    }

    std::vector<MemoryResult> mem_isolated_results;
    std::vector<ComputeResult> compute_isolated_results;
    std::vector<ConcurrentResult> concurrent_results;

    for (int rep = 0; rep < cfg.repeats; ++rep) {
        printf("Repeat %d/%d\n", rep + 1, cfg.repeats);
        MemoryResult mem = measure_memory_once(mem_ctx, mem_bufs, mem_flush, cfg);
        ComputeResult compute = measure_compute_once(compute_ctx, compute_bufs, compute_flush, cfg);
        ConcurrentResult concurrent = measure_concurrent_once(
            mem_ctx, compute_ctx, mem_bufs, compute_bufs, mem_flush, compute_flush,
            cfg, mem, compute);
        mem_isolated_results.push_back(mem);
        compute_isolated_results.push_back(compute);
        concurrent_results.push_back(concurrent);

        printf("  [Memory only]   %.2f GiB/s, %.3f ms/iter\n", mem.bandwidth_gib_s, mem.time_ms);
        printf("  [Compute only]  %.2f TFLOP/s, %.3f ms/iter\n", compute.tflops, compute.time_ms);
        printf("  [Concurrent]    memory %.2f GiB/s (%.1f%% retained, %.2fx), compute %.2f TFLOP/s (%.1f%% retained, %.2fx), overlap %.1f%%\n\n",
               concurrent.mem.bandwidth_gib_s,
               100.0 * concurrent.mem.bandwidth_gib_s / mem.bandwidth_gib_s,
               concurrent.mem_slowdown,
               concurrent.compute.tflops,
               100.0 * concurrent.compute.tflops / compute.tflops,
               concurrent.compute_slowdown,
               concurrent.overlap_pct);
    }

    MemoryResult avg_mem;
    ComputeResult avg_compute;
    ConcurrentResult avg_concurrent;
    for (const auto& r : mem_isolated_results) {
        avg_mem.time_ms += r.time_ms;
        avg_mem.bandwidth_gib_s += r.bandwidth_gib_s;
    }
    for (const auto& r : compute_isolated_results) {
        avg_compute.time_ms += r.time_ms;
        avg_compute.tflops += r.tflops;
    }
    for (const auto& r : concurrent_results) {
        avg_concurrent.mem.time_ms += r.mem.time_ms;
        avg_concurrent.mem.bandwidth_gib_s += r.mem.bandwidth_gib_s;
        avg_concurrent.compute.time_ms += r.compute.time_ms;
        avg_concurrent.compute.tflops += r.compute.tflops;
        avg_concurrent.wall_time_ms += r.wall_time_ms;
        avg_concurrent.mem_slowdown += r.mem_slowdown;
        avg_concurrent.compute_slowdown += r.compute_slowdown;
        avg_concurrent.overlap_pct += r.overlap_pct;
    }

    double denom = static_cast<double>(cfg.repeats);
    avg_mem.time_ms /= denom;
    avg_mem.bandwidth_gib_s /= denom;
    avg_compute.time_ms /= denom;
    avg_compute.tflops /= denom;
    avg_concurrent.mem.time_ms /= denom;
    avg_concurrent.mem.bandwidth_gib_s /= denom;
    avg_concurrent.compute.time_ms /= denom;
    avg_concurrent.compute.tflops /= denom;
    avg_concurrent.wall_time_ms /= denom;
    avg_concurrent.mem_slowdown /= denom;
    avg_concurrent.compute_slowdown /= denom;
    avg_concurrent.overlap_pct /= denom;

    printf("=============================================================\n");
    printf("Averages\n");
    printf("=============================================================\n");
    printf("Memory isolated:    %.2f GiB/s, %.3f ms/iter\n", avg_mem.bandwidth_gib_s, avg_mem.time_ms);
    printf("Compute isolated:   %.2f TFLOP/s, %.3f ms/iter\n", avg_compute.tflops, avg_compute.time_ms);
    printf("Memory concurrent:  %.2f GiB/s, %.3f ms/iter, slowdown %.2fx, retention %.1f%%\n",
           avg_concurrent.mem.bandwidth_gib_s, avg_concurrent.mem.time_ms,
           avg_concurrent.mem_slowdown, 100.0 * avg_concurrent.mem.bandwidth_gib_s / avg_mem.bandwidth_gib_s);
    printf("Compute concurrent: %.2f TFLOP/s, %.3f ms/iter, slowdown %.2fx, retention %.1f%%\n",
           avg_concurrent.compute.tflops, avg_concurrent.compute.time_ms,
           avg_concurrent.compute_slowdown, 100.0 * avg_concurrent.compute.tflops / avg_compute.tflops);
    printf("Concurrent wall:    %.3f ms/iter, estimated overlap %.1f%%\n\n",
           avg_concurrent.wall_time_ms, avg_concurrent.overlap_pct);

    write_csv(cfg.csv_path, cfg, actual_mem_working_set_bytes, avg_mem, avg_compute, avg_concurrent);

    if (cfg.flush_l2) {
        free_flush_buffer(mem_ctx.ctx, mem_flush);
        free_flush_buffer(compute_ctx.ctx, compute_flush);
    }
    free_memory_buffers(mem_ctx.ctx, mem_bufs);
    free_compute_buffers(compute_ctx.ctx, compute_bufs);
    destroy_green_context(mem_ctx);
    destroy_green_context(compute_ctx);

    return 0;
}

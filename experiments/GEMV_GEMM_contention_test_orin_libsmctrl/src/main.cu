#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "kernels.cuh"
#include "libsmctrl.h"

#define CHECK_CUDA(call)                                                                  \
    do {                                                                                  \
        cudaError_t err__ = (call);                                                       \
        if (err__ != cudaSuccess) {                                                       \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,              \
                    cudaGetErrorString(err__));                                           \
            std::exit(EXIT_FAILURE);                                                      \
        }                                                                                 \
    } while (0)

#define CHECK_CU(call)                                                                    \
    do {                                                                                  \
        CUresult err__ = (call);                                                          \
        if (err__ != CUDA_SUCCESS) {                                                      \
            const char* err_str__ = nullptr;                                              \
            cuGetErrorString(err__, &err_str__);                                          \
            fprintf(stderr, "CUDA driver error at %s:%d: %s\n", __FILE__, __LINE__,      \
                    err_str__ ? err_str__ : "unknown");                                   \
            std::exit(EXIT_FAILURE);                                                      \
        }                                                                                 \
    } while (0)

struct BenchmarkConfig {
    int min_sms = 2;
    int max_sms = -1;
    int step_sms = -1;
    int gemv_M = 8192;
    int gemv_N = 8192;
    int gemm_size = 1024;
    int num_iters = 8;
    int tpb_gemv = 256;
    int tpb_gemm = 256;
    int num_repeats = 5;
    bool perfect_sync = false;
    std::string csv_path = "./results.csv";
};

struct RuntimeInfo {
    int total_sms = 0;
    uint32_t total_tpcs = 0;
    int sms_per_tpc = 1;
};

struct GemvResult {
    double gflops = 0.0;
    double time_ms = 0.0;
};

struct GemmResult {
    double gflops = 0.0;
    double time_ms = 0.0;
};

struct ConcurrentResult {
    GemvResult gemv;
    GemmResult gemm;
    double wall_time_ms = 0.0;
    double gemv_slowdown = 0.0;
    double gemm_slowdown = 0.0;
    double overlap_pct = 0.0;
};

struct FullBenchmarkResult {
    int num_sms = 0;
    int tpc_count = 0;
    int gemv_M = 0;
    int gemv_N = 0;
    int gemm_size = 0;
    GemvResult isolated_gemv;
    GemmResult isolated_gemm;
    ConcurrentResult concurrent;
};

struct Buffers {
    __half* gemv_A = nullptr;
    __half* gemv_x = nullptr;
    __half* gemv_y = nullptr;
    __half* gemm_A = nullptr;
    __half* gemm_B = nullptr;
    __half* gemm_C = nullptr;
};

void print_usage(const char* prog_name) {
    printf("Usage: %s [options]\n", prog_name);
    printf("Options:\n");
    printf("  --min_sms <N>       Minimum SM count to test (default: 2)\n");
    printf("  --max_sms <N>       Maximum SM count to test (default: all available)\n");
    printf("  --step_sms <N>      Sweep increment in SMs (default: SMs per TPC)\n");
    printf("  --gemv_M <N>        GEMV matrix rows (default: 8192)\n");
    printf("  --gemv_N <N>        GEMV matrix cols (default: 8192)\n");
    printf("  --gemm_size <N>     Matrix dimension for GEMM (default: 1024)\n");
    printf("  --num_iters <N>     Kernel launches per measurement (default: 8)\n");
    printf("  --tpb_gemv <N>      Threads per block for GEMV (default: 256)\n");
    printf("  --tpb_gemm <N>      Compatibility option, unused by WMMA launch (default: 256)\n");
    printf("  --repeats <N>       Number of repeats per configuration (default: 5)\n");
    printf("  --perfect_sync      Use device synchronization before paired launches\n");
    printf("  --csv <path>        Output CSV path (default: ./results.csv)\n");
    printf("  --help              Print usage\n");
}

BenchmarkConfig parse_args(int argc, char** argv) {
    BenchmarkConfig cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "--min_sms" && i + 1 < argc) {
            cfg.min_sms = std::atoi(argv[++i]);
        } else if (arg == "--max_sms" && i + 1 < argc) {
            cfg.max_sms = std::atoi(argv[++i]);
        } else if (arg == "--step_sms" && i + 1 < argc) {
            cfg.step_sms = std::atoi(argv[++i]);
        } else if (arg == "--gemv_M" && i + 1 < argc) {
            cfg.gemv_M = std::atoi(argv[++i]);
        } else if (arg == "--gemv_N" && i + 1 < argc) {
            cfg.gemv_N = std::atoi(argv[++i]);
        } else if (arg == "--gemm_size" && i + 1 < argc) {
            cfg.gemm_size = std::atoi(argv[++i]);
        } else if (arg == "--num_iters" && i + 1 < argc) {
            cfg.num_iters = std::atoi(argv[++i]);
        } else if (arg == "--tpb_gemv" && i + 1 < argc) {
            cfg.tpb_gemv = std::atoi(argv[++i]);
        } else if (arg == "--tpb_gemm" && i + 1 < argc) {
            cfg.tpb_gemm = std::atoi(argv[++i]);
        } else if (arg == "--repeats" && i + 1 < argc) {
            cfg.num_repeats = std::atoi(argv[++i]);
        } else if (arg == "--perfect_sync") {
            cfg.perfect_sync = true;
        } else if (arg == "--csv" && i + 1 < argc) {
            cfg.csv_path = argv[++i];
        } else {
            fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            std::exit(EXIT_FAILURE);
        }
    }
    return cfg;
}

int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

std::vector<int> make_tpc_sweep(const BenchmarkConfig& cfg, const RuntimeInfo& runtime) {
    const int step_sms = cfg.step_sms > 0 ? cfg.step_sms : runtime.sms_per_tpc;
    const int min_sms = std::max(1, cfg.min_sms);
    const int max_sms = cfg.max_sms > 0 ? std::min(cfg.max_sms, runtime.total_sms) : runtime.total_sms;

    const int min_tpc = std::max(1, ceil_div(min_sms, runtime.sms_per_tpc));
    const int max_tpc = std::max(min_tpc, max_sms / runtime.sms_per_tpc);
    const int step_tpc = std::max(1, ceil_div(step_sms, runtime.sms_per_tpc));

    std::vector<int> tpc_counts;
    for (int tpc = min_tpc; tpc <= max_tpc; tpc += step_tpc) {
        tpc_counts.push_back(tpc);
    }
    if (!tpc_counts.empty() && tpc_counts.back() != max_tpc) {
        tpc_counts.push_back(max_tpc);
    }
    return tpc_counts;
}

uint64_t make_first_n_tpc_mask(int tpc_count) {
    uint64_t mask = 0;
    libsmctrl_make_mask(&mask, 0, tpc_count);
    return mask;
}

void allocate_buffers(const BenchmarkConfig& cfg, Buffers& buffers, cudaStream_t stream) {
    const size_t gemv_a_elems = static_cast<size_t>(cfg.gemv_M) * cfg.gemv_N;
    const size_t gemv_x_elems = static_cast<size_t>(cfg.gemv_N);
    const size_t gemv_y_elems = static_cast<size_t>(cfg.gemv_M);
    const int gemm_dim = ceil_div(cfg.gemm_size, 16) * 16;
    const size_t gemm_elems = static_cast<size_t>(gemm_dim) * gemm_dim;

    CHECK_CUDA(cudaMalloc(&buffers.gemv_A, gemv_a_elems * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&buffers.gemv_x, gemv_x_elems * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&buffers.gemv_y, gemv_y_elems * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&buffers.gemm_A, gemm_elems * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&buffers.gemm_B, gemm_elems * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&buffers.gemm_C, gemm_elems * sizeof(__half)));

    fill_half_buffer(stream, buffers.gemv_A, gemv_a_elems, 1.0f);
    fill_half_buffer(stream, buffers.gemv_x, gemv_x_elems, 1.0f);
    CHECK_CUDA(cudaMemsetAsync(buffers.gemv_y, 0, gemv_y_elems * sizeof(__half), stream));
    fill_half_buffer(stream, buffers.gemm_A, gemm_elems, 1.0f);
    fill_half_buffer(stream, buffers.gemm_B, gemm_elems, 1.0f);
    CHECK_CUDA(cudaMemsetAsync(buffers.gemm_C, 0, gemm_elems * sizeof(__half), stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
}

void free_buffers(Buffers& buffers) {
    CHECK_CUDA(cudaFree(buffers.gemv_A));
    CHECK_CUDA(cudaFree(buffers.gemv_x));
    CHECK_CUDA(cudaFree(buffers.gemv_y));
    CHECK_CUDA(cudaFree(buffers.gemm_A));
    CHECK_CUDA(cudaFree(buffers.gemm_B));
    CHECK_CUDA(cudaFree(buffers.gemm_C));
}

GemvResult measure_gemv_only(
    cudaStream_t stream,
    uint64_t mask,
    const Buffers& buffers,
    const BenchmarkConfig& cfg
) {
    for (int i = 0; i < 2; ++i) {
        for (int iter = 0; iter < cfg.num_iters; ++iter) {
            libsmctrl_set_next_mask(mask);
            launch_gemv_kernel(stream, buffers.gemv_A, buffers.gemv_x, buffers.gemv_y, cfg.gemv_M, cfg.gemv_N,
                               cfg.tpb_gemv);
        }
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    float total_time_ms = 0.0f;
    for (int iter = 0; iter < cfg.num_iters; ++iter) {
        libsmctrl_set_next_mask(mask);
        CHECK_CUDA(cudaEventRecord(start, stream));
        launch_gemv_kernel(stream, buffers.gemv_A, buffers.gemv_x, buffers.gemv_y, cfg.gemv_M, cfg.gemv_N,
                           cfg.tpb_gemv);
        CHECK_CUDA(cudaEventRecord(stop, stream));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float iter_time_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&iter_time_ms, start, stop));
        total_time_ms += iter_time_ms;
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    GemvResult result;
    result.time_ms = total_time_ms / cfg.num_iters;
    result.gflops = (2.0 * cfg.gemv_M * cfg.gemv_N) / (result.time_ms * 1e6);
    return result;
}

GemmResult measure_gemm_only(
    cudaStream_t stream,
    uint64_t mask,
    const Buffers& buffers,
    const BenchmarkConfig& cfg
) {
    const int gemm_dim = ceil_div(cfg.gemm_size, 16) * 16;
    for (int i = 0; i < 2; ++i) {
        for (int iter = 0; iter < cfg.num_iters; ++iter) {
            libsmctrl_set_next_mask(mask);
            launch_gemm_kernel(stream, buffers.gemm_A, buffers.gemm_B, buffers.gemm_C, gemm_dim, gemm_dim, gemm_dim,
                               cfg.tpb_gemm);
        }
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    float total_time_ms = 0.0f;
    for (int iter = 0; iter < cfg.num_iters; ++iter) {
        libsmctrl_set_next_mask(mask);
        CHECK_CUDA(cudaEventRecord(start, stream));
        launch_gemm_kernel(stream, buffers.gemm_A, buffers.gemm_B, buffers.gemm_C, gemm_dim, gemm_dim, gemm_dim,
                           cfg.tpb_gemm);
        CHECK_CUDA(cudaEventRecord(stop, stream));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float iter_time_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&iter_time_ms, start, stop));
        total_time_ms += iter_time_ms;
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    GemmResult result;
    result.time_ms = total_time_ms / cfg.num_iters;
    result.gflops = (2.0 * cfg.gemm_size * cfg.gemm_size * cfg.gemm_size) / (result.time_ms * 1e6);
    return result;
}

ConcurrentResult measure_concurrent(
    cudaStream_t stream_gemv,
    cudaStream_t stream_gemm,
    uint64_t mask,
    const Buffers& buffers,
    const BenchmarkConfig& cfg,
    const GemvResult& isolated_gemv,
    const GemmResult& isolated_gemm
) {
    const int gemm_dim = ceil_div(cfg.gemm_size, 16) * 16;

    cudaEvent_t gemv_start, gemv_end, gemm_start, gemm_end;
    CHECK_CUDA(cudaEventCreate(&gemv_start));
    CHECK_CUDA(cudaEventCreate(&gemv_end));
    CHECK_CUDA(cudaEventCreate(&gemm_start));
    CHECK_CUDA(cudaEventCreate(&gemm_end));

    std::vector<cudaEvent_t> sync_barriers(cfg.num_iters);
    std::vector<cudaEvent_t> gemv_done(cfg.num_iters);
    std::vector<cudaEvent_t> gemm_done(cfg.num_iters);
    for (int iter = 0; iter < cfg.num_iters; ++iter) {
        CHECK_CUDA(cudaEventCreate(&sync_barriers[iter]));
        CHECK_CUDA(cudaEventCreate(&gemv_done[iter]));
        CHECK_CUDA(cudaEventCreate(&gemm_done[iter]));
    }

    float total_gemv_ms = 0.0f;
    float total_gemm_ms = 0.0f;
    float total_wall_ms = 0.0f;

    for (int iter = 0; iter < cfg.num_iters; ++iter) {
        if (cfg.perfect_sync) {
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            CHECK_CUDA(cudaEventRecord(sync_barriers[iter], 0));
            CHECK_CUDA(cudaStreamWaitEvent(stream_gemv, sync_barriers[iter], 0));
            CHECK_CUDA(cudaStreamWaitEvent(stream_gemm, sync_barriers[iter], 0));
            if (iter > 0) {
                CHECK_CUDA(cudaStreamWaitEvent(stream_gemv, gemm_done[iter - 1], 0));
                CHECK_CUDA(cudaStreamWaitEvent(stream_gemm, gemv_done[iter - 1], 0));
            }
        }

        CHECK_CUDA(cudaEventRecord(gemv_start, stream_gemv));
        CHECK_CUDA(cudaEventRecord(gemm_start, stream_gemm));

        libsmctrl_set_next_mask(mask);
        launch_gemv_kernel(stream_gemv, buffers.gemv_A, buffers.gemv_x, buffers.gemv_y, cfg.gemv_M, cfg.gemv_N,
                           cfg.tpb_gemv);
        libsmctrl_set_next_mask(mask);
        launch_gemm_kernel(stream_gemm, buffers.gemm_A, buffers.gemm_B, buffers.gemm_C, gemm_dim, gemm_dim, gemm_dim,
                           cfg.tpb_gemm);

        CHECK_CUDA(cudaEventRecord(gemv_end, stream_gemv));
        CHECK_CUDA(cudaEventRecord(gemm_end, stream_gemm));
        CHECK_CUDA(cudaEventRecord(gemv_done[iter], stream_gemv));
        CHECK_CUDA(cudaEventRecord(gemm_done[iter], stream_gemm));

        CHECK_CUDA(cudaEventSynchronize(gemv_end));
        CHECK_CUDA(cudaEventSynchronize(gemm_end));

        float gemv_iter_ms = 0.0f;
        float gemm_iter_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&gemv_iter_ms, gemv_start, gemv_end));
        CHECK_CUDA(cudaEventElapsedTime(&gemm_iter_ms, gemm_start, gemm_end));

        total_gemv_ms += gemv_iter_ms;
        total_gemm_ms += gemm_iter_ms;
        total_wall_ms += std::max(gemv_iter_ms, gemm_iter_ms);
    }

    for (int iter = 0; iter < cfg.num_iters; ++iter) {
        CHECK_CUDA(cudaEventDestroy(sync_barriers[iter]));
        CHECK_CUDA(cudaEventDestroy(gemv_done[iter]));
        CHECK_CUDA(cudaEventDestroy(gemm_done[iter]));
    }
    CHECK_CUDA(cudaEventDestroy(gemv_start));
    CHECK_CUDA(cudaEventDestroy(gemv_end));
    CHECK_CUDA(cudaEventDestroy(gemm_start));
    CHECK_CUDA(cudaEventDestroy(gemm_end));

    ConcurrentResult result;
    result.gemv.time_ms = total_gemv_ms / cfg.num_iters;
    result.gemm.time_ms = total_gemm_ms / cfg.num_iters;
    result.wall_time_ms = total_wall_ms / cfg.num_iters;

    const double gemv_flops = 2.0 * cfg.gemv_M * cfg.gemv_N;
    const double gemm_flops = 2.0 * cfg.gemm_size * cfg.gemm_size * cfg.gemm_size;
    result.gemv.gflops = gemv_flops / (result.gemv.time_ms * 1e6);
    result.gemm.gflops = gemm_flops / (result.gemm.time_ms * 1e6);
    result.gemv_slowdown = result.gemv.time_ms / isolated_gemv.time_ms;
    result.gemm_slowdown = result.gemm.time_ms / isolated_gemm.time_ms;
    result.overlap_pct =
        100.0 * (1.0 - result.wall_time_ms / (isolated_gemv.time_ms + isolated_gemm.time_ms));
    return result;
}

template <typename T>
T average_results(const std::vector<T>& values) {
    T avg{};
    for (const auto& value : values) {
        avg.gflops += value.gflops;
        avg.time_ms += value.time_ms;
    }
    avg.gflops /= values.size();
    avg.time_ms /= values.size();
    return avg;
}

ConcurrentResult average_results(const std::vector<ConcurrentResult>& values) {
    ConcurrentResult avg{};
    for (const auto& value : values) {
        avg.gemv.gflops += value.gemv.gflops;
        avg.gemv.time_ms += value.gemv.time_ms;
        avg.gemm.gflops += value.gemm.gflops;
        avg.gemm.time_ms += value.gemm.time_ms;
        avg.wall_time_ms += value.wall_time_ms;
        avg.gemv_slowdown += value.gemv_slowdown;
        avg.gemm_slowdown += value.gemm_slowdown;
        avg.overlap_pct += value.overlap_pct;
    }
    avg.gemv.gflops /= values.size();
    avg.gemv.time_ms /= values.size();
    avg.gemm.gflops /= values.size();
    avg.gemm.time_ms /= values.size();
    avg.wall_time_ms /= values.size();
    avg.gemv_slowdown /= values.size();
    avg.gemm_slowdown /= values.size();
    avg.overlap_pct /= values.size();
    return avg;
}

FullBenchmarkResult benchmark_tpc_count(
    int tpc_count,
    const RuntimeInfo& runtime,
    const BenchmarkConfig& cfg,
    const Buffers& buffers,
    cudaStream_t stream_gemv,
    cudaStream_t stream_gemm
) {
    const uint64_t mask = make_first_n_tpc_mask(tpc_count);

    std::vector<GemvResult> gemv_results;
    std::vector<GemmResult> gemm_results;
    std::vector<ConcurrentResult> concurrent_results;

    gemv_results.reserve(cfg.num_repeats);
    gemm_results.reserve(cfg.num_repeats);
    concurrent_results.reserve(cfg.num_repeats);

    for (int rep = 0; rep < cfg.num_repeats; ++rep) {
        GemvResult gemv = measure_gemv_only(stream_gemv, mask, buffers, cfg);
        GemmResult gemm = measure_gemm_only(stream_gemm, mask, buffers, cfg);
        ConcurrentResult concurrent =
            measure_concurrent(stream_gemv, stream_gemm, mask, buffers, cfg, gemv, gemm);

        gemv_results.push_back(gemv);
        gemm_results.push_back(gemm);
        concurrent_results.push_back(concurrent);
    }

    FullBenchmarkResult result;
    result.num_sms = tpc_count * runtime.sms_per_tpc;
    result.tpc_count = tpc_count;
    result.gemv_M = cfg.gemv_M;
    result.gemv_N = cfg.gemv_N;
    result.gemm_size = cfg.gemm_size;
    result.isolated_gemv = average_results(gemv_results);
    result.isolated_gemm = average_results(gemm_results);
    result.concurrent = average_results(concurrent_results);
    return result;
}

void write_csv(const std::string& path, const std::vector<FullBenchmarkResult>& results, const RuntimeInfo& runtime) {
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) {
        fprintf(stderr, "Failed to open CSV file: %s\n", path.c_str());
        std::exit(EXIT_FAILURE);
    }

    fprintf(f,
            "num_sms,tpc_count,sms_per_tpc,gemv_M,gemv_N,gemm_size,mode,gemv_gflops,gemm_gflops,"
            "gemv_time_ms,gemm_time_ms,gemv_slowdown,gemm_slowdown,overlap_pct\n");

    for (const auto& r : results) {
        fprintf(f, "%d,%d,%d,%d,%d,%d,isolated_gemv,%.3f,0.0,%.3f,0.0,1.0,0.0,0.0\n", r.num_sms, r.tpc_count,
                runtime.sms_per_tpc, r.gemv_M, r.gemv_N, r.gemm_size, r.isolated_gemv.gflops,
                r.isolated_gemv.time_ms);
        fprintf(f, "%d,%d,%d,%d,%d,%d,isolated_gemm,0.0,%.3f,0.0,%.3f,0.0,1.0,0.0\n", r.num_sms, r.tpc_count,
                runtime.sms_per_tpc, r.gemv_M, r.gemv_N, r.gemm_size, r.isolated_gemm.gflops,
                r.isolated_gemm.time_ms);
        fprintf(f, "%d,%d,%d,%d,%d,%d,concurrent,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n", r.num_sms, r.tpc_count,
                runtime.sms_per_tpc, r.gemv_M, r.gemv_N, r.gemm_size, r.concurrent.gemv.gflops,
                r.concurrent.gemm.gflops, r.concurrent.gemv.time_ms, r.concurrent.gemm.time_ms,
                r.concurrent.gemv_slowdown, r.concurrent.gemm_slowdown, r.concurrent.overlap_pct);
    }

    std::fclose(f);
}

int main(int argc, char** argv) {
    BenchmarkConfig cfg = parse_args(argc, argv);

    CHECK_CU(cuInit(0));
    CHECK_CUDA(cudaSetDevice(0));

    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

    RuntimeInfo runtime;
    runtime.total_sms = prop.multiProcessorCount;
    int libsmctrl_status = libsmctrl_get_tpc_info_cuda(&runtime.total_tpcs, 0);
    if (libsmctrl_status != 0) {
        fprintf(stderr, "libsmctrl_get_tpc_info_cuda failed with code %d\n", libsmctrl_status);
        return EXIT_FAILURE;
    }
    runtime.sms_per_tpc = std::max(1, runtime.total_sms / static_cast<int>(runtime.total_tpcs));

    if (cfg.max_sms <= 0) {
        cfg.max_sms = runtime.total_sms;
    }

    std::vector<int> tpc_sweep = make_tpc_sweep(cfg, runtime);
    if (tpc_sweep.empty()) {
        fprintf(stderr, "No valid TPC-aligned sweep points for requested SM range.\n");
        return EXIT_FAILURE;
    }

    printf("=============================================================\n");
    printf("GEMV-GEMM Contention Test (Orin/libsmctrl)\n");
    printf("=============================================================\n");
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("CUDA Runtime: %d.%d\n", CUDART_VERSION / 1000, (CUDART_VERSION % 1000) / 10);
    printf("Total SMs: %d\n", runtime.total_sms);
    printf("Total TPCs: %u\n", runtime.total_tpcs);
    printf("SMs per TPC: %d\n", runtime.sms_per_tpc);
    printf("Requested SM sweep: %d - %d (step %d)\n", cfg.min_sms, cfg.max_sms,
           cfg.step_sms > 0 ? cfg.step_sms : runtime.sms_per_tpc);
    printf("Actual masked SM sweep:");
    for (size_t i = 0; i < tpc_sweep.size(); ++i) {
        printf(" %d", tpc_sweep[i] * runtime.sms_per_tpc);
    }
    printf("\n");
    printf("Unified Iterations: %d\n", cfg.num_iters);
    printf("Repeats: %d\n", cfg.num_repeats);
    printf("GEMV: [%d x %d] FP16\n", cfg.gemv_M, cfg.gemv_N);
    printf("GEMM: [%d x %d x %d] FP16 WMMA\n", cfg.gemm_size, cfg.gemm_size, cfg.gemm_size);
    printf("CSV output: %s\n", cfg.csv_path.c_str());
    printf("=============================================================\n\n");

    cudaStream_t stream_gemv;
    cudaStream_t stream_gemm;
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream_gemv, cudaStreamNonBlocking));
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream_gemm, cudaStreamNonBlocking));

    Buffers buffers;
    allocate_buffers(cfg, buffers, stream_gemv);

    std::vector<FullBenchmarkResult> results;
    results.reserve(tpc_sweep.size());

    for (int tpc_count : tpc_sweep) {
        const int active_sms = tpc_count * runtime.sms_per_tpc;
        printf("Testing with %d SMs (%d TPCs):\n", active_sms, tpc_count);

        FullBenchmarkResult result = benchmark_tpc_count(tpc_count, runtime, cfg, buffers, stream_gemv, stream_gemm);
        results.push_back(result);

        printf("  [GEMV Only]  Throughput: %.2f GFLOPS, Time: %.2f ms\n", result.isolated_gemv.gflops,
               result.isolated_gemv.time_ms);
        printf("  [GEMM Only]  Throughput: %.2f GFLOPS, Time: %.2f ms\n", result.isolated_gemm.gflops,
               result.isolated_gemm.time_ms);
        printf("  [Concurrent] Wall Time: %.2f ms (overlap: %.1f%%)\n", result.concurrent.wall_time_ms,
               result.concurrent.overlap_pct);
        printf("               GEMV: %.2f GFLOPS (%.1f%% retained), Time: %.2f ms, Slowdown: %.2fx\n",
               result.concurrent.gemv.gflops,
               100.0 * result.concurrent.gemv.gflops / result.isolated_gemv.gflops, result.concurrent.gemv.time_ms,
               result.concurrent.gemv_slowdown);
        printf("               GEMM: %.2f GFLOPS (%.1f%% retained), Time: %.2f ms, Slowdown: %.2fx\n",
               result.concurrent.gemm.gflops,
               100.0 * result.concurrent.gemm.gflops / result.isolated_gemm.gflops, result.concurrent.gemm.time_ms,
               result.concurrent.gemm_slowdown);
        printf("\n");
    }

    write_csv(cfg.csv_path, results, runtime);
    printf("Results written to: %s\n", cfg.csv_path.c_str());

    free_buffers(buffers);
    CHECK_CUDA(cudaStreamDestroy(stream_gemv));
    CHECK_CUDA(cudaStreamDestroy(stream_gemm));

    printf("\nBenchmark Complete.\n");
    return 0;
}

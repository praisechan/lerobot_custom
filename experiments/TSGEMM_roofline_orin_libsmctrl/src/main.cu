#include <cuda.h>
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

struct Config {
    int min_sms = 2;
    int max_sms = -1;
    int step_sms = -1;
    int row_size = 8192;
    int K = 4096;
    int N = 16;
    size_t bandwidth_bytes = 1ULL * 1024 * 1024 * 1024;
    int bandwidth_tpb = 256;
    int compute_repeats = 4096;
    int real_iters = 8;
    int repeats = 5;
    std::string csv_path = "./results.csv";
};

struct RuntimeInfo {
    int total_sms = 0;
    uint32_t total_tpcs = 0;
    int sms_per_tpc = 1;
};

struct Buffers {
    Vec4U32* bw_input = nullptr;
    uint64_t* bw_sink = nullptr;
    __half* A = nullptr;
    __half* B = nullptr;
    __half* C = nullptr;
    float* compute_sink = nullptr;
};

struct RowResult {
    int num_sms = 0;
    int tpc_count = 0;
    int sms_per_tpc = 0;
    int row_size = 0;
    int K = 0;
    int N = 0;
    double arithmetic_intensity = 0.0;
    double total_bytes = 0.0;
    double bandwidth_roof_gbps = 0.0;
    double compute_roof_gflops = 0.0;
    double memory_roof_gflops = 0.0;
    double predicted_gflops = 0.0;
    double required_full_bw_gflops = 0.0;
    double real_gflops = 0.0;
    double real_time_ms = 0.0;
    double implied_real_bw_gbps = 0.0;
};

void print_usage(const char* prog_name) {
    printf("Usage: %s [options]\n", prog_name);
    printf("Options:\n");
    printf("  --min_sms <N>            Minimum requested SM count (default: 2)\n");
    printf("  --max_sms <N>            Maximum requested SM count (default: all SMs)\n");
    printf("  --step_sms <N>           Requested SM step (default: SMs per TPC)\n");
    printf("  --row_size <N>           TS-GEMM matrix A row count M (default: 8192)\n");
    printf("  --K <N>                  Shared inner dimension K (default: 4096)\n");
    printf("  --N <N>                  Skinny output width N, must be multiple of 16 (default: 16)\n");
    printf("  --bandwidth_bytes <N>    Bytes for bandwidth roof kernel (default: 1073741824)\n");
    printf("  --bandwidth_tpb <N>      Threads per block for bandwidth kernel (default: 256)\n");
    printf("  --compute_repeats <N>    MMA repeats in compute roof kernel (default: 4096)\n");
    printf("  --real_iters <N>         Real TS-GEMM launches per measurement (default: 8)\n");
    printf("  --repeats <N>            Repeats per sweep point (default: 5)\n");
    printf("  --csv <path>             Output CSV path (default: ./results.csv)\n");
}

Config parse_args(int argc, char** argv) {
    Config cfg;
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
        } else if (arg == "--row_size" && i + 1 < argc) {
            cfg.row_size = std::atoi(argv[++i]);
        } else if (arg == "--K" && i + 1 < argc) {
            cfg.K = std::atoi(argv[++i]);
        } else if (arg == "--N" && i + 1 < argc) {
            cfg.N = std::atoi(argv[++i]);
        } else if (arg == "--bandwidth_bytes" && i + 1 < argc) {
            cfg.bandwidth_bytes = std::strtoull(argv[++i], nullptr, 10);
        } else if (arg == "--bandwidth_tpb" && i + 1 < argc) {
            cfg.bandwidth_tpb = std::atoi(argv[++i]);
        } else if (arg == "--compute_repeats" && i + 1 < argc) {
            cfg.compute_repeats = std::atoi(argv[++i]);
        } else if (arg == "--real_iters" && i + 1 < argc) {
            cfg.real_iters = std::atoi(argv[++i]);
        } else if (arg == "--repeats" && i + 1 < argc) {
            cfg.repeats = std::atoi(argv[++i]);
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

std::vector<int> make_tpc_sweep(const Config& cfg, const RuntimeInfo& runtime) {
    const int step_sms = cfg.step_sms > 0 ? cfg.step_sms : runtime.sms_per_tpc;
    const int min_sms = std::max(1, cfg.min_sms);
    const int max_sms = cfg.max_sms > 0 ? std::min(cfg.max_sms, runtime.total_sms) : runtime.total_sms;
    const int min_tpc = std::max(1, ceil_div(min_sms, runtime.sms_per_tpc));
    const int max_tpc = std::max(min_tpc, max_sms / runtime.sms_per_tpc);
    const int step_tpc = std::max(1, ceil_div(step_sms, runtime.sms_per_tpc));

    std::vector<int> tpcs;
    for (int tpc = min_tpc; tpc <= max_tpc; tpc += step_tpc) {
        tpcs.push_back(tpc);
    }
    if (!tpcs.empty() && tpcs.back() != max_tpc) {
        tpcs.push_back(max_tpc);
    }
    return tpcs;
}

uint64_t make_first_n_tpc_mask(int tpc_count) {
    uint64_t mask = 0;
    libsmctrl_make_mask(&mask, 0, tpc_count);
    return mask;
}

void validate_config(const Config& cfg) {
    if (cfg.row_size <= 0 || cfg.K <= 0 || cfg.N <= 0) {
        fprintf(stderr, "row_size, K, and N must be positive.\n");
        std::exit(EXIT_FAILURE);
    }
    if (cfg.row_size % 16 != 0 || cfg.K % 16 != 0 || cfg.N % 16 != 0) {
        fprintf(stderr, "row_size, K, and N must be multiples of 16 for the WMMA kernels.\n");
        std::exit(EXIT_FAILURE);
    }
    if (cfg.bandwidth_tpb != 256) {
        fprintf(stderr, "This experiment currently expects bandwidth_tpb=256 for the checksum reduction.\n");
        std::exit(EXIT_FAILURE);
    }
}

double theoretical_total_bytes(const Config& cfg) {
    return 2.0 * (static_cast<double>(cfg.row_size) * cfg.K +
                  static_cast<double>(cfg.K) * cfg.N +
                  static_cast<double>(cfg.row_size) * cfg.N);
}

double theoretical_flops(const Config& cfg) {
    return 2.0 * static_cast<double>(cfg.row_size) * cfg.K * cfg.N;
}

double arithmetic_intensity(const Config& cfg) {
    return theoretical_flops(cfg) / theoretical_total_bytes(cfg);
}

void allocate_buffers(const Config& cfg, Buffers& buffers, cudaStream_t stream) {
    CHECK_CUDA(cudaMalloc(&buffers.bw_input, cfg.bandwidth_bytes));
    CHECK_CUDA(cudaMemsetAsync(buffers.bw_input, 0xAB, cfg.bandwidth_bytes, stream));
    CHECK_CUDA(cudaMalloc(&buffers.bw_sink, 4096 * sizeof(uint64_t)));
    CHECK_CUDA(cudaMemsetAsync(buffers.bw_sink, 0, 4096 * sizeof(uint64_t), stream));

    const size_t a_elems = static_cast<size_t>(cfg.row_size) * cfg.K;
    const size_t b_elems = static_cast<size_t>(cfg.K) * cfg.N;
    const size_t c_elems = static_cast<size_t>(cfg.row_size) * cfg.N;
    CHECK_CUDA(cudaMalloc(&buffers.A, a_elems * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&buffers.B, b_elems * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&buffers.C, c_elems * sizeof(__half)));
    fill_half_buffer(stream, buffers.A, a_elems, 1.0f);
    fill_half_buffer(stream, buffers.B, b_elems, 1.0f);
    CHECK_CUDA(cudaMemsetAsync(buffers.C, 0, c_elems * sizeof(__half), stream));

    CHECK_CUDA(cudaMalloc(&buffers.compute_sink, static_cast<size_t>(4096) * sizeof(float)));
    CHECK_CUDA(cudaMemsetAsync(buffers.compute_sink, 0, static_cast<size_t>(4096) * sizeof(float), stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
}

void free_buffers(Buffers& buffers) {
    CHECK_CUDA(cudaFree(buffers.bw_input));
    CHECK_CUDA(cudaFree(buffers.bw_sink));
    CHECK_CUDA(cudaFree(buffers.A));
    CHECK_CUDA(cudaFree(buffers.B));
    CHECK_CUDA(cudaFree(buffers.C));
    CHECK_CUDA(cudaFree(buffers.compute_sink));
}

double measure_bandwidth_roof(cudaStream_t stream, uint64_t mask, const Buffers& buffers, const Config& cfg,
                              int active_sms) {
    for (int i = 0; i < 2; ++i) {
        libsmctrl_set_next_mask(mask);
        launch_bandwidth_kernel(stream, buffers.bw_input, buffers.bw_sink, cfg.bandwidth_bytes, cfg.bandwidth_tpb,
                                active_sms);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    double total_gbps = 0.0;
    for (int rep = 0; rep < cfg.repeats; ++rep) {
        libsmctrl_set_next_mask(mask);
        CHECK_CUDA(cudaEventRecord(start, stream));
        launch_bandwidth_kernel(stream, buffers.bw_input, buffers.bw_sink, cfg.bandwidth_bytes, cfg.bandwidth_tpb,
                                active_sms);
        CHECK_CUDA(cudaEventRecord(stop, stream));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float elapsed_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
        total_gbps += (cfg.bandwidth_bytes / 1e9) / (elapsed_ms / 1e3);
    }
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return total_gbps / cfg.repeats;
}

double measure_compute_roof(cudaStream_t stream, uint64_t mask, const Buffers& buffers, const Config& cfg,
                            int active_sms) {
    for (int i = 0; i < 2; ++i) {
        libsmctrl_set_next_mask(mask);
        launch_tsgemm_compute_roof_kernel(stream, buffers.compute_sink, active_sms, cfg.compute_repeats);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    const int warps_per_block = 4;
    const int blocks = active_sms * 4;
    const double flops_per_launch =
        static_cast<double>(blocks) * warps_per_block * cfg.compute_repeats * 2.0 * WMMA_M * WMMA_N * WMMA_K;

    double total_gflops = 0.0;
    for (int rep = 0; rep < cfg.repeats; ++rep) {
        libsmctrl_set_next_mask(mask);
        CHECK_CUDA(cudaEventRecord(start, stream));
        launch_tsgemm_compute_roof_kernel(stream, buffers.compute_sink, active_sms, cfg.compute_repeats);
        CHECK_CUDA(cudaEventRecord(stop, stream));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float elapsed_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
        total_gflops += flops_per_launch / (elapsed_ms * 1e6);
    }
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return total_gflops / cfg.repeats;
}

double measure_real_tsgemm(cudaStream_t stream, uint64_t mask, const Buffers& buffers, const Config& cfg,
                           double* time_ms_out) {
    for (int i = 0; i < 2; ++i) {
        for (int iter = 0; iter < cfg.real_iters; ++iter) {
            libsmctrl_set_next_mask(mask);
            launch_tsgemm_real_kernel(stream, buffers.A, buffers.B, buffers.C, cfg.row_size, cfg.N, cfg.K);
        }
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    float total_time_ms = 0.0f;
    for (int iter = 0; iter < cfg.real_iters; ++iter) {
        libsmctrl_set_next_mask(mask);
        CHECK_CUDA(cudaEventRecord(start, stream));
        launch_tsgemm_real_kernel(stream, buffers.A, buffers.B, buffers.C, cfg.row_size, cfg.N, cfg.K);
        CHECK_CUDA(cudaEventRecord(stop, stream));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float iter_time_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&iter_time_ms, start, stop));
        total_time_ms += iter_time_ms;
    }
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    const double avg_time_ms = total_time_ms / cfg.real_iters;
    *time_ms_out = avg_time_ms;
    return theoretical_flops(cfg) / (avg_time_ms * 1e6);
}

void write_csv(const std::string& path, const std::vector<RowResult>& results) {
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) {
        fprintf(stderr, "Failed to open %s for writing.\n", path.c_str());
        std::exit(EXIT_FAILURE);
    }

    fprintf(f,
            "row_size,K,N,num_sms,tpc_count,sms_per_tpc,arithmetic_intensity_flops_per_byte,"
            "theoretical_total_bytes,bandwidth_roof_GBps,compute_roof_GFLOPS,memory_roof_GFLOPS,"
            "predicted_GFLOPS,required_full_bw_GFLOPS,real_GFLOPS,real_time_ms,implied_real_GBps\n");

    for (const auto& r : results) {
        fprintf(f,
                "%d,%d,%d,%d,%d,%d,%.6f,%.0f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                r.row_size, r.K, r.N, r.num_sms, r.tpc_count, r.sms_per_tpc, r.arithmetic_intensity,
                r.total_bytes, r.bandwidth_roof_gbps, r.compute_roof_gflops, r.memory_roof_gflops,
                r.predicted_gflops, r.required_full_bw_gflops, r.real_gflops, r.real_time_ms,
                r.implied_real_bw_gbps);
    }

    std::fclose(f);
}

int main(int argc, char** argv) {
    Config cfg = parse_args(argc, argv);
    validate_config(cfg);

    CHECK_CU(cuInit(0));
    CHECK_CUDA(cudaSetDevice(0));

    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

    RuntimeInfo runtime;
    runtime.total_sms = prop.multiProcessorCount;
    int status = libsmctrl_get_tpc_info_cuda(&runtime.total_tpcs, 0);
    if (status != 0) {
        fprintf(stderr, "libsmctrl_get_tpc_info_cuda failed with code %d\n", status);
        return EXIT_FAILURE;
    }
    runtime.sms_per_tpc = std::max(1, runtime.total_sms / static_cast<int>(runtime.total_tpcs));

    if (cfg.max_sms <= 0) {
        cfg.max_sms = runtime.total_sms;
    }

    std::vector<int> tpc_sweep = make_tpc_sweep(cfg, runtime);
    if (tpc_sweep.empty()) {
        fprintf(stderr, "No valid sweep points.\n");
        return EXIT_FAILURE;
    }

    printf("=============================================================\n");
    printf("TS-GEMM Roofline Experiment (Orin/libsmctrl)\n");
    printf("=============================================================\n");
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Total SMs: %d\n", runtime.total_sms);
    printf("Total TPCs: %u\n", runtime.total_tpcs);
    printf("SMs per TPC: %d\n", runtime.sms_per_tpc);
    printf("TS-GEMM shape: A[%d x %d] * B[%d x %d] -> C[%d x %d]\n", cfg.row_size, cfg.K, cfg.K, cfg.N,
           cfg.row_size, cfg.N);
    printf("Arithmetic intensity (theoretical): %.3f FLOP/byte\n", arithmetic_intensity(cfg));
    printf("Bandwidth bytes: %.2f GiB\n", cfg.bandwidth_bytes / (1024.0 * 1024.0 * 1024.0));
    printf("Compute repeats: %d\n", cfg.compute_repeats);
    printf("Real kernel iterations: %d\n", cfg.real_iters);
    printf("Measurement repeats: %d\n", cfg.repeats);
    printf("CSV output: %s\n", cfg.csv_path.c_str());
    printf("=============================================================\n\n");

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    Buffers buffers;
    allocate_buffers(cfg, buffers, stream);

    std::vector<RowResult> results;
    results.reserve(tpc_sweep.size());

    for (int tpc_count : tpc_sweep) {
        const int active_sms = tpc_count * runtime.sms_per_tpc;
        const uint64_t mask = make_first_n_tpc_mask(tpc_count);
        RowResult row;
        row.num_sms = active_sms;
        row.tpc_count = tpc_count;
        row.sms_per_tpc = runtime.sms_per_tpc;
        row.row_size = cfg.row_size;
        row.K = cfg.K;
        row.N = cfg.N;
        row.arithmetic_intensity = arithmetic_intensity(cfg);
        row.total_bytes = theoretical_total_bytes(cfg);

        row.bandwidth_roof_gbps = measure_bandwidth_roof(stream, mask, buffers, cfg, active_sms);
        row.compute_roof_gflops = measure_compute_roof(stream, mask, buffers, cfg, active_sms);
        row.memory_roof_gflops = row.arithmetic_intensity * row.bandwidth_roof_gbps;
        row.real_gflops = measure_real_tsgemm(stream, mask, buffers, cfg, &row.real_time_ms);
        row.predicted_gflops = std::min(row.memory_roof_gflops, row.compute_roof_gflops);
        row.implied_real_bw_gbps = row.real_gflops / row.arithmetic_intensity;

        results.push_back(row);

        printf("SMs=%2d  BW roof=%7.2f GB/s  Compute roof=%8.2f GFLOPS  Memory roof=%8.2f GFLOPS  "
               "Real=%8.2f GFLOPS (%.3f ms)\n",
               row.num_sms, row.bandwidth_roof_gbps, row.compute_roof_gflops, row.memory_roof_gflops,
               row.real_gflops, row.real_time_ms);
    }

    double full_bw_gbps = 0.0;
    for (const auto& row : results) {
        full_bw_gbps = std::max(full_bw_gbps, row.bandwidth_roof_gbps);
    }
    const double required_full_bw_gflops = arithmetic_intensity(cfg) * full_bw_gbps;
    for (auto& row : results) {
        row.required_full_bw_gflops = required_full_bw_gflops;
    }

    write_csv(cfg.csv_path, results);

    int min_compute_sms = -1;
    for (const auto& row : results) {
        if (row.compute_roof_gflops >= required_full_bw_gflops) {
            min_compute_sms = row.num_sms;
            break;
        }
    }

    printf("\nPeak bandwidth roof: %.2f GB/s\n", full_bw_gbps);
    printf("Required compute for full-bandwidth TS-GEMM stream: %.2f GFLOPS\n", required_full_bw_gflops);
    if (min_compute_sms > 0) {
        printf("Minimum SMs whose compute roof can digest full-bandwidth data: %d\n", min_compute_sms);
    } else {
        printf("No measured SM point reached the compute needed for the full-bandwidth stream.\n");
    }
    printf("Results written to: %s\n", cfg.csv_path.c_str());

    free_buffers(buffers);
    CHECK_CUDA(cudaStreamDestroy(stream));
    return 0;
}

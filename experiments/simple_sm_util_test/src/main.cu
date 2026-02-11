#include "green_context_utils.h"
#include "kernels.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#define CHECK_CUDA(call)                                                    \
  do {                                                                      \
    cudaError_t _err = (call);                                              \
    if (_err != cudaSuccess) {                                              \
      std::cerr << "CUDA error: " << cudaGetErrorString(_err)              \
                << " (" << __FILE__ << ":" << __LINE__ << ")\n";          \
      std::exit(EXIT_FAILURE);                                              \
    }                                                                       \
  } while (0)

struct Options {
  int min_sms = 1;
  int max_sms = -1;
  int step = 1;
  size_t bytes = 1ULL << 30;
  int tpb = 256;
  int iters = -1;
  int repeats = 5;
  std::string csv_path = "./results.csv";
};

static void print_usage(const char* prog) {
  std::cout << "Usage: " << prog << " [options]\n"
            << "  --min_sms N      (default 1)\n"
            << "  --max_sms N      (default SM_total)\n"
            << "  --step N         (default 1)\n"
            << "  --bytes N        (total bytes read per measurement, default 1GiB)\n"
            << "  --tpb N          (threads per block, default 256)\n"
            << "  --iters N        (override loop iterations)\n"
            << "  --repeats N      (default 5)\n"
            << "  --csv PATH       (default ./results.csv)\n";
}

static bool parse_args(int argc, char** argv, Options* opts) {
  if (!opts) {
    return false;
  }
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto require_value = [&](int idx) -> const char* {
      if (idx + 1 >= argc) {
        std::cerr << "Missing value for " << arg << "\n";
        return nullptr;
      }
      return argv[idx + 1];
    };

    if (arg == "--min_sms") {
      const char* v = require_value(i);
      if (!v) return false;
      opts->min_sms = std::atoi(v);
      ++i;
    } else if (arg == "--max_sms") {
      const char* v = require_value(i);
      if (!v) return false;
      opts->max_sms = std::atoi(v);
      ++i;
    } else if (arg == "--step") {
      const char* v = require_value(i);
      if (!v) return false;
      opts->step = std::atoi(v);
      ++i;
    } else if (arg == "--bytes") {
      const char* v = require_value(i);
      if (!v) return false;
      opts->bytes = static_cast<size_t>(std::strtoull(v, nullptr, 10));
      ++i;
    } else if (arg == "--tpb") {
      const char* v = require_value(i);
      if (!v) return false;
      opts->tpb = std::atoi(v);
      ++i;
    } else if (arg == "--iters") {
      const char* v = require_value(i);
      if (!v) return false;
      opts->iters = std::atoi(v);
      ++i;
    } else if (arg == "--repeats") {
      const char* v = require_value(i);
      if (!v) return false;
      opts->repeats = std::atoi(v);
      ++i;
    } else if (arg == "--csv") {
      const char* v = require_value(i);
      if (!v) return false;
      opts->csv_path = v;
      ++i;
    } else if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      std::exit(EXIT_SUCCESS);
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      return false;
    }
  }
  return true;
}

static double mean(const std::vector<double>& v) {
  if (v.empty()) return 0.0;
  double sum = 0.0;
  for (double x : v) sum += x;
  return sum / static_cast<double>(v.size());
}

static double stdev(const std::vector<double>& v, double m) {
  if (v.size() < 2) return 0.0;
  double acc = 0.0;
  for (double x : v) {
    double d = x - m;
    acc += d * d;
  }
  return std::sqrt(acc / static_cast<double>(v.size() - 1));
}

int main(int argc, char** argv) {
  Options opts;
  if (!parse_args(argc, argv, &opts)) {
    print_usage(argv[0]);
    return EXIT_FAILURE;
  }

  std::string err;
  int sm_total = 0;
  if (!get_sm_total(&sm_total, &err)) {
    std::cerr << "Failed to query SM count: " << err << "\n";
    return EXIT_FAILURE;
  }

  if (opts.max_sms <= 0) {
    opts.max_sms = sm_total;
  }
  opts.min_sms = std::max(1, opts.min_sms);
  opts.max_sms = std::min(sm_total, opts.max_sms);
  if (opts.min_sms > opts.max_sms || opts.step <= 0) {
    std::cerr << "Invalid SM sweep range\n";
    return EXIT_FAILURE;
  }

  bool prefer_green = green_contexts_supported();
  if (!prefer_green) {
    std::cerr << "Warning: CUDA Green Contexts not available in headers; "
              << "falling back to primary context.\n";
  }

  std::ofstream csv(opts.csv_path);
  if (!csv) {
    std::cerr << "Failed to open CSV output: " << opts.csv_path << "\n";
    return EXIT_FAILURE;
  }

  csv << "sm_count,green_used,time_ms_mean,time_ms_std,total_bytes_read,"
      << "read_GBps_mean,read_GBps_std,checksum,tpb,vec_bytes,unroll,blocks,iters\n";

  for (int sm = opts.min_sms; sm <= opts.max_sms; sm += opts.step) {
    GreenContext ctx;
    bool used_green = false;
    std::string ctx_err;
    if (!create_context_with_sm_count(sm, prefer_green, &ctx, &ctx_err, &used_green)) {
      std::cerr << "Context creation failed for sm=" << sm << ": " << ctx_err << "\n";
      return EXIT_FAILURE;
    }
    if (!ctx_err.empty()) {
      std::cerr << "Warning: " << ctx_err << "\n";
      if (!used_green) {
        prefer_green = false;
      }
    }

    if (!make_context_current(&ctx, &err)) {
      std::cerr << "Failed to set context: " << err << "\n";
      destroy_context(&ctx);
      return EXIT_FAILURE;
    }

    CHECK_CUDA(cudaFree(0));

    const size_t vec_bytes = kVecBytes;
    size_t src_bytes = (opts.bytes / vec_bytes) * vec_bytes;
    if (src_bytes == 0) {
      src_bytes = vec_bytes;
    }

    int iters = opts.iters;
    if (iters <= 0) {
      iters = 1;
    }
    const size_t num_vec = src_bytes / vec_bytes;
    const size_t total_bytes_read = src_bytes * static_cast<size_t>(iters);

    const int blocks_per_sm = 8;
    int blocks = sm * blocks_per_sm;
    const size_t max_blocks = (num_vec + opts.tpb * kUnroll - 1) / (opts.tpb * kUnroll);
    if (blocks > static_cast<int>(max_blocks)) {
      blocks = static_cast<int>(max_blocks);
    }
    blocks = std::max(blocks, 1);

    VecType* d_src = nullptr;
    uint64_t* d_sink = nullptr;
    const size_t sink_elems = static_cast<size_t>(blocks) * opts.tpb;

    CHECK_CUDA(cudaMalloc(&d_src, src_bytes));
    CHECK_CUDA(cudaMalloc(&d_sink, sink_elems * sizeof(uint64_t)));
    CHECK_CUDA(cudaMemset(d_src, 1, src_bytes));
    CHECK_CUDA(cudaMemset(d_sink, 0, sink_elems * sizeof(uint64_t)));

    read_bw_kernel<<<blocks, opts.tpb>>>(d_src, num_vec, d_sink, 1);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<double> times_ms;
    std::vector<double> bw_gbps;
    times_ms.reserve(opts.repeats);
    bw_gbps.reserve(opts.repeats);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    for (int r = 0; r < opts.repeats; ++r) {
      CHECK_CUDA(cudaEventRecord(start));
      read_bw_kernel<<<blocks, opts.tpb>>>(d_src, num_vec, d_sink, iters);
      CHECK_CUDA(cudaEventRecord(stop));
      CHECK_CUDA(cudaEventSynchronize(stop));
      float ms = 0.0f;
      CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
      const double t_ms = static_cast<double>(ms);
      times_ms.push_back(t_ms);
      const double gbps = (static_cast<double>(total_bytes_read) / 1.0e9) / (t_ms / 1.0e3);
      bw_gbps.push_back(gbps);
    }

    uint64_t checksum = 0;
    CHECK_CUDA(cudaMemcpy(&checksum, d_sink, sizeof(uint64_t), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_sink));

    destroy_context(&ctx);

    const double tmean = mean(times_ms);
    const double tstd = stdev(times_ms, tmean);
    const double gbps_mean = mean(bw_gbps);
    const double gbps_std = stdev(bw_gbps, gbps_mean);

    csv << sm << "," << (used_green ? 1 : 0) << ","
        << std::fixed << std::setprecision(3) << tmean << ","
        << std::fixed << std::setprecision(3) << tstd << ","
        << total_bytes_read << ","
        << std::fixed << std::setprecision(3) << gbps_mean << ","
        << std::fixed << std::setprecision(3) << gbps_std << ","
        << checksum << ","
        << opts.tpb << "," << kVecBytes << "," << kUnroll << ","
        << blocks << "," << iters << "\n";

    std::cout << "SM " << sm << ": " << std::fixed << std::setprecision(2)
              << gbps_mean << " GB/s (" << (used_green ? "green" : "normal")
              << ")\n";
  }

  std::cout << "Results written to " << opts.csv_path << "\n";
  return EXIT_SUCCESS;
}

#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <algorithm>

// Include libsmctrl from the csrc directory
#include "libsmctrl.h"

// Include our kernel definitions
#include "kernels.cuh"

// Error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CUDA_CHECK_DRIVER(call) do { \
    CUresult err = call; \
    if (err != CUDA_SUCCESS) { \
        const char* errStr; \
        cuGetErrorString(err, &errStr); \
        fprintf(stderr, "CUDA Driver error at %s:%d: %s\n", __FILE__, __LINE__, errStr); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Configuration structure
struct Config {
    int min_sms;
    int max_sms;
    int step;
    size_t bytes;
    int tpb;
    int iters;
    int repeats;
    const char* csv_path;
    bool iters_set;
    
    Config() : min_sms(1), max_sms(-1), step(1), 
               bytes(1073741824ULL), tpb(256), iters(-1), 
               repeats(5), csv_path("./results.csv"), iters_set(false) {}
};

// Parse command line arguments
void parse_args(int argc, char** argv, Config& cfg) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--min_sms") == 0 && i + 1 < argc) {
            cfg.min_sms = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--max_sms") == 0 && i + 1 < argc) {
            cfg.max_sms = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--step") == 0 && i + 1 < argc) {
            cfg.step = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--bytes") == 0 && i + 1 < argc) {
            cfg.bytes = strtoull(argv[++i], nullptr, 10);
        } else if (strcmp(argv[i], "--tpb") == 0 && i + 1 < argc) {
            cfg.tpb = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            cfg.iters = atoi(argv[++i]);
            cfg.iters_set = true;
        } else if (strcmp(argv[i], "--repeats") == 0 && i + 1 < argc) {
            cfg.repeats = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc) {
            cfg.csv_path = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("Options:\n");
            printf("  --min_sms N      Minimum SM/TPC count (default: 1)\n");
            printf("  --max_sms N      Maximum SM/TPC count (default: all available)\n");
            printf("  --step N         Step size for SM/TPC sweep (default: 1)\n");
            printf("  --bytes N        Total bytes to read per measurement (default: 1073741824)\n");
            printf("  --tpb N          Threads per block (default: 256)\n");
            printf("  --iters N        Iterations per thread (default: auto-calculated)\n");
            printf("  --repeats N      Number of measurements per config (default: 5)\n");
            printf("  --csv PATH       Output CSV file path (default: ./results.csv)\n");
            printf("  --help, -h       Show this help message\n");
            exit(0);
        }
    }
}

// Compute statistics
void compute_stats(const std::vector<float>& values, float& mean, float& stddev) {
    mean = 0.0f;
    for (float v : values) mean += v;
    mean /= values.size();
    
    stddev = 0.0f;
    for (float v : values) {
        float diff = v - mean;
        stddev += diff * diff;
    }
    stddev = sqrtf(stddev / values.size());
}

int main(int argc, char** argv) {
    Config cfg;
    parse_args(argc, argv, cfg);
    
    // Initialize CUDA
    CUDA_CHECK_DRIVER(cuInit(0));
    
    // Get device properties
    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("Device: %s\n", prop.name);
    printf("Total SMs: %d\n", prop.multiProcessorCount);
    
    // Get TPC information using libsmctrl
    uint32_t num_tpcs;
    int result = libsmctrl_get_tpc_info_cuda(&num_tpcs, device);
    if (result != 0) {
        fprintf(stderr, "Error: libsmctrl_get_tpc_info_cuda failed with code %d\n", result);
        fprintf(stderr, "This GPU may not support SM masking (requires compute capability >= 3.5)\n");
        return EXIT_FAILURE;
    }
    
    printf("Total TPCs: %u\n", num_tpcs);
    printf("SMs per TPC: %d\n", prop.multiProcessorCount / num_tpcs);
    
    // Set default max_sms if not specified
    if (cfg.max_sms < 0) {
        cfg.max_sms = num_tpcs;
    }
    
    // Validate configuration
    if (cfg.min_sms < 1 || cfg.min_sms > (int)num_tpcs) {
        fprintf(stderr, "Error: min_sms must be between 1 and %u\n", num_tpcs);
        return EXIT_FAILURE;
    }
    if (cfg.max_sms < cfg.min_sms || cfg.max_sms > (int)num_tpcs) {
        fprintf(stderr, "Error: max_sms must be between min_sms and %u\n", num_tpcs);
        return EXIT_FAILURE;
    }
    
    printf("\nConfiguration:\n");
    printf("  SM/TPC range: %d to %d (step %d)\n", cfg.min_sms, cfg.max_sms, cfg.step);
    printf("  Total bytes per measurement: %zu (%.2f GiB)\n", 
           cfg.bytes, cfg.bytes / (1024.0 * 1024.0 * 1024.0));
    printf("  Threads per block: %d\n", cfg.tpb);
    printf("  Repeats per config: %d\n", cfg.repeats);
    printf("  Output CSV: %s\n", cfg.csv_path);
    
    // Create CUDA stream for masked execution
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    // Allocate device memory
    // We need enough for vectorized float4 access
    const size_t num_float4 = (cfg.bytes + 15) / 16;  // Round up
    const size_t actual_bytes = num_float4 * 16;
    
    float4* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, actual_bytes));
    
    // Initialize data (run once without timing)
    const int init_blocks = (num_float4 + cfg.tpb - 1) / cfg.tpb;
    init_data<<<init_blocks, cfg.tpb>>>(d_data, num_float4);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Sink buffer for checksums (one uint64_t per block)
    // We'll use a conservative estimate for max blocks
    const int max_blocks = 1024;  // Will be adjusted per launch
    uint64_t* d_sink;
    CUDA_CHECK(cudaMalloc(&d_sink, max_blocks * sizeof(uint64_t)));
    
    // CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Open CSV file
    FILE* csv = fopen(cfg.csv_path, "w");
    if (!csv) {
        fprintf(stderr, "Error: Cannot open %s for writing\n", cfg.csv_path);
        return EXIT_FAILURE;
    }
    
    // Write CSV header
    fprintf(csv, "tpc_count,time_ms_mean,time_ms_std,total_bytes_read,"
                 "read_GBps_mean,read_GBps_std,checksum,tpb,vec_bytes,unroll,blocks,iters\n");
    
    printf("\nStarting sweep...\n");
    printf("%-10s %-15s %-15s %-15s\n", "TPC_Count", "Time(ms)", "Read_BW(GB/s)", "Checksum");
    printf("-------------------------------------------------------------\n");
    
    // Sweep over TPC counts
    for (int tpc_count = cfg.min_sms; tpc_count <= cfg.max_sms; tpc_count += cfg.step) {
        // Create mask for first tpc_count TPCs
        uint64_t mask;
        libsmctrl_make_mask(&mask, 0, tpc_count);
        
        // Note: Using next mask instead of stream mask for broader CUDA version compatibility
        // next mask must be set before each kernel launch
        
        // Calculate launch configuration
        // We want enough blocks to keep the enabled TPCs busy
        const int sms_per_tpc = prop.multiProcessorCount / num_tpcs;
        const int active_sms = tpc_count * sms_per_tpc;
        
        // Heuristic: 4 blocks per active SM for good occupancy
        const int blocks = active_sms * 4;
        const int total_threads = blocks * cfg.tpb;
        
        // Calculate iterations per thread
        int iters;
        if (cfg.iters_set) {
            iters = cfg.iters;
        } else {
            // Each iteration reads UNROLL * sizeof(float4) = 4 * 16 = 64 bytes per thread
            const int bytes_per_thread_per_iter = 4 * 16;  // UNROLL=4, sizeof(float4)=16
            const size_t total_bytes_per_iter = (size_t)total_threads * bytes_per_thread_per_iter;
            iters = (cfg.bytes + total_bytes_per_iter - 1) / total_bytes_per_iter;
            if (iters < 1) iters = 1;
        }
        
        // Warm-up launch (not timed)
        libsmctrl_set_next_mask(mask);
        read_bandwidth_kernel<4><<<blocks, cfg.tpb, 0, stream>>>(d_data, d_sink, actual_bytes, iters);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        // Timed measurements
        std::vector<float> times_ms;
        uint64_t checksum = 0;
        
        for (int rep = 0; rep < cfg.repeats; rep++) {
            libsmctrl_set_next_mask(mask);  // Must set before each launch when using next mask
            CUDA_CHECK(cudaEventRecord(start, stream));
            read_bandwidth_kernel<4><<<blocks, cfg.tpb, 0, stream>>>(d_data, d_sink, actual_bytes, iters);
            CUDA_CHECK(cudaEventRecord(stop, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            
            float time_ms;
            CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, stop));
            times_ms.push_back(time_ms);
            
            // Read checksum from first block (for verification)
            if (rep == 0) {
                CUDA_CHECK(cudaMemcpy(&checksum, d_sink, sizeof(uint64_t), cudaMemcpyDeviceToHost));
            }
        }
        
        // Compute statistics
        float time_mean, time_std;
        compute_stats(times_ms, time_mean, time_std);
        
        // Compute bandwidth (GB/s)
        // Note: Using actual bytes read, not allocated buffer size
        const size_t bytes_per_thread = (size_t)iters * 4 * 16;  // iters * UNROLL * sizeof(float4)
        const size_t total_bytes_read = (size_t)total_threads * bytes_per_thread;
        const float bw_mean = (total_bytes_read / (time_mean * 1e-3)) / 1e9;
        const float bw_std = (total_bytes_read / 1e9) * time_std / (time_mean * time_mean * 1e-3);
        
        // Print to console
        printf("%-10d %-15.3f %-15.2f 0x%016lx\n", 
               tpc_count, time_mean, bw_mean, checksum);
        
        // Write to CSV
        fprintf(csv, "%d,%.6f,%.6f,%zu,%.6f,%.6f,%lu,%d,%d,%d,%d,%d\n",
                tpc_count, time_mean, time_std, total_bytes_read,
                bw_mean, bw_std, checksum,
                cfg.tpb, 16, 4, blocks, iters);
        fflush(csv);
    }
    
    printf("\nSweep complete! Results written to %s\n", cfg.csv_path);
    
    // Cleanup
    fclose(csv);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_sink));
    CUDA_CHECK(cudaStreamDestroy(stream));
    
    return 0;
}

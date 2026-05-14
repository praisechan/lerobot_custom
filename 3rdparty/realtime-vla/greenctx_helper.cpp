#include <cuda.h>
#include <cuda_runtime_api.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

std::string cu_error_string(CUresult result) {
    const char* name = nullptr;
    const char* text = nullptr;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &text);
    std::string out = name ? name : "CUDA_ERROR_UNKNOWN";
    if (text) {
        out += ": ";
        out += text;
    }
    return out;
}

void check_cu(CUresult result, const char* expr, const char* file, int line) {
    if (result != CUDA_SUCCESS) {
        throw std::runtime_error(
            std::string(file) + ":" + std::to_string(line) + " " + expr + " failed: " +
            cu_error_string(result));
    }
}

void check_cuda(cudaError_t result, const char* expr, const char* file, int line) {
    if (result != cudaSuccess) {
        throw std::runtime_error(
            std::string(file) + ":" + std::to_string(line) + " " + expr + " failed: " +
            cudaGetErrorString(result));
    }
}

#define CHECK_CU(expr) check_cu((expr), #expr, __FILE__, __LINE__)
#define CHECK_CUDA(expr) check_cuda((expr), #expr, __FILE__, __LINE__)

int gcd_int(int a, int b) {
    a = std::abs(a);
    b = std::abs(b);
    while (b != 0) {
        int t = a % b;
        a = b;
        b = t;
    }
    return a;
}

std::uintptr_t ptr_to_int(CUstream stream) {
    return reinterpret_cast<std::uintptr_t>(stream);
}

std::uintptr_t ptr_to_int(CUcontext ctx) {
    return reinterpret_cast<std::uintptr_t>(ctx);
}

struct OneGreenContext {
    std::vector<CUdevResource> resources;
    CUdevResourceDesc desc = nullptr;
    CUgreenCtx green = nullptr;
    CUcontext ctx = nullptr;
    CUstream stream = nullptr;
    unsigned long long green_id = 0;
    unsigned long long ctx_id = 0;
    unsigned long long stream_id = 0;
    unsigned int requested_sms = 0;
    unsigned int actual_sms = 0;
    unsigned int groups = 0;

    void fill_ids_and_verify(const char* label) {
        CHECK_CU(cuCtxFromGreenCtx(&ctx, green));
        CHECK_CU(cuGreenCtxGetId(green, &green_id));
        CHECK_CU(cuCtxGetId(ctx, &ctx_id));
        CHECK_CU(cuGreenCtxStreamCreate(&stream, green, CU_STREAM_NON_BLOCKING, 0));
        CHECK_CU(cuStreamGetId(stream, &stream_id));

        CUdevResource actual_resource;
        CHECK_CU(cuGreenCtxGetDevResource(green, &actual_resource, CU_DEV_RESOURCE_TYPE_SM));
        actual_sms = actual_resource.sm.smCount;

        CUcontext stream_ctx = nullptr;
        CUgreenCtx stream_green = nullptr;
        CHECK_CU(cuStreamGetCtx_v2(stream, &stream_ctx, &stream_green));
        unsigned long long stream_green_id = 0;
        CHECK_CU(cuGreenCtxGetId(stream_green, &stream_green_id));
        if (stream_green_id != green_id) {
            throw std::runtime_error(std::string(label) + " stream is not bound to its GreenContext");
        }
        if (stream_ctx != ctx) {
            std::fprintf(stderr,
                         "[GreenCtx] warning: %s stream CUcontext pointer differs from cuCtxFromGreenCtx result; GreenContext ID matches (%llu).\n",
                         label, green_id);
        }
    }

    void synchronize() const {
        if (stream) {
            CHECK_CU(cuCtxSetCurrent(ctx));
            CHECK_CU(cuStreamSynchronize(stream));
        }
    }

    void destroy() {
        if (stream) {
            CUresult current_result = cuCtxSetCurrent(ctx);
            if (current_result != CUDA_SUCCESS) {
                std::fprintf(stderr, "[GreenCtx] cuCtxSetCurrent during destroy failed: %s\n",
                             cu_error_string(current_result).c_str());
            }
            CUresult sync_result = cuStreamSynchronize(stream);
            if (sync_result != CUDA_SUCCESS) {
                std::fprintf(stderr, "[GreenCtx] cuStreamSynchronize during destroy failed: %s\n",
                             cu_error_string(sync_result).c_str());
            }
            CUresult destroy_result = cuStreamDestroy(stream);
            if (destroy_result != CUDA_SUCCESS) {
                std::fprintf(stderr, "[GreenCtx] cuStreamDestroy failed: %s\n",
                             cu_error_string(destroy_result).c_str());
            }
            stream = nullptr;
        }
        if (green) {
            CUresult destroy_result = cuGreenCtxDestroy(green);
            if (destroy_result != CUDA_SUCCESS) {
                std::fprintf(stderr, "[GreenCtx] cuGreenCtxDestroy failed: %s\n",
                             cu_error_string(destroy_result).c_str());
            }
            green = nullptr;
        }
        ctx = nullptr;
    }
};

}  // namespace

class GreenContextPair {
public:
    GreenContextPair(int device_index, int encoder_sms, int decoder_sms, bool verbose)
        : device_index_(device_index), encoder_sms_(encoder_sms), decoder_sms_(decoder_sms), verbose_(verbose) {
        CHECK_CU(cuInit(0));
        CHECK_CUDA(cudaSetDevice(device_index_));
        CHECK_CUDA(cudaFree(nullptr));
        CHECK_CU(cuCtxGetCurrent(&primary_ctx_));
        if (!primary_ctx_) {
            throw std::runtime_error("CUDA primary context is not current after cudaFree(nullptr)");
        }
        CHECK_CU(cuCtxGetId(primary_ctx_, &primary_ctx_id_));
        CHECK_CU(cuDeviceGet(&device_, device_index_));
        CHECK_CU(cuDeviceGetDevResource(device_, &device_resource_, CU_DEV_RESOURCE_TYPE_SM));

        validate_and_create();
        log_info();
        CHECK_CU(cuCtxSetCurrent(primary_ctx_));
    }

    ~GreenContextPair() {
        try {
            encoder_.destroy();
            decoder_.destroy();
            if (primary_ctx_) {
                cuCtxSetCurrent(primary_ctx_);
            }
        } catch (const std::exception& ex) {
            std::fprintf(stderr, "[GreenCtx] destructor ignored exception: %s\n", ex.what());
        }
    }

    std::uintptr_t encoder_stream_ptr() const {
        return ptr_to_int(encoder_.stream);
    }

    std::uintptr_t decoder_stream_ptr() const {
        return ptr_to_int(decoder_.stream);
    }

    void set_encoder_current() const {
        CHECK_CU(cuCtxSetCurrent(encoder_.ctx));
    }

    void set_decoder_current() const {
        CHECK_CU(cuCtxSetCurrent(decoder_.ctx));
    }

    void restore_primary_current() const {
        CHECK_CU(cuCtxSetCurrent(primary_ctx_));
    }

    void synchronize_encoder() const {
        encoder_.synchronize();
        CHECK_CU(cuCtxSetCurrent(primary_ctx_));
    }

    void synchronize_decoder() const {
        decoder_.synchronize();
        CHECK_CU(cuCtxSetCurrent(primary_ctx_));
    }

    void synchronize_both() const {
        encoder_.synchronize();
        decoder_.synchronize();
        CHECK_CU(cuCtxSetCurrent(primary_ctx_));
    }

    py::dict info() const {
        py::dict d;
        d["device_index"] = device_index_;
        d["total_sms"] = static_cast<int>(device_resource_.sm.smCount);
        d["min_sm_partition_size"] = static_cast<int>(device_resource_.sm.minSmPartitionSize);
        d["sm_coscheduled_alignment"] = static_cast<int>(device_resource_.sm.smCoscheduledAlignment);
        d["base_sms"] = base_sms_;
        d["split_groups_created"] = groups_created_;
        d["split_use_flags"] = split_use_flags_;
        d["split_ignores_sm_coscheduling"] =
            static_cast<bool>(split_use_flags_ & CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING);
        d["primary_ctx_ptr"] = ptr_to_int(primary_ctx_);
        d["primary_ctx_id"] = primary_ctx_id_;
        add_one(d, "encoder", encoder_);
        add_one(d, "decoder", decoder_);
        return d;
    }

private:
    static void add_one(py::dict& d, const char* prefix, const OneGreenContext& ctx) {
        std::string p(prefix);
        d[py::str(p + "_requested_sms")] = static_cast<int>(ctx.requested_sms);
        d[py::str(p + "_actual_sms")] = static_cast<int>(ctx.actual_sms);
        d[py::str(p + "_groups")] = static_cast<int>(ctx.groups);
        d[py::str(p + "_green_ctx_id")] = ctx.green_id;
        d[py::str(p + "_ctx_id")] = ctx.ctx_id;
        d[py::str(p + "_ctx_ptr")] = ptr_to_int(ctx.ctx);
        d[py::str(p + "_stream_id")] = ctx.stream_id;
        d[py::str(p + "_stream_ptr")] = ptr_to_int(ctx.stream);
    }

    void validate_and_create() {
        const int total_sms = static_cast<int>(device_resource_.sm.smCount);
        const int min_partition = static_cast<int>(device_resource_.sm.minSmPartitionSize);
        const int alignment = static_cast<int>(device_resource_.sm.smCoscheduledAlignment);

        if (encoder_sms_ <= 0 || decoder_sms_ <= 0) {
            throw std::runtime_error("--encoder-sms and --decoder-sms must be positive");
        }
        if (encoder_sms_ + decoder_sms_ > total_sms) {
            throw std::runtime_error(
                "Requested " + std::to_string(encoder_sms_) + " + " + std::to_string(decoder_sms_) +
                " SMs, but device exposes " + std::to_string(total_sms));
        }
        auto validate_sms = [&](const char* label, int sms) {
            if (sms < min_partition || sms % alignment != 0) {
                throw std::runtime_error(
                    std::string(label) + "=" + std::to_string(sms) +
                    " must be >= minSmPartitionSize " + std::to_string(min_partition) +
                    " and aligned to smCoscheduledAlignment " + std::to_string(alignment));
            }
        };
        validate_sms("encoder_sms", encoder_sms_);
        validate_sms("decoder_sms", decoder_sms_);

        base_sms_ = gcd_int(encoder_sms_, decoder_sms_);
        if (base_sms_ < min_partition || base_sms_ % alignment != 0) {
            throw std::runtime_error(
                "Cannot form both requested partitions from one symmetric split: gcd(" +
                std::to_string(encoder_sms_) + ", " + std::to_string(decoder_sms_) + ")=" +
                std::to_string(base_sms_) + ", minSmPartitionSize=" + std::to_string(min_partition) +
                ", smCoscheduledAlignment=" + std::to_string(alignment));
        }

        const unsigned int encoder_groups = static_cast<unsigned int>(encoder_sms_ / base_sms_);
        const unsigned int decoder_groups = static_cast<unsigned int>(decoder_sms_ / base_sms_);
        groups_created_ = encoder_groups + decoder_groups;
        std::vector<CUdevResource> partitions(groups_created_);
        CUdevResource remaining;
        unsigned int requested_groups = groups_created_;
        split_use_flags_ = 0;
        CHECK_CU(cuDevSmResourceSplitByCount(
            partitions.data(), &requested_groups, &device_resource_, &remaining, 0,
            static_cast<unsigned int>(base_sms_)));

        if (requested_groups < groups_created_) {
            const unsigned int default_groups = requested_groups;
            requested_groups = groups_created_;
            split_use_flags_ = CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING;
            CHECK_CU(cuDevSmResourceSplitByCount(
                partitions.data(), &requested_groups, &device_resource_, &remaining, split_use_flags_,
                static_cast<unsigned int>(base_sms_)));
            if (verbose_ && requested_groups >= groups_created_) {
                std::fprintf(
                    stderr,
                    "[GreenCtx] default split created only %u groups of %d SMs; using "
                    "CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING to create %u groups.\n",
                    default_groups, base_sms_, requested_groups);
            }
        }

        groups_created_ = requested_groups;
        if (groups_created_ < encoder_groups + decoder_groups) {
            throw std::runtime_error(
                "CUDA created only " + std::to_string(groups_created_) + " partitions of " +
                std::to_string(base_sms_) + " SMs; need " +
                std::to_string(encoder_groups + decoder_groups) +
                ". Retried with CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING.");
        }

        encoder_.resources.assign(partitions.begin(), partitions.begin() + encoder_groups);
        decoder_.resources.assign(
            partitions.begin() + encoder_groups,
            partitions.begin() + encoder_groups + decoder_groups);
        encoder_.requested_sms = static_cast<unsigned int>(encoder_sms_);
        decoder_.requested_sms = static_cast<unsigned int>(decoder_sms_);
        encoder_.groups = encoder_groups;
        decoder_.groups = decoder_groups;

        CHECK_CU(cuDevResourceGenerateDesc(&encoder_.desc, encoder_.resources.data(), encoder_groups));
        CHECK_CU(cuDevResourceGenerateDesc(&decoder_.desc, decoder_.resources.data(), decoder_groups));
        CHECK_CU(cuGreenCtxCreate(&encoder_.green, encoder_.desc, device_, CU_GREEN_CTX_DEFAULT_STREAM));
        CHECK_CU(cuGreenCtxCreate(&decoder_.green, decoder_.desc, device_, CU_GREEN_CTX_DEFAULT_STREAM));
        encoder_.fill_ids_and_verify("encoder");
        decoder_.fill_ids_and_verify("decoder");

        if (encoder_.actual_sms != static_cast<unsigned int>(encoder_sms_) ||
            decoder_.actual_sms != static_cast<unsigned int>(decoder_sms_)) {
            throw std::runtime_error(
                "Actual GreenContext SM counts do not match requested counts: encoder actual/requested " +
                std::to_string(encoder_.actual_sms) + "/" + std::to_string(encoder_sms_) +
                ", decoder actual/requested " + std::to_string(decoder_.actual_sms) + "/" +
                std::to_string(decoder_sms_));
        }
    }

    void log_info() const {
        if (!verbose_) {
            return;
        }
        std::fprintf(stderr,
                     "[GreenCtx] device=%d total_sms=%u min_partition=%u alignment=%u base_sms=%d split_groups=%u split_flags=0x%x\n",
                     device_index_, device_resource_.sm.smCount, device_resource_.sm.minSmPartitionSize,
                     device_resource_.sm.smCoscheduledAlignment, base_sms_, groups_created_, split_use_flags_);
        std::fprintf(stderr,
                     "[GreenCtx] encoder: requested_sms=%u actual_sms=%u groups=%u green_id=%llu ctx_id=%llu stream_id=%llu stream_ptr=0x%llx\n",
                     encoder_.requested_sms, encoder_.actual_sms, encoder_.groups, encoder_.green_id,
                     encoder_.ctx_id, encoder_.stream_id,
                     static_cast<unsigned long long>(encoder_stream_ptr()));
        std::fprintf(stderr,
                     "[GreenCtx] decoder: requested_sms=%u actual_sms=%u groups=%u green_id=%llu ctx_id=%llu stream_id=%llu stream_ptr=0x%llx\n",
                     decoder_.requested_sms, decoder_.actual_sms, decoder_.groups, decoder_.green_id,
                     decoder_.ctx_id, decoder_.stream_id,
                     static_cast<unsigned long long>(decoder_stream_ptr()));
        std::fprintf(stderr,
                     "[GreenCtx] partitions are disjoint by construction: both descriptors use non-overlapping groups from one cuDevSmResourceSplitByCount call.\n");
    }

    int device_index_ = 0;
    int encoder_sms_ = 0;
    int decoder_sms_ = 0;
    bool verbose_ = true;
    CUdevice device_ = 0;
    CUcontext primary_ctx_ = nullptr;
    unsigned long long primary_ctx_id_ = 0;
    CUdevResource device_resource_;
    int base_sms_ = 0;
    unsigned int groups_created_ = 0;
    unsigned int split_use_flags_ = 0;
    OneGreenContext encoder_;
    OneGreenContext decoder_;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<GreenContextPair>(m, "GreenContextPair")
        .def(py::init<int, int, int, bool>(), py::arg("device_index"), py::arg("encoder_sms"),
             py::arg("decoder_sms"), py::arg("verbose") = true)
        .def("encoder_stream_ptr", &GreenContextPair::encoder_stream_ptr)
        .def("decoder_stream_ptr", &GreenContextPair::decoder_stream_ptr)
        .def("set_encoder_current", &GreenContextPair::set_encoder_current)
        .def("set_decoder_current", &GreenContextPair::set_decoder_current)
        .def("restore_primary_current", &GreenContextPair::restore_primary_current)
        .def("synchronize_encoder", &GreenContextPair::synchronize_encoder,
             py::call_guard<py::gil_scoped_release>())
        .def("synchronize_decoder", &GreenContextPair::synchronize_decoder,
             py::call_guard<py::gil_scoped_release>())
        .def("synchronize_both", &GreenContextPair::synchronize_both,
             py::call_guard<py::gil_scoped_release>())
        .def("info", &GreenContextPair::info);
}

#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

#include "libsmctrl.h"

namespace py = pybind11;

namespace {

void check_cuda(cudaError_t err, const char *what) {
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << what << " failed: " << cudaGetErrorString(err);
        throw std::runtime_error(oss.str());
    }
}

void check_smctrl(int err, const char *what) {
    if (err != 0) {
        std::ostringstream oss;
        oss << what << " failed with code " << err;
        throw std::runtime_error(oss.str());
    }
}

std::string mask_to_hex(uint128_t mask) {
    const auto low = static_cast<std::uint64_t>(mask);
    const auto high = static_cast<std::uint64_t>(mask >> 64);
    std::ostringstream oss;
    oss << "0x";
    if (high != 0) {
        oss << std::hex << std::setfill('0') << std::setw(16) << high
            << std::setw(16) << low;
    } else {
        oss << std::hex << std::setfill('0') << std::setw(16) << low;
    }
    return oss.str();
}

uint128_t make_mask_ext_checked(std::uint32_t low_tpc, std::uint32_t high_tpc) {
    uint128_t mask = 0;
    check_smctrl(
        libsmctrl_make_mask_ext(&mask, low_tpc, high_tpc),
        "libsmctrl_make_mask_ext");
    return mask;
}

py::dict mask_dict(uint128_t mask, std::uint32_t low_tpc, std::uint32_t high_tpc) {
    const auto low64 = static_cast<std::uint64_t>(mask);
    const auto high64 = static_cast<std::uint64_t>(mask >> 64);
    py::dict out;
    out["low_tpc"] = low_tpc;
    out["high_tpc"] = high_tpc;
    out["enabled_tpc_count"] = high_tpc - low_tpc;
    out["disabled_mask_hex"] = mask_to_hex(mask);
    out["disabled_mask_low64"] = low64;
    out["disabled_mask_high64"] = high64;
    out["semantics"] = "bit set means disabled TPC";
    return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Python bindings for libsmctrl stream TPC masks";

    m.def("get_tpc_info", [](int device) {
        cudaDeviceProp prop {};
        check_cuda(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");

        std::uint32_t total_tpcs = 0;
        check_smctrl(
            libsmctrl_get_tpc_info_cuda(&total_tpcs, device),
            "libsmctrl_get_tpc_info_cuda");
        int driver_version = 0;
        CUresult driver_result = cuDriverGetVersion(&driver_version);
        if (driver_result != CUDA_SUCCESS) {
            const char *name = nullptr;
            cuGetErrorName(driver_result, &name);
            std::ostringstream oss;
            oss << "cuDriverGetVersion failed: " << (name ? name : "unknown");
            throw std::runtime_error(oss.str());
        }
        if (total_tpcs == 0) {
            throw std::runtime_error("libsmctrl reported zero TPCs");
        }

        py::dict out;
        out["device_index"] = device;
        out["gpu_name"] = std::string(prop.name);
        out["compute_capability"] =
            std::to_string(prop.major) + "." + std::to_string(prop.minor);
        out["cuda_driver_version"] = driver_version;
        out["total_sms"] = prop.multiProcessorCount;
        out["total_tpcs"] = total_tpcs;
        out["sms_per_tpc_exact"] =
            static_cast<double>(prop.multiProcessorCount) / static_cast<double>(total_tpcs);
        out["sms_per_tpc_floor"] = prop.multiProcessorCount / static_cast<int>(total_tpcs);
        return out;
    }, py::arg("device") = 0);

    m.def("make_mask", [](std::uint32_t low_tpc, std::uint32_t high_tpc) {
        return mask_dict(make_mask_ext_checked(low_tpc, high_tpc), low_tpc, high_tpc);
    }, py::arg("low_tpc"), py::arg("high_tpc"));

    m.def("set_stream_mask", [](std::uintptr_t stream_ptr,
                                std::uint32_t low_tpc,
                                std::uint32_t high_tpc) {
        uint128_t mask = make_mask_ext_checked(low_tpc, high_tpc);
        check_smctrl(
            libsmctrl_set_stream_mask_ext(reinterpret_cast<void *>(stream_ptr), mask),
            "libsmctrl_set_stream_mask_ext");
        return mask_dict(mask, low_tpc, high_tpc);
    }, py::arg("stream_ptr"), py::arg("low_tpc"), py::arg("high_tpc"));

    m.def("validate_stream_mask", [](std::uintptr_t stream_ptr,
                                     int low_tpc,
                                     int high_tpc,
                                     bool echo) {
        check_smctrl(
            libsmctrl_validate_stream_mask(
                reinterpret_cast<void *>(stream_ptr), low_tpc, high_tpc, echo),
            "libsmctrl_validate_stream_mask");
        return true;
    }, py::arg("stream_ptr"), py::arg("low_tpc"), py::arg("high_tpc"),
       py::arg("echo") = false);
}

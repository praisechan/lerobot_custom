#include "green_context_utils.h"

#include <sstream>

static std::string cu_error_to_string(CUresult result) {
  const char* name = nullptr;
  const char* msg = nullptr;
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &msg);
  std::ostringstream oss;
  oss << (name ? name : "CUDA_ERROR_UNKNOWN") << ": "
      << (msg ? msg : "unknown error");
  return oss.str();
}

bool init_cuda_driver(std::string* err) {
  const CUresult res = cuInit(0);
  if (res != CUDA_SUCCESS) {
    if (err) {
      *err = "cuInit failed: " + cu_error_to_string(res);
    }
    return false;
  }
  return true;
}

bool get_sm_total(int* sm_total, std::string* err) {
  if (!sm_total) {
    if (err) {
      *err = "sm_total output pointer is null";
    }
    return false;
  }
  if (!init_cuda_driver(err)) {
    return false;
  }
  CUdevice dev = 0;
  CUresult res = cuDeviceGet(&dev, 0);
  if (res != CUDA_SUCCESS) {
    if (err) {
      *err = "cuDeviceGet failed: " + cu_error_to_string(res);
    }
    return false;
  }
  int sm = 0;
  res = cuDeviceGetAttribute(&sm, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
  if (res != CUDA_SUCCESS) {
    if (err) {
      *err = "cuDeviceGetAttribute failed: " + cu_error_to_string(res);
    }
    return false;
  }
  *sm_total = sm;
  return true;
}

bool green_contexts_supported() {
#if defined(CU_EXEC_AFFINITY_TYPE_SM_COUNT)
  return true;
#else
  return false;
#endif
}

bool create_context_with_sm_count(int sm_count,
                                  bool prefer_green,
                                  GreenContext* out,
                                  std::string* err,
                                  bool* used_green) {
  if (!out) {
    if (err) {
      *err = "GreenContext output pointer is null";
    }
    return false;
  }
  if (!init_cuda_driver(err)) {
    return false;
  }

  CUdevice dev = 0;
  CUresult res = cuDeviceGet(&dev, 0);
  if (res != CUDA_SUCCESS) {
    if (err) {
      *err = "cuDeviceGet failed: " + cu_error_to_string(res);
    }
    return false;
  }

  int sm_total = 0;
  res = cuDeviceGetAttribute(&sm_total, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
  if (res != CUDA_SUCCESS) {
    if (err) {
      *err = "cuDeviceGetAttribute failed: " + cu_error_to_string(res);
    }
    return false;
  }

  if (sm_count <= 0 || sm_count > sm_total) {
    if (err) {
      *err = "requested sm_count is out of range";
    }
    return false;
  }

  out->sm_count = sm_count;
  out->sm_total = sm_total;

#if defined(CU_EXEC_AFFINITY_TYPE_SM_COUNT)
  if (prefer_green) {
    CUexecAffinityParam params[1];
    params[0].type = CU_EXEC_AFFINITY_TYPE_SM_COUNT;
    params[0].param.smCount = sm_count;

    CUcontext ctx = nullptr;
    res = cuCtxCreate_v3(&ctx, params, 1, 0, dev);
    if (res == CUDA_SUCCESS) {
      out->ctx = ctx;
      out->is_green = true;
      out->is_primary = false;
      if (used_green) {
        *used_green = true;
      }
      return true;
    }
    if (err) {
      *err = "cuCtxCreate_v3 failed: " + cu_error_to_string(res);
    }
  }
#endif

  CUcontext primary = nullptr;
  res = cuDevicePrimaryCtxRetain(&primary, dev);
  if (res != CUDA_SUCCESS) {
    if (err) {
      *err = "cuDevicePrimaryCtxRetain failed: " + cu_error_to_string(res);
    }
    return false;
  }

  res = cuCtxSetCurrent(primary);
  if (res != CUDA_SUCCESS) {
    if (err) {
      *err = "cuCtxSetCurrent failed: " + cu_error_to_string(res);
    }
    return false;
  }

  out->ctx = primary;
  out->is_green = false;
  out->is_primary = true;
  if (used_green) {
    *used_green = false;
  }

  if (prefer_green && err) {
    if (err->empty()) {
      *err = "Green Context not available; falling back to primary context";
    } else {
      *err = "Green Context not available (" + *err + "); falling back to primary context";
    }
  }

  return true;
}

bool make_context_current(GreenContext* ctx, std::string* err) {
  if (!ctx || !ctx->ctx) {
    if (err) {
      *err = "context is null";
    }
    return false;
  }
  const CUresult res = cuCtxSetCurrent(ctx->ctx);
  if (res != CUDA_SUCCESS) {
    if (err) {
      *err = "cuCtxSetCurrent failed: " + cu_error_to_string(res);
    }
    return false;
  }
  return true;
}

void destroy_context(GreenContext* ctx) {
  if (!ctx || !ctx->ctx) {
    return;
  }
  if (ctx->is_green) {
    cuCtxDestroy(ctx->ctx);
  } else if (ctx->is_primary) {
    CUdevice dev = 0;
    if (cuDeviceGet(&dev, 0) == CUDA_SUCCESS) {
      cuDevicePrimaryCtxRelease(dev);
    }
  }
  ctx->ctx = nullptr;
  ctx->is_green = false;
  ctx->is_primary = false;
}

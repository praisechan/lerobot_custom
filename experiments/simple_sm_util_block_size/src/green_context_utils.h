#pragma once

#include <cuda.h>
#include <string>

struct GreenContext {
  CUcontext ctx = nullptr;
  bool is_green = false;
  bool is_primary = false;
  int sm_count = 0;
  int sm_total = 0;
};

bool init_cuda_driver(std::string* err);
bool get_sm_total(int* sm_total, std::string* err);
bool green_contexts_supported();

bool create_context_with_sm_count(int sm_count,
                                  bool prefer_green,
                                  GreenContext* out,
                                  std::string* err,
                                  bool* used_green);

bool make_context_current(GreenContext* ctx, std::string* err);
void destroy_context(GreenContext* ctx);

// Copyright 2024 RISC Zero, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <stdexcept>

template <typename... Types> inline std::string fmt(const char* fmt, Types... args) {
  size_t len = std::snprintf(nullptr, 0, fmt, args...);
  std::string ret(++len, '\0');
  std::snprintf(&ret.front(), len, fmt, args...);
  ret.resize(--len);
  return ret;
}

#define CUDA_OK(expr)                                                                              \
  do {                                                                                             \
    cudaError_t code = expr;                                                                       \
    if (code != cudaSuccess) {                                                                     \
      auto file = std::strstr(__FILE__, "sppark");                                                 \
      auto msg = fmt("%s@%s:%d failed: \"%s\"",                                                    \
                     #expr,                                                                        \
                     file ? file : __FILE__,                                                       \
                     __LINE__,                                                                     \
                     cudaGetErrorString(code));                                                    \
      throw std::runtime_error{msg};                                                               \
    }                                                                                              \
  } while (0)

class CudaStream {
private:
  cudaStream_t stream;

public:
  CudaStream() { cudaStreamCreate(&stream); }
  ~CudaStream() { cudaStreamDestroy(stream); }

  inline operator cudaStream_t() const { return stream; }
};

struct LaunchConfig {
  dim3 grid;
  dim3 block;
  size_t shared;

  LaunchConfig(dim3 grid, dim3 block, size_t shared = 0)
      : grid(grid), block(block), shared(shared) {}
  LaunchConfig(int grid, int block, size_t shared = 0) : grid(grid), block(block), shared(shared) {}
};

inline LaunchConfig getSimpleConfig(uint32_t count) {
  int device;
  CUDA_OK(cudaGetDevice(&device));

  int maxThreads;
  CUDA_OK(cudaDeviceGetAttribute(&maxThreads, cudaDevAttrMaxThreadsPerBlock, device));

  int block = maxThreads / 4;
  int grid = (count + block - 1) / block;
  return LaunchConfig{grid, block, 0};
}

inline LaunchConfig getOptimizedConfig(uint32_t count, const char* kernel_name = nullptr) {
  int device;
  CUDA_OK(cudaGetDevice(&device));

  cudaDeviceProp props;
  CUDA_OK(cudaGetDeviceProperties(&props, device));

  // Calculate optimal block size based on occupancy
  int block_size = 256; // Default

  // Kernel-specific optimizations
  if (kernel_name) {
    if (strstr(kernel_name, "ntt") || strstr(kernel_name, "fft")) {
      block_size = 512; // Compute-intensive
    } else if (strstr(kernel_name, "hash") || strstr(kernel_name, "mem")) {
      block_size = 128; // Memory-bound
    } else if (strstr(kernel_name, "accum")) {
      block_size = 256; // Balanced
    } else if (strstr(kernel_name, "witgen")) {
      block_size = 384; // Witness generation optimized
    }
  }

  // Ensure block size doesn't exceed device limits
  block_size = std::min(block_size, props.maxThreadsPerBlock);

  // Calculate grid size with better occupancy
  int grid = (count + block_size - 1) / block_size;

  // Limit grid size to multiprocessor count * 2 for better scheduling
  int max_grid = props.multiProcessorCount * 2;
  grid = std::min(grid, max_grid);

  return LaunchConfig{grid, block_size, 0};
}

template <typename... ExpTypes, typename... ActTypes>
const char* launchKernel(void (*kernel)(ExpTypes...),
                         uint32_t count,
                         uint32_t shared_size,
                         ActTypes&&... args) {
  try {
    CudaStream stream;
    LaunchConfig cfg = getOptimizedConfig(count, "generic");
    cudaLaunchConfig_t config;
    config.attrs = nullptr;
    config.numAttrs = 0;
    config.gridDim = cfg.grid;
    config.blockDim = cfg.block;
    config.dynamicSmemBytes = shared_size;
    config.stream = stream;
    CUDA_OK(cudaLaunchKernelEx(&config, kernel, std::forward<ActTypes>(args)...));
    CUDA_OK(cudaStreamSynchronize(stream));
  } catch (const std::exception& err) {
    return strdup(err.what());
  } catch (...) {
    return strdup("Generic exception");
  }
  return nullptr;
}

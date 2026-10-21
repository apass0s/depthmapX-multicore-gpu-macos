// depthmapX GPU module - shared helpers for the CUDA sources

#pragma once

#include "dmx_gpu_api.h"

#include <cuda_runtime.h>

#include <string>

namespace dmxgpu {

// Records the error message for dmx_gpu_last_error() and returns the API error code.
int setError(int code, const std::string &message);
int setCudaError(cudaError_t err, const char *call, const char *file, int line);
void clearError();

// Device memory that must be left free for the driver and other applications.
constexpr size_t kReservedDeviceBytes = size_t(256) << 20;

// Returns DMX_GPU_ERR_OUT_OF_MEMORY (with a message) if `bytes` more cannot be allocated.
int checkFreeMemory(size_t bytes, const char *what);

} // namespace dmxgpu

// Evaluates a CUDA runtime call; on failure records the error and returns from the
// enclosing function with the matching DMX_GPU_ERR_* code.
#define DMX_CUDA_CHECK(call)                                                                                           \
    do {                                                                                                               \
        cudaError_t dmx_err_ = (call);                                                                                 \
        if (dmx_err_ != cudaSuccess) {                                                                                 \
            return dmxgpu::setCudaError(dmx_err_, #call, __FILE__, __LINE__);                                         \
        }                                                                                                              \
    } while (0)

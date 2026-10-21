// depthmapX GPU module - error reporting and device probing

#include "dmx_gpu_common.cuh"

#include <cstdio>
#include <cstring>

namespace dmxgpu {

namespace {
thread_local std::string t_lastError;

__global__ void probeKernel(int *out) { *out = 42; }
} // namespace

int setError(int code, const std::string &message) {
    t_lastError = message;
    return code;
}

int setCudaError(cudaError_t err, const char *call, const char *file, int line) {
    // Clear sticky-free errors so later calls on this thread start clean.
    cudaGetLastError();
    char buf[512];
    std::snprintf(buf, sizeof(buf), "%s failed at %s:%d: %s (%s)", call, file, line, cudaGetErrorString(err),
                  cudaGetErrorName(err));
    int code = DMX_GPU_ERR_CUDA;
    switch (err) {
    case cudaErrorMemoryAllocation:
        code = DMX_GPU_ERR_OUT_OF_MEMORY;
        break;
    case cudaErrorNoDevice:
    case cudaErrorInsufficientDriver:
        code = DMX_GPU_ERR_NO_DEVICE;
        break;
    case cudaErrorNoKernelImageForDevice:
    case cudaErrorInvalidDeviceFunction:
    case cudaErrorUnsupportedPtxVersion:
        code = DMX_GPU_ERR_UNSUPPORTED;
        break;
    default:
        break;
    }
    return setError(code, buf);
}

void clearError() { t_lastError.clear(); }

int checkFreeMemory(size_t bytes, const char *what) {
    size_t freeBytes = 0, totalBytes = 0;
    DMX_CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));
    if (bytes + kReservedDeviceBytes > freeBytes) {
        char buf[256];
        std::snprintf(buf, sizeof(buf), "%s needs %.1f MiB of GPU memory but only %.1f MiB are free", what,
                      bytes / 1048576.0, freeBytes / 1048576.0);
        return setError(DMX_GPU_ERR_OUT_OF_MEMORY, buf);
    }
    return DMX_GPU_OK;
}

} // namespace dmxgpu

extern "C" {

DMX_GPU_API int dmx_gpu_api_version(void) { return DMX_GPU_API_VERSION; }

DMX_GPU_API const char *dmx_gpu_last_error(void) { return dmxgpu::t_lastError.c_str(); }

DMX_GPU_API int dmx_gpu_probe(dmx_gpu_device_info *info) {
    dmxgpu::clearError();
    if (!info) {
        return dmxgpu::setError(DMX_GPU_ERR_INVALID_ARG, "dmx_gpu_probe: info is null");
    }
    std::memset(info, 0, sizeof(*info));

    int count = 0;
    DMX_CUDA_CHECK(cudaGetDeviceCount(&count));
    if (count == 0) {
        return dmxgpu::setError(DMX_GPU_ERR_NO_DEVICE, "no CUDA device found");
    }
    DMX_CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp prop;
    DMX_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::snprintf(info->name, sizeof(info->name), "%s", prop.name);
    info->cc_major = prop.major;
    info->cc_minor = prop.minor;
    info->sm_count = prop.multiProcessorCount;
    DMX_CUDA_CHECK(cudaRuntimeGetVersion(&info->runtime_version));
    DMX_CUDA_CHECK(cudaDriverGetVersion(&info->driver_version));

    size_t freeBytes = 0, totalBytes = 0;
    DMX_CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));
    info->free_mem = freeBytes;
    info->total_mem = totalBytes;

    // Run a kernel: fails here (not mid-analysis) if this DLL has no image for the device.
    int *d = nullptr;
    DMX_CUDA_CHECK(cudaMalloc(&d, sizeof(int)));
    dmxgpu::probeKernel<<<1, 1>>>(d);
    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess) {
        err = cudaDeviceSynchronize();
    }
    int value = 0;
    if (err == cudaSuccess) {
        err = cudaMemcpy(&value, d, sizeof(int), cudaMemcpyDeviceToHost);
    }
    cudaFree(d);
    if (err != cudaSuccess) {
        return dmxgpu::setCudaError(err, "probe kernel", __FILE__, __LINE__);
    }
    if (value != 42) {
        return dmxgpu::setError(DMX_GPU_ERR_CUDA, "probe kernel returned a wrong value");
    }
    return DMX_GPU_OK;
}

} // extern "C"

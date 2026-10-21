// depthmapX GPU module - VGA visual local measures
//
// One thread block per source node u. The block keeps a slice of row u (the pixels u
// sees) and the running union of its neighbours' rows in shared memory, then walks
// the neighbours v of u: popcount(row v & row u) is the number of v's pixels that u
// also sees, and OR-ing row v into the union gives the "total neighbourhood". Rows
// wider than the shared-memory slice are processed in several word ranges.
//
// All quantities are integers, so results are exactly those of the CPU code.

#include "dmx_gpu_common.cuh"

#include <cstdio>
#include <vector>

namespace {

constexpr int kThreads = 256;
// Words per shared-memory slice: two uint64 arrays of this size use 48 KiB, the
// default dynamic shared memory limit, so no opt-in attribute is needed.
constexpr int kMaxSliceWords = 3072;

__global__ void visLocalKernel(const unsigned long long *__restrict__ rows, int wordsPerRow,
                               const int *__restrict__ pixelToRow, const long long *__restrict__ dupOffsets,
                               const int *__restrict__ dupPixels, int firstRow, int sliceStart, int sliceWords,
                               bool addDuplicates, long long *__restrict__ clusterAcc, int *__restrict__ unionAcc) {
    extern __shared__ unsigned long long shared[];
    unsigned long long *sliceU = shared;
    unsigned long long *sliceUnion = shared + sliceWords;

    const int u = firstRow + blockIdx.x;
    const unsigned long long *rowU = rows + size_t(u) * wordsPerRow;

    for (int t = threadIdx.x; t < sliceWords; t += blockDim.x) {
        sliceU[t] = rowU[sliceStart + t];
        sliceUnion[t] = 0ull;
    }
    __syncthreads();

    long long intersect = 0;
    // Every thread walks the same neighbour list, so control flow stays uniform.
    for (int w = 0; w < wordsPerRow; ++w) {
        unsigned long long bits = rowU[w];
        while (bits) {
            const int bit = __ffsll(bits) - 1;
            bits &= bits - 1;
            const int v = pixelToRow[w * 64 + bit];
            if (v < 0) {
                continue;
            }
            const unsigned long long *rowV = rows + size_t(v) * wordsPerRow + sliceStart;
            for (int t = threadIdx.x; t < sliceWords; t += blockDim.x) {
                const unsigned long long x = rowV[t];
                intersect += __popcll(x & sliceU[t]);
                sliceUnion[t] |= x;
            }
            if (addDuplicates && threadIdx.x == 0) {
                for (long long d = dupOffsets[v]; d < dupOffsets[v + 1]; ++d) {
                    const int p = dupPixels[d];
                    intersect += (rowU[p >> 6] >> (p & 63)) & 1ull;
                }
            }
        }
    }

    int unionCount = 0;
    for (int t = threadIdx.x; t < sliceWords; t += blockDim.x) {
        unionCount += __popcll(sliceUnion[t]);
    }

    // Block reduction of both counters.
    __shared__ long long redIntersect[kThreads / 32];
    __shared__ int redUnion[kThreads / 32];
    for (int offset = 16; offset > 0; offset >>= 1) {
        intersect += __shfl_down_sync(0xffffffffu, intersect, offset);
        unionCount += __shfl_down_sync(0xffffffffu, unionCount, offset);
    }
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (lane == 0) {
        redIntersect[warp] = intersect;
        redUnion[warp] = unionCount;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        long long totalIntersect = 0;
        int totalUnion = 0;
        for (int i = 0; i < kThreads / 32; ++i) {
            totalIntersect += redIntersect[i];
            totalUnion += redUnion[i];
        }
        clusterAcc[blockIdx.x] += totalIntersect;
        unionAcc[blockIdx.x] += totalUnion;
    }
}

template <typename T> int upload(T **dev, const T *host, size_t count) {
    *dev = nullptr;
    if (count == 0) { // keep a valid pointer for empty arrays
        DMX_CUDA_CHECK(cudaMalloc(dev, sizeof(T)));
        DMX_CUDA_CHECK(cudaMemset(*dev, 0, sizeof(T)));
        return DMX_GPU_OK;
    }
    DMX_CUDA_CHECK(cudaMalloc(dev, count * sizeof(T)));
    DMX_CUDA_CHECK(cudaMemcpy(*dev, host, count * sizeof(T), cudaMemcpyHostToDevice));
    return DMX_GPU_OK;
}

} // namespace

struct dmx_vis_local {
    unsigned long long *rows = nullptr;
    int *pixelToRow = nullptr;
    long long *dupOffsets = nullptr;
    int *dupPixels = nullptr;
    long long *clusterAcc = nullptr;
    int *unionAcc = nullptr;
    int nRows = 0;
    int wordsPerRow = 0;
    int nPixels = 0;
    bool hasDuplicates = false;
    int batchCapacity = 0;

    ~dmx_vis_local() {
        cudaFree(rows);
        cudaFree(pixelToRow);
        cudaFree(dupOffsets);
        cudaFree(dupPixels);
        cudaFree(clusterAcc);
        cudaFree(unionAcc);
    }
};

extern "C" {

DMX_GPU_API int dmx_vis_local_create(dmx_vis_local **out, const uint64_t *rows, int32_t nRows, int32_t wordsPerRow,
                                     const int32_t *pixelToRow, int32_t nPixels, const int64_t *dupOffsets,
                                     const int32_t *dupPixels) {
    dmxgpu::clearError();
    if (!out || !rows || !pixelToRow || !dupOffsets || nRows <= 0 || wordsPerRow <= 0 || nPixels <= 0 ||
        int64_t(wordsPerRow) * 64 < nPixels) {
        return dmxgpu::setError(DMX_GPU_ERR_INVALID_ARG, "dmx_vis_local_create: invalid arguments");
    }
    *out = nullptr;
    const int64_t nDups = dupOffsets[nRows];
    if (nDups > 0 && !dupPixels) {
        return dmxgpu::setError(DMX_GPU_ERR_INVALID_ARG, "dmx_vis_local_create: dup_pixels is null");
    }

    const size_t rowBytes = size_t(nRows) * wordsPerRow * sizeof(unsigned long long);
    const size_t needed = rowBytes + size_t(nPixels) * sizeof(int) + size_t(nRows + 1) * sizeof(long long) +
                          size_t(nDups) * sizeof(int);
    if (int rc = dmxgpu::checkFreeMemory(needed, "VGA visual local")) {
        return rc;
    }

    auto *ctx = new dmx_vis_local();
    int rc = DMX_GPU_OK;
    if (!rc) rc = upload(&ctx->rows, reinterpret_cast<const unsigned long long *>(rows), size_t(nRows) * wordsPerRow);
    if (!rc) rc = upload(&ctx->pixelToRow, reinterpret_cast<const int *>(pixelToRow), size_t(nPixels));
    if (!rc) rc = upload(&ctx->dupOffsets, reinterpret_cast<const long long *>(dupOffsets), size_t(nRows) + 1);
    if (!rc) rc = upload(&ctx->dupPixels, reinterpret_cast<const int *>(dupPixels), size_t(nDups));
    if (rc) {
        delete ctx;
        return rc;
    }
    ctx->nRows = nRows;
    ctx->wordsPerRow = wordsPerRow;
    ctx->nPixels = nPixels;
    ctx->hasDuplicates = nDups > 0;
    *out = ctx;
    return DMX_GPU_OK;
}

DMX_GPU_API int dmx_vis_local_run(dmx_vis_local *ctx, int32_t firstRow, int32_t nRows, int64_t *clusterOut,
                                  int32_t *unionOut) {
    dmxgpu::clearError();
    if (!ctx || !clusterOut || !unionOut || firstRow < 0 || nRows <= 0 || firstRow + int64_t(nRows) > ctx->nRows) {
        return dmxgpu::setError(DMX_GPU_ERR_INVALID_ARG, "dmx_vis_local_run: invalid arguments");
    }
    if (nRows > ctx->batchCapacity) {
        cudaFree(ctx->clusterAcc);
        cudaFree(ctx->unionAcc);
        ctx->clusterAcc = nullptr;
        ctx->unionAcc = nullptr;
        ctx->batchCapacity = 0;
        DMX_CUDA_CHECK(cudaMalloc(&ctx->clusterAcc, size_t(nRows) * sizeof(long long)));
        DMX_CUDA_CHECK(cudaMalloc(&ctx->unionAcc, size_t(nRows) * sizeof(int)));
        ctx->batchCapacity = nRows;
    }
    DMX_CUDA_CHECK(cudaMemset(ctx->clusterAcc, 0, size_t(nRows) * sizeof(long long)));
    DMX_CUDA_CHECK(cudaMemset(ctx->unionAcc, 0, size_t(nRows) * sizeof(int)));

    for (int sliceStart = 0; sliceStart < ctx->wordsPerRow; sliceStart += kMaxSliceWords) {
        const int sliceWords = (ctx->wordsPerRow - sliceStart < kMaxSliceWords) ? ctx->wordsPerRow - sliceStart
                                                                                : kMaxSliceWords;
        const size_t sharedBytes = size_t(2) * sliceWords * sizeof(unsigned long long);
        visLocalKernel<<<nRows, kThreads, sharedBytes>>>(ctx->rows, ctx->wordsPerRow, ctx->pixelToRow,
                                                         ctx->dupOffsets, ctx->dupPixels, firstRow, sliceStart,
                                                         sliceWords, ctx->hasDuplicates && sliceStart == 0,
                                                         ctx->clusterAcc, ctx->unionAcc);
        DMX_CUDA_CHECK(cudaGetLastError());
    }
    DMX_CUDA_CHECK(cudaDeviceSynchronize());
    DMX_CUDA_CHECK(cudaMemcpy(clusterOut, ctx->clusterAcc, size_t(nRows) * sizeof(long long), cudaMemcpyDeviceToHost));
    DMX_CUDA_CHECK(cudaMemcpy(unionOut, ctx->unionAcc, size_t(nRows) * sizeof(int), cudaMemcpyDeviceToHost));
    return DMX_GPU_OK;
}

DMX_GPU_API void dmx_vis_local_destroy(dmx_vis_local *ctx) { delete ctx; }

} // extern "C"

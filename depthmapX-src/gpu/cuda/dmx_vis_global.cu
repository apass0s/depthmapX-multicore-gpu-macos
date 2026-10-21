// depthmapX GPU module - VGA visual global measures (multi-source BFS)
//
// Up to `lanes` breadth-first searches run together; node v keeps one bit per search
// in `frontier` (reached at the current level) and `visited`. A level expands every
// frontier node through its visibility runs. A run is a contiguous range of positions
// along a grid line, so OR-ing a node's frontier bits into all cells of a run is a
// range update. Each run is split into two overlapping power-of-two blocks (OR is
// idempotent, so the overlap is harmless):
//   1. pull:      block[o][k][p] = OR of frontier bits of the nodes with a block at p
//   2. downsweep: for k = K-1 .. 1, a block of size 2^k at p covers the blocks of
//                 size 2^(k-1) at p and p + 2^(k-1); gathered as
//                 block[k-1][p] |= block[k][p] | block[k][p - 2^(k-1)]
//   3. combine:   reached = OR over orientations of block[o][0][position of v];
//                 next frontier = reached & ~visited
// Level sizes are counted per lane. All arithmetic is integer, so the depth
// distribution equals the CPU search's.

#include "dmx_gpu_common.cuh"

#include <algorithm>
#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr int kOrientations = 4;
constexpr int kCountChunk = 256; // nodes per block in the counting kernel

using u64 = unsigned long long;

inline unsigned blocksFor(size_t threads) { return unsigned((threads + kThreads - 1) / kThreads); }

__global__ void seedKernel(u64 *frontier, u64 *visited, int laneWords, int firstSource, int nSources) {
    const int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= nSources) {
        return;
    }
    const size_t word = size_t(firstSource + lane) * laneWords + lane / 64;
    const u64 bit = 1ull << (lane % 64);
    frontier[word] = bit; // one lane per source node, so no other thread writes this node
    visited[word] = bit;
}

__global__ void pullKernel(const long long *__restrict__ keyOffsets, const int *__restrict__ keyNodes,
                           const u64 *__restrict__ frontier, u64 *__restrict__ blocks, size_t nKeys, int laneWords) {
    const size_t t = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= nKeys * laneWords) {
        return;
    }
    const size_t key = t / laneWords;
    const int w = int(t % laneWords);
    u64 acc = 0;
    for (long long e = keyOffsets[key], end = keyOffsets[key + 1]; e < end; ++e) {
        acc |= frontier[size_t(keyNodes[e]) * laneWords + w];
    }
    blocks[t] = acc;
}

__global__ void downsweepKernel(u64 *blocks, int nLevels, size_t nPositions, int laneWords, int k) {
    const size_t t = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t perOrientation = nPositions * laneWords;
    if (t >= kOrientations * perOrientation) {
        return;
    }
    const int o = int(t / perOrientation);
    const size_t rem = t % perOrientation;
    const size_t p = rem / laneWords;
    const int w = int(rem % laneWords);
    const size_t half = size_t(1) << (k - 1);
    const size_t levelK = (size_t(o) * nLevels + k) * nPositions;
    const size_t levelBelow = (size_t(o) * nLevels + (k - 1)) * nPositions;
    u64 v = blocks[(levelK + p) * laneWords + w];
    if (p >= half) {
        v |= blocks[(levelK + p - half) * laneWords + w];
    }
    if (v) {
        blocks[(levelBelow + p) * laneWords + w] |= v;
    }
}

__global__ void combineKernel(const u64 *__restrict__ blocks, const int *__restrict__ nodePositions, u64 *frontier,
                              u64 *visited, int nNodes, int nLevels, size_t nPositions, int laneWords,
                              int *anyReached) {
    const size_t t = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= size_t(nNodes) * laneWords) {
        return;
    }
    const int v = int(t / laneWords);
    const int w = int(t % laneWords);
    u64 reached = 0;
    for (int o = 0; o < kOrientations; ++o) {
        const size_t level0 = size_t(o) * nLevels * nPositions;
        reached |= blocks[(level0 + size_t(nodePositions[size_t(o) * nNodes + v])) * laneWords + w];
    }
    const u64 next = reached & ~visited[t];
    visited[t] |= next;
    frontier[t] = next;
    if (next) {
        *anyReached = 1;
    }
}

// One block per (chunk of nodes, lane word); thread b counts lane w * 64 + b.
__global__ void countKernel(const u64 *__restrict__ frontier, int nNodes, int laneWords, int nSources,
                            int *levelCounts) {
    const int w = blockIdx.y;
    const int lane = w * 64 + threadIdx.x;
    const int begin = blockIdx.x * kCountChunk;
    const int end = min(nNodes, begin + kCountChunk);
    int count = 0;
    for (int v = begin; v < end; ++v) {
        count += int((frontier[size_t(v) * laneWords + w] >> threadIdx.x) & 1ull);
    }
    if (count && lane < nSources) {
        atomicAdd(&levelCounts[lane], count);
    }
}

} // namespace

struct dmx_vis_global {
    long long *keyOffsets = nullptr;
    int *keyNodes = nullptr;
    int *nodePositions = nullptr;
    u64 *blocks = nullptr;
    u64 *frontier = nullptr;
    u64 *visited = nullptr;
    int *levelCounts = nullptr;
    int *anyReached = nullptr;
    int nNodes = 0;
    size_t nPositions = 0;
    int nLevels = 0;
    int lanes = 0;
    int laneWords = 0;

    ~dmx_vis_global() {
        cudaFree(keyOffsets);
        cudaFree(keyNodes);
        cudaFree(nodePositions);
        cudaFree(blocks);
        cudaFree(frontier);
        cudaFree(visited);
        cudaFree(levelCounts);
        cudaFree(anyReached);
    }
};

extern "C" {

DMX_GPU_API int dmx_vis_global_create(dmx_vis_global **out, int32_t nNodes, int32_t nPositions, int32_t nLevels,
                                      const int64_t *keyOffsets, const int32_t *keyNodes,
                                      const int32_t *nodePositions, int32_t maxLanes) {
    dmxgpu::clearError();
    if (!out || !keyOffsets || !nodePositions || nNodes <= 0 || nPositions <= 0 || nLevels <= 0 || nLevels > 31 ||
        maxLanes < 64) {
        return dmxgpu::setError(DMX_GPU_ERR_INVALID_ARG, "dmx_vis_global_create: invalid arguments");
    }
    *out = nullptr;
    const size_t nKeys = size_t(kOrientations) * nLevels * nPositions;
    const int64_t nEntries = keyOffsets[nKeys];
    if (nEntries > 0 && !keyNodes) {
        return dmxgpu::setError(DMX_GPU_ERR_INVALID_ARG, "dmx_vis_global_create: key_nodes is null");
    }

    const size_t fixedBytes = (nKeys + 1) * sizeof(long long) + size_t(nEntries) * sizeof(int) +
                              size_t(kOrientations) * nNodes * sizeof(int);
    size_t freeBytes = 0, totalBytes = 0;
    DMX_CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));
    // Largest lane count (multiple of 64, at most maxLanes) whose buffers fit.
    int lanes = (maxLanes / 64) * 64;
    auto laneBytes = [&](int l) {
        const size_t words = size_t(l / 64);
        return nKeys * words * sizeof(u64) + 2 * size_t(nNodes) * words * sizeof(u64) + size_t(l) * sizeof(int);
    };
    while (lanes > 64 && fixedBytes + laneBytes(lanes) + dmxgpu::kReservedDeviceBytes > freeBytes) {
        lanes -= 64;
    }
    if (int rc = dmxgpu::checkFreeMemory(fixedBytes + laneBytes(lanes), "VGA visual global")) {
        return rc;
    }

    auto *ctx = new dmx_vis_global();
    ctx->nNodes = nNodes;
    ctx->nPositions = size_t(nPositions);
    ctx->nLevels = nLevels;
    ctx->lanes = lanes;
    ctx->laneWords = lanes / 64;
    auto fail = [&](int rc) {
        delete ctx;
        return rc;
    };
#define DMX_CTX_CHECK(call)                                                                                            \
    do {                                                                                                               \
        cudaError_t e_ = (call);                                                                                       \
        if (e_ != cudaSuccess) {                                                                                       \
            return fail(dmxgpu::setCudaError(e_, #call, __FILE__, __LINE__));                                         \
        }                                                                                                              \
    } while (0)
    DMX_CTX_CHECK(cudaMalloc(&ctx->keyOffsets, (nKeys + 1) * sizeof(long long)));
    DMX_CTX_CHECK(cudaMemcpy(ctx->keyOffsets, keyOffsets, (nKeys + 1) * sizeof(long long), cudaMemcpyHostToDevice));
    DMX_CTX_CHECK(cudaMalloc(&ctx->keyNodes, size_t(std::max<int64_t>(nEntries, 1)) * sizeof(int)));
    if (nEntries > 0) {
        DMX_CTX_CHECK(cudaMemcpy(ctx->keyNodes, keyNodes, size_t(nEntries) * sizeof(int), cudaMemcpyHostToDevice));
    }
    DMX_CTX_CHECK(cudaMalloc(&ctx->nodePositions, size_t(kOrientations) * nNodes * sizeof(int)));
    DMX_CTX_CHECK(cudaMemcpy(ctx->nodePositions, nodePositions, size_t(kOrientations) * nNodes * sizeof(int),
                             cudaMemcpyHostToDevice));
    const size_t words = size_t(ctx->laneWords);
    DMX_CTX_CHECK(cudaMalloc(&ctx->blocks, nKeys * words * sizeof(u64)));
    DMX_CTX_CHECK(cudaMalloc(&ctx->frontier, size_t(nNodes) * words * sizeof(u64)));
    DMX_CTX_CHECK(cudaMalloc(&ctx->visited, size_t(nNodes) * words * sizeof(u64)));
    DMX_CTX_CHECK(cudaMalloc(&ctx->levelCounts, size_t(lanes) * sizeof(int)));
    DMX_CTX_CHECK(cudaMalloc(&ctx->anyReached, sizeof(int)));
#undef DMX_CTX_CHECK
    *out = ctx;
    return DMX_GPU_OK;
}

DMX_GPU_API int dmx_vis_global_lanes(const dmx_vis_global *ctx) { return ctx ? ctx->lanes : 0; }

DMX_GPU_API int dmx_vis_global_run(dmx_vis_global *ctx, int32_t firstSource, int32_t nSources, int32_t maxDepth,
                                   int32_t *levelCounts, int32_t levelCapacity, int32_t *nLevelsOut) {
    dmxgpu::clearError();
    if (!ctx || !levelCounts || !nLevelsOut || firstSource < 0 || nSources <= 0 || nSources > ctx->lanes ||
        firstSource + int64_t(nSources) > ctx->nNodes || levelCapacity <= 0) {
        return dmxgpu::setError(DMX_GPU_ERR_INVALID_ARG, "dmx_vis_global_run: invalid arguments");
    }
    const int lw = ctx->laneWords;
    const size_t nodeWords = size_t(ctx->nNodes) * lw;
    const size_t nKeys = size_t(kOrientations) * ctx->nLevels * ctx->nPositions;

    DMX_CUDA_CHECK(cudaMemset(ctx->frontier, 0, nodeWords * sizeof(u64)));
    DMX_CUDA_CHECK(cudaMemset(ctx->visited, 0, nodeWords * sizeof(u64)));
    seedKernel<<<blocksFor(size_t(nSources)), kThreads>>>(ctx->frontier, ctx->visited, lw, firstSource, nSources);
    DMX_CUDA_CHECK(cudaGetLastError());

    // Level 0: every source reaches itself.
    for (int i = 0; i < nSources; ++i) {
        levelCounts[i] = 1;
    }
    int level = 0;
    while (maxDepth < 0 || level < maxDepth) {
        pullKernel<<<blocksFor(nKeys * lw), kThreads>>>(ctx->keyOffsets, ctx->keyNodes, ctx->frontier, ctx->blocks,
                                                       nKeys, lw);
        DMX_CUDA_CHECK(cudaGetLastError());
        for (int k = ctx->nLevels - 1; k >= 1; --k) {
            downsweepKernel<<<blocksFor(size_t(kOrientations) * ctx->nPositions * lw), kThreads>>>(
                ctx->blocks, ctx->nLevels, ctx->nPositions, lw, k);
            DMX_CUDA_CHECK(cudaGetLastError());
        }
        DMX_CUDA_CHECK(cudaMemset(ctx->anyReached, 0, sizeof(int)));
        combineKernel<<<blocksFor(nodeWords), kThreads>>>(ctx->blocks, ctx->nodePositions, ctx->frontier,
                                                         ctx->visited, ctx->nNodes, ctx->nLevels, ctx->nPositions,
                                                         lw, ctx->anyReached);
        DMX_CUDA_CHECK(cudaGetLastError());
        int anyReached = 0;
        DMX_CUDA_CHECK(cudaMemcpy(&anyReached, ctx->anyReached, sizeof(int), cudaMemcpyDeviceToHost));
        if (!anyReached) {
            break;
        }
        ++level;
        if (level >= levelCapacity) {
            *nLevelsOut = level + 1;
            return dmxgpu::setError(DMX_GPU_ERR_INVALID_ARG, "dmx_vis_global_run: level capacity exceeded");
        }
        DMX_CUDA_CHECK(cudaMemset(ctx->levelCounts, 0, size_t(nSources) * sizeof(int)));
        dim3 grid(unsigned((ctx->nNodes + kCountChunk - 1) / kCountChunk), unsigned(lw));
        countKernel<<<grid, 64>>>(ctx->frontier, ctx->nNodes, lw, nSources, ctx->levelCounts);
        DMX_CUDA_CHECK(cudaGetLastError());
        DMX_CUDA_CHECK(cudaMemcpy(levelCounts + size_t(level) * nSources, ctx->levelCounts,
                                  size_t(nSources) * sizeof(int), cudaMemcpyDeviceToHost));
    }
    *nLevelsOut = level + 1;
    return DMX_GPU_OK;
}

DMX_GPU_API void dmx_vis_global_destroy(dmx_vis_global *ctx) { delete ctx; }

} // extern "C"

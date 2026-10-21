// depthmapX GPU module (Metal) - the C interface (dmx_gpu_api.h) on the device layer
//
// Same contract and results as the CUDA module (gpu/cuda); the kernels are in
// dmx_metal_kernels.metal. What differs is planned here:
//  - memory is unified: the budget is the GPU's recommended working set minus what is
//    allocated and a reserve for the rest of the system, and no buffer may exceed the
//    device's maximum buffer length;
//  - grids are indexed with 32-bit thread positions, so lanes are reduced until every
//    grid has fewer than 2^32 threads;
//  - threadgroup memory and threads per threadgroup come from the device, so the row
//    slices of the local measures are sized at run time (Apple GPUs: 32 KiB);
//  - the local measures are committed in command buffers of kRowsPerCommandBuffer rows,
//    so no command buffer runs long enough to trip the macOS GPU watchdog.

#include "dmx_metal_backend.h"
#include "dmx_metal_shared.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <vector>

using dmxmetal::Buffer;
using dmxmetal::Device;
using dmxmetal::Kernel;

namespace {

constexpr uint64_t kReservedBytes = uint64_t(512) << 20; // left to the system and other apps
constexpr uint64_t kGridLimit = uint64_t(1) << 32;       // 32-bit thread positions
constexpr int kRowsPerCommandBuffer = 256;
constexpr uint32_t kMaxGroupThreads = 256;

thread_local std::string t_lastError;

int setError(int code, const std::string &message) {
    t_lastError = message;
    return code;
}

void clearError() { t_lastError.clear(); }

// One analysis records commands at a time; the lock also guards the device creation.
std::mutex g_mutex;
std::unique_ptr<Device> g_device;
bool g_deviceTried = false;
int g_deviceCode = DMX_GPU_OK;
std::string g_deviceError;

Device *device(int &code) { // caller holds g_mutex
    if (!g_deviceTried) {
        g_deviceTried = true;
        g_device = dmxmetal::createDevice(&g_deviceCode, g_deviceError);
    }
    if (!g_device) {
        code = setError(g_deviceCode != DMX_GPU_OK ? g_deviceCode : DMX_GPU_ERR_NO_DEVICE,
                        g_deviceError.empty() ? "no Metal device" : g_deviceError);
        return nullptr;
    }
    return g_device.get();
}

uint64_t availableBytes(const Device &dev) {
    const uint64_t limit = dev.workingSetBytes(), used = dev.allocatedBytes() + kReservedBytes;
    return limit > used ? limit - used : 0;
}

int checkMemory(const Device &dev, uint64_t bytes, const char *what) {
    const uint64_t available = availableBytes(dev);
    if (bytes > available) {
        char buf[256];
        std::snprintf(buf, sizeof(buf), "%s needs %.1f MiB of GPU memory but only %.1f MiB are available", what,
                      bytes / 1048576.0, available / 1048576.0);
        return setError(DMX_GPU_ERR_OUT_OF_MEMORY, buf);
    }
    return DMX_GPU_OK;
}

size_t roundUp16(size_t bytes) { return (bytes + 15) & ~size_t(15); }
uint32_t groupsFor(uint64_t threads, uint32_t perGroup) { return uint32_t((threads + perGroup - 1) / perGroup); }

// Threads per threadgroup for a kernel: at most kMaxGroupThreads, a multiple of the
// 32-wide SIMD groups when the pipeline allows that many.
uint32_t groupThreads(const Device &dev, Kernel kernel) {
    uint32_t t = std::min(kMaxGroupThreads, dev.maxThreadsPerThreadgroup(kernel));
    return t >= 32 ? t / 32 * 32 : std::max<uint32_t>(t, 1);
}

// Allocates (at least 16 bytes) and fills with `bytes` of data when data is given.
int makeBuffer(Device &dev, Buffer *&out, const void *data, size_t bytes, const char *what) {
    out = nullptr;
    if (bytes > dev.maxBufferBytes()) {
        char buf[256];
        std::snprintf(buf, sizeof(buf), "%s: a buffer of %.1f MiB exceeds the Metal limit of %.1f MiB", what,
                      bytes / 1048576.0, dev.maxBufferBytes() / 1048576.0);
        return setError(DMX_GPU_ERR_OUT_OF_MEMORY, buf);
    }
    std::string error;
    out = dev.allocate(std::max<size_t>(bytes, 16), error);
    if (!out) {
        return setError(DMX_GPU_ERR_OUT_OF_MEMORY, std::string(what) + ": " + error);
    }
    if (data && bytes) {
        std::memcpy(dev.contents(out), data, bytes);
    }
    return DMX_GPU_OK;
}

} // namespace

struct dmx_vis_local {
    Device *dev = nullptr;
    Buffer *rows = nullptr, *pixelToRow = nullptr, *dupOffsets = nullptr, *dupPixels = nullptr;
    Buffer *clusterAcc = nullptr, *unionAcc = nullptr;
    int nRows = 0, wordsPerRow = 0, nPixels = 0, sliceWords = 0, batchCapacity = 0;
    uint32_t threads = 0;
    bool hasDuplicates = false;

    ~dmx_vis_local() {
        for (Buffer *b : {rows, pixelToRow, dupOffsets, dupPixels, clusterAcc, unionAcc}) {
            if (b) {
                dev->release(b);
            }
        }
    }
};

struct dmx_vis_global {
    Device *dev = nullptr;
    Buffer *keyOffsets = nullptr, *keyNodes = nullptr, *nodePositions = nullptr, *blocks = nullptr;
    Buffer *frontier = nullptr, *visited = nullptr, *levelCounts = nullptr, *anyReached = nullptr;
    int nNodes = 0, nPositions = 0, nLevels = 0, lanes = 0, laneWords = 0;
    uint64_t nKeys = 0;
    uint32_t pullThreads = 0, downsweepThreads = 0, combineThreads = 0, countThreads = 0;

    ~dmx_vis_global() {
        for (Buffer *b : {keyOffsets, keyNodes, nodePositions, blocks, frontier, visited, levelCounts, anyReached}) {
            if (b) {
                dev->release(b);
            }
        }
    }
};

extern "C" {

DMX_GPU_API int dmx_gpu_api_version(void) { return DMX_GPU_API_VERSION; }

DMX_GPU_API const char *dmx_gpu_last_error(void) { return t_lastError.c_str(); }

DMX_GPU_API int dmx_gpu_probe(dmx_gpu_device_info *info) {
    clearError();
    if (!info) {
        return setError(DMX_GPU_ERR_INVALID_ARG, "dmx_gpu_probe: info is null");
    }
    std::memset(info, 0, sizeof(*info));
    std::lock_guard<std::mutex> lock(g_mutex);
    int rc = DMX_GPU_OK;
    Device *dev = device(rc);
    if (!dev) {
        return rc;
    }
    // Run a kernel: fails here, not in the middle of an analysis, if the GPU cannot.
    Buffer *out = nullptr;
    if ((rc = makeBuffer(*dev, out, nullptr, sizeof(int), "probe"))) {
        return rc;
    }
    *static_cast<int *>(dev->contents(out)) = 0;
    std::string error;
    bool ok = dev->begin(error);
    if (ok) {
        dev->setBuffer(out, 0, 0);
        dev->dispatch(Kernel::Probe, 1, 1, 1);
        ok = dev->commit(true, error);
    }
    const int value = *static_cast<int *>(dev->contents(out));
    dev->release(out);
    if (!ok) {
        return setError(DMX_GPU_ERR_CUDA, "probe kernel: " + error);
    }
    if (value != 42) {
        return setError(DMX_GPU_ERR_CUDA, "probe kernel returned a wrong value");
    }
    dev->describe(*info);
    return DMX_GPU_OK;
}

// ---------------------------------------------------------------------------------
// VGA visual local measures

DMX_GPU_API int dmx_vis_local_create(dmx_vis_local **out, const uint64_t *rows, int32_t nRows, int32_t wordsPerRow,
                                     const int32_t *pixelToRow, int32_t nPixels, const int64_t *dupOffsets,
                                     const int32_t *dupPixels) {
    clearError();
    if (!out || !rows || !pixelToRow || !dupOffsets || nRows <= 0 || wordsPerRow <= 0 || nPixels <= 0 ||
        int64_t(wordsPerRow) * 64 < nPixels) {
        return setError(DMX_GPU_ERR_INVALID_ARG, "dmx_vis_local_create: invalid arguments");
    }
    *out = nullptr;
    const int64_t nDups = dupOffsets[nRows];
    if (nDups > 0 && !dupPixels) {
        return setError(DMX_GPU_ERR_INVALID_ARG, "dmx_vis_local_create: dup_pixels is null");
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    int rc = DMX_GPU_OK;
    Device *dev = device(rc);
    if (!dev) {
        return rc;
    }

    // Threadgroup memory: two row slices (u64) plus one reduction slot (i64 + int) per thread.
    const uint32_t threads = groupThreads(*dev, Kernel::VisLocal);
    const size_t reductionBytes = roundUp16(size_t(threads) * 8) + roundUp16(size_t(threads) * 4);
    const size_t tgMemory = dev->maxThreadgroupMemory();
    if (tgMemory < reductionBytes + 32) {
        return setError(DMX_GPU_ERR_UNSUPPORTED, "dmx_vis_local_create: too little threadgroup memory");
    }
    int sliceWords = int(std::min<size_t>((tgMemory - reductionBytes) / 16, size_t(1) << 20)) & ~1;
    sliceWords = std::min(sliceWords, wordsPerRow);

    const size_t rowBytes = size_t(nRows) * size_t(wordsPerRow) * sizeof(uint64_t);
    const size_t needed = rowBytes + size_t(nPixels) * 4 + size_t(nRows + 1) * 8 + size_t(nDups) * 4;
    if ((rc = checkMemory(*dev, needed, "VGA visual local"))) {
        return rc;
    }

    auto *ctx = new dmx_vis_local();
    ctx->dev = dev;
    if (!rc) rc = makeBuffer(*dev, ctx->rows, rows, rowBytes, "VGA visual local");
    if (!rc) rc = makeBuffer(*dev, ctx->pixelToRow, pixelToRow, size_t(nPixels) * 4, "VGA visual local");
    if (!rc) rc = makeBuffer(*dev, ctx->dupOffsets, dupOffsets, size_t(nRows + 1) * 8, "VGA visual local");
    if (!rc) rc = makeBuffer(*dev, ctx->dupPixels, dupPixels, size_t(nDups) * 4, "VGA visual local");
    if (rc) {
        delete ctx;
        return rc;
    }
    ctx->nRows = nRows;
    ctx->wordsPerRow = wordsPerRow;
    ctx->nPixels = nPixels;
    ctx->hasDuplicates = nDups > 0;
    ctx->threads = threads;
    ctx->sliceWords = sliceWords;
    *out = ctx;
    return DMX_GPU_OK;
}

DMX_GPU_API int dmx_vis_local_run(dmx_vis_local *ctx, int32_t firstRow, int32_t nRows, int64_t *clusterOut,
                                  int32_t *unionOut) {
    clearError();
    if (!ctx || !clusterOut || !unionOut || firstRow < 0 || nRows <= 0 || firstRow + int64_t(nRows) > ctx->nRows) {
        return setError(DMX_GPU_ERR_INVALID_ARG, "dmx_vis_local_run: invalid arguments");
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    Device &dev = *ctx->dev;
    int rc = DMX_GPU_OK;
    if (nRows > ctx->batchCapacity) {
        for (Buffer **b : {&ctx->clusterAcc, &ctx->unionAcc}) {
            if (*b) {
                dev.release(*b);
                *b = nullptr;
            }
        }
        ctx->batchCapacity = 0;
        if (!rc) rc = makeBuffer(dev, ctx->clusterAcc, nullptr, size_t(nRows) * 8, "VGA visual local");
        if (!rc) rc = makeBuffer(dev, ctx->unionAcc, nullptr, size_t(nRows) * 4, "VGA visual local");
        if (rc) {
            return rc;
        }
        ctx->batchCapacity = nRows;
    }
    auto *cluster = static_cast<int64_t *>(dev.contents(ctx->clusterAcc));
    auto *unions = static_cast<int32_t *>(dev.contents(ctx->unionAcc));
    std::memset(cluster, 0, size_t(nRows) * 8);
    std::memset(unions, 0, size_t(nRows) * 4);

    std::string error;
    for (int chunk = 0; chunk < nRows; chunk += kRowsPerCommandBuffer) {
        const int rowsInChunk = std::min(kRowsPerCommandBuffer, nRows - chunk);
        if (!dev.begin(error)) {
            return setError(DMX_GPU_ERR_CUDA, "VGA visual local: " + error);
        }
        dev.setBuffer(ctx->rows, 0, 0);
        dev.setBuffer(ctx->pixelToRow, 0, 1);
        dev.setBuffer(ctx->dupOffsets, 0, 2);
        dev.setBuffer(ctx->dupPixels, 0, 3);
        dev.setBuffer(ctx->clusterAcc, size_t(chunk) * 8, 4);
        dev.setBuffer(ctx->unionAcc, size_t(chunk) * 4, 5);
        dev.setThreadgroupMemory(roundUp16(size_t(ctx->threads) * 8), 2);
        dev.setThreadgroupMemory(roundUp16(size_t(ctx->threads) * 4), 3);
        for (int sliceStart = 0; sliceStart < ctx->wordsPerRow; sliceStart += ctx->sliceWords) {
            DmxVisLocalParams p;
            p.wordsPerRow = ctx->wordsPerRow;
            p.firstRow = firstRow + chunk;
            p.sliceStart = sliceStart;
            p.sliceWords = std::min(ctx->sliceWords, ctx->wordsPerRow - sliceStart);
            p.addDuplicates = (ctx->hasDuplicates && sliceStart == 0) ? 1 : 0;
            dev.setBytes(&p, sizeof(p), 6);
            dev.setThreadgroupMemory(roundUp16(size_t(p.sliceWords) * 8), 0);
            dev.setThreadgroupMemory(roundUp16(size_t(p.sliceWords) * 8), 1);
            dev.dispatch(Kernel::VisLocal, uint32_t(rowsInChunk), 1, ctx->threads);
        }
        if (!dev.commit(false, error)) {
            return setError(DMX_GPU_ERR_CUDA, "VGA visual local: " + error);
        }
    }
    if (!dev.waitAll(error)) {
        return setError(DMX_GPU_ERR_CUDA, "VGA visual local: " + error);
    }
    std::memcpy(clusterOut, cluster, size_t(nRows) * 8);
    std::memcpy(unionOut, unions, size_t(nRows) * 4);
    return DMX_GPU_OK;
}

DMX_GPU_API void dmx_vis_local_destroy(dmx_vis_local *ctx) {
    std::lock_guard<std::mutex> lock(g_mutex);
    delete ctx;
}

// ---------------------------------------------------------------------------------
// VGA visual global measures

DMX_GPU_API int dmx_vis_global_create(dmx_vis_global **out, int32_t nNodes, int32_t nPositions, int32_t nLevels,
                                      const int64_t *keyOffsets, const int32_t *keyNodes,
                                      const int32_t *nodePositions, int32_t maxLanes) {
    clearError();
    if (!out || !keyOffsets || !nodePositions || nNodes <= 0 || nPositions <= 0 || nLevels <= 0 || nLevels > 31 ||
        maxLanes < 64) {
        return setError(DMX_GPU_ERR_INVALID_ARG, "dmx_vis_global_create: invalid arguments");
    }
    *out = nullptr;
    const uint64_t nKeys = uint64_t(DMX_METAL_ORIENTATIONS) * uint64_t(nLevels) * uint64_t(nPositions);
    const int64_t nEntries = keyOffsets[nKeys];
    if (nEntries > 0 && !keyNodes) {
        return setError(DMX_GPU_ERR_INVALID_ARG, "dmx_vis_global_create: key_nodes is null");
    }
    if (nKeys >= kGridLimit) {
        return setError(DMX_GPU_ERR_UNSUPPORTED, "VGA visual global: map too large for 32-bit Metal grids");
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    int rc = DMX_GPU_OK;
    Device *dev = device(rc);
    if (!dev) {
        return rc;
    }

    const uint64_t fixedBytes = (nKeys + 1) * 8 + uint64_t(nEntries) * 4 + uint64_t(DMX_METAL_ORIENTATIONS) * nNodes * 4;
    auto laneBytes = [&](int l) {
        const uint64_t words = uint64_t(l / 64);
        return nKeys * words * 8 + 2 * uint64_t(nNodes) * words * 8 + uint64_t(l) * 4;
    };
    auto fits = [&](int l) {
        const uint64_t words = uint64_t(l / 64);
        return fixedBytes + laneBytes(l) <= availableBytes(*dev) && nKeys * words < kGridLimit &&
               nKeys * words * 8 <= dev->maxBufferBytes() && uint64_t(nNodes) * words * 8 <= dev->maxBufferBytes();
    };
    // Largest lane count (multiple of 64, at most maxLanes) that fits.
    int lanes = (maxLanes / 64) * 64;
    while (lanes > 64 && !fits(lanes)) {
        lanes -= 64;
    }
    if ((rc = checkMemory(*dev, fixedBytes + laneBytes(lanes), "VGA visual global"))) {
        return rc;
    }
    if (nKeys * uint64_t(lanes / 64) >= kGridLimit) {
        return setError(DMX_GPU_ERR_UNSUPPORTED, "VGA visual global: map too large for 32-bit Metal grids");
    }

    auto *ctx = new dmx_vis_global();
    ctx->dev = dev;
    ctx->nNodes = nNodes;
    ctx->nPositions = nPositions;
    ctx->nLevels = nLevels;
    ctx->nKeys = nKeys;
    ctx->lanes = lanes;
    ctx->laneWords = lanes / 64;
    ctx->pullThreads = groupThreads(*dev, Kernel::VisGlobalPull);
    ctx->downsweepThreads = groupThreads(*dev, Kernel::VisGlobalDownsweep);
    ctx->combineThreads = groupThreads(*dev, Kernel::VisGlobalCombine);
    ctx->countThreads = std::min<uint32_t>(64, groupThreads(*dev, Kernel::VisGlobalCount)); // one per lane bit
    const size_t words = size_t(ctx->laneWords);
    const char *what = "VGA visual global";
    if (!rc) rc = makeBuffer(*dev, ctx->keyOffsets, keyOffsets, size_t(nKeys + 1) * 8, what);
    if (!rc) rc = makeBuffer(*dev, ctx->keyNodes, keyNodes, size_t(nEntries) * 4, what);
    if (!rc) rc = makeBuffer(*dev, ctx->nodePositions, nodePositions, size_t(DMX_METAL_ORIENTATIONS) * nNodes * 4, what);
    if (!rc) rc = makeBuffer(*dev, ctx->blocks, nullptr, size_t(nKeys) * words * 8, what);
    if (!rc) rc = makeBuffer(*dev, ctx->frontier, nullptr, size_t(nNodes) * words * 8, what);
    if (!rc) rc = makeBuffer(*dev, ctx->visited, nullptr, size_t(nNodes) * words * 8, what);
    if (!rc) rc = makeBuffer(*dev, ctx->levelCounts, nullptr, size_t(lanes) * 4, what);
    if (!rc) rc = makeBuffer(*dev, ctx->anyReached, nullptr, 4, what);
    if (rc) {
        delete ctx;
        return rc;
    }
    *out = ctx;
    return DMX_GPU_OK;
}

DMX_GPU_API int dmx_vis_global_lanes(const dmx_vis_global *ctx) { return ctx ? ctx->lanes : 0; }

DMX_GPU_API int dmx_vis_global_run(dmx_vis_global *ctx, int32_t firstSource, int32_t nSources, int32_t maxDepth,
                                   int32_t *levelCounts, int32_t levelCapacity, int32_t *nLevelsOut) {
    clearError();
    if (!ctx || !levelCounts || !nLevelsOut || firstSource < 0 || nSources <= 0 || nSources > ctx->lanes ||
        firstSource + int64_t(nSources) > ctx->nNodes || levelCapacity <= 0) {
        return setError(DMX_GPU_ERR_INVALID_ARG, "dmx_vis_global_run: invalid arguments");
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    Device &dev = *ctx->dev;
    const int lw = ctx->laneWords;
    const size_t nodeWords = size_t(ctx->nNodes) * size_t(lw);
    auto *frontier = static_cast<uint64_t *>(dev.contents(ctx->frontier));
    auto *visited = static_cast<uint64_t *>(dev.contents(ctx->visited));
    auto *counts = static_cast<int32_t *>(dev.contents(ctx->levelCounts));
    auto *anyReached = static_cast<int32_t *>(dev.contents(ctx->anyReached));

    // Seed: lane i starts at node firstSource + i (unified memory: written by the CPU).
    std::memset(frontier, 0, nodeWords * 8);
    std::memset(visited, 0, nodeWords * 8);
    for (int i = 0; i < nSources; ++i) {
        const size_t word = size_t(firstSource + i) * size_t(lw) + size_t(i / 64);
        frontier[word] |= uint64_t(1) << (i % 64);
        visited[word] |= uint64_t(1) << (i % 64);
    }
    // Level 0: every source reaches itself.
    for (int i = 0; i < nSources; ++i) {
        levelCounts[i] = 1;
    }

    DmxVisGlobalParams p;
    p.nNodes = ctx->nNodes;
    p.nLevels = ctx->nLevels;
    p.nPositions = ctx->nPositions;
    p.laneWords = lw;
    p.k = 0;
    p.nSources = nSources;
    p.nKeys = unsigned(ctx->nKeys);

    std::string error;
    int level = 0;
    while (maxDepth < 0 || level < maxDepth) {
        std::memset(counts, 0, size_t(nSources) * 4);
        *anyReached = 0;
        if (!dev.begin(error)) {
            return setError(DMX_GPU_ERR_CUDA, "VGA visual global: " + error);
        }
        dev.setBuffer(ctx->keyOffsets, 0, 0);
        dev.setBuffer(ctx->keyNodes, 0, 1);
        dev.setBuffer(ctx->frontier, 0, 2);
        dev.setBuffer(ctx->blocks, 0, 3);
        dev.setBytes(&p, sizeof(p), 4);
        dev.dispatch(Kernel::VisGlobalPull, groupsFor(ctx->nKeys * uint64_t(lw), ctx->pullThreads), 1,
                     ctx->pullThreads);
        for (int k = ctx->nLevels - 1; k >= 1; --k) {
            p.k = k;
            dev.setBuffer(ctx->blocks, 0, 0);
            dev.setBytes(&p, sizeof(p), 1);
            dev.dispatch(Kernel::VisGlobalDownsweep,
                         groupsFor(uint64_t(DMX_METAL_ORIENTATIONS) * uint64_t(ctx->nPositions) * uint64_t(lw),
                                   ctx->downsweepThreads),
                         1, ctx->downsweepThreads);
        }
        p.k = 0;
        dev.setBuffer(ctx->blocks, 0, 0);
        dev.setBuffer(ctx->nodePositions, 0, 1);
        dev.setBuffer(ctx->frontier, 0, 2);
        dev.setBuffer(ctx->visited, 0, 3);
        dev.setBuffer(ctx->anyReached, 0, 4);
        dev.setBytes(&p, sizeof(p), 5);
        dev.dispatch(Kernel::VisGlobalCombine, groupsFor(nodeWords, ctx->combineThreads), 1, ctx->combineThreads);
        dev.setBuffer(ctx->frontier, 0, 0);
        dev.setBuffer(ctx->levelCounts, 0, 1);
        dev.setBytes(&p, sizeof(p), 2);
        dev.dispatch(Kernel::VisGlobalCount, groupsFor(uint64_t(ctx->nNodes), DMX_METAL_COUNT_CHUNK), uint32_t(lw),
                     ctx->countThreads);
        if (!dev.commit(true, error)) {
            return setError(DMX_GPU_ERR_CUDA, "VGA visual global: " + error);
        }
        if (*anyReached == 0) {
            break;
        }
        ++level;
        if (level >= levelCapacity) {
            *nLevelsOut = level + 1;
            return setError(DMX_GPU_ERR_INVALID_ARG, "dmx_vis_global_run: level capacity exceeded");
        }
        std::memcpy(levelCounts + size_t(level) * size_t(nSources), counts, size_t(nSources) * 4);
    }
    *nLevelsOut = level + 1;
    return DMX_GPU_OK;
}

DMX_GPU_API void dmx_vis_global_destroy(dmx_vis_global *ctx) {
    std::lock_guard<std::mutex> lock(g_mutex);
    delete ctx;
}

} // extern "C"

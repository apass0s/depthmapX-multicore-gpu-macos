// depthmapX GPU module (Metal) - compute kernels for VGA visibility
//
// The same algorithms as the CUDA kernels (gpu/cuda/dmx_vis_local.cu,
// gpu/cuda/dmx_vis_global.cu); see those files and dmx_gpu_api.h for the data layout.
// Differences that come from Apple GPUs:
//  - threadgroup memory is at most 32 KiB (CUDA: 48 KiB), so the host sizes the row
//    slices of vis_local from the device limit;
//  - the threadgroup size is whatever the pipeline allows (it can be below 256), so no
//    kernel assumes one; reductions go through threadgroup memory, not 32-lane shuffles,
//    which Metal has for 32-bit types only;
//  - buffers are in unified memory: the host writes inputs and reads results directly,
//    and seeds / clears the search buffers itself between command buffers.
// All arithmetic is integer (64-bit words), so results do not depend on fast-math or
// on the order in which threads run.
//
// This file is compiled at run time by the Metal module (dmx_metal_device.mm) and, for
// tests on machines without Metal, as C++ by the CPU emulator (emu/), which defines the
// few Metal names used here (emu/metal_shim.h). Attributes are written DMX_ATTR(...) so
// that the C++ build can drop them.

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;
typedef ulong u64;
typedef long i64;
#define DMX_ATTR(...) [[__VA_ARGS__]]
// dmx_metal_shared.h is prepended to this source by the host.
#else
#include "emu/metal_shim.h"
#include "dmx_metal_shared.h"
#endif

// 64-bit popcount / count-trailing-zeros from the 32-bit built-ins, which every Metal
// version has for uint (the emulator runs this same code).
inline int popcount64(u64 x) { return int(popcount(uint(x))) + int(popcount(uint(x >> 32))); }
inline int ctz64(u64 x) { // x != 0
    const uint lo = uint(x);
    return lo != 0 ? int(ctz(lo)) : 32 + int(ctz(uint(x >> 32)));
}

// Writes 42: run by dmx_gpu_probe to check that the kernels compiled and execute.
kernel void dmx_probe(device int *out DMX_ATTR(buffer(0)), uint gid DMX_ATTR(thread_position_in_grid)) {
    if (gid == 0) {
        out[0] = 42;
    }
}

// ---------------------------------------------------------------------------------
// VGA visual local measures: one threadgroup per source row u (row p.firstRow + group).
// The group keeps a slice of row u and the running union of its neighbours' rows in
// threadgroup memory, walks the neighbours v of u and adds popcount(row v & row u) and
// OR-s row v into the union. Rows wider than a slice are processed by several
// dispatches (p.sliceStart), which add to clusterAcc / unionAcc.

kernel void dmx_vis_local_rows(const device u64 *rows DMX_ATTR(buffer(0)),
                          const device int *pixelToRow DMX_ATTR(buffer(1)),
                          const device i64 *dupOffsets DMX_ATTR(buffer(2)),
                          const device int *dupPixels DMX_ATTR(buffer(3)),
                          device i64 *clusterAcc DMX_ATTR(buffer(4)),
                          device int *unionAcc DMX_ATTR(buffer(5)),
                          constant DmxVisLocalParams &p DMX_ATTR(buffer(6)),
                          threadgroup u64 *sliceU DMX_ATTR(threadgroup(0)),
                          threadgroup u64 *sliceUnion DMX_ATTR(threadgroup(1)),
                          threadgroup i64 *redIntersect DMX_ATTR(threadgroup(2)),
                          threadgroup int *redUnion DMX_ATTR(threadgroup(3)),
                          uint group DMX_ATTR(threadgroup_position_in_grid),
                          uint tid DMX_ATTR(thread_index_in_threadgroup),
                          uint tgSize DMX_ATTR(threads_per_threadgroup)) {
    const int u = p.firstRow + int(group);
    const int step = int(tgSize);
    const device u64 *rowU = rows + u64(u) * u64(p.wordsPerRow);

    for (int t = int(tid); t < p.sliceWords; t += step) {
        sliceU[t] = rowU[p.sliceStart + t];
        sliceUnion[t] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Every thread walks the whole neighbour list (uniform control flow); thread tid owns
    // the slice words t = tid, tid + step, ..., so the union words have a single writer.
    i64 intersect = 0;
    for (int w = 0; w < p.wordsPerRow; ++w) {
        u64 bits = rowU[w];
        while (bits != 0) {
            const int bit = ctz64(bits);
            bits &= bits - 1;
            const int v = pixelToRow[w * 64 + bit];
            if (v < 0) {
                continue;
            }
            const device u64 *rowV = rows + u64(v) * u64(p.wordsPerRow) + u64(p.sliceStart);
            for (int t = int(tid); t < p.sliceWords; t += step) {
                const u64 x = rowV[t];
                intersect += i64(popcount64(x & sliceU[t]));
                sliceUnion[t] |= x;
            }
            if (p.addDuplicates != 0 && tid == 0) {
                for (i64 d = dupOffsets[v]; d < dupOffsets[v + 1]; ++d) {
                    const int px = dupPixels[d];
                    intersect += i64((rowU[px >> 6] >> u64(px & 63)) & u64(1));
                }
            }
        }
    }

    int unionCount = 0;
    for (int t = int(tid); t < p.sliceWords; t += step) {
        unionCount += popcount64(sliceUnion[t]);
    }
    redIntersect[tid] = intersect;
    redUnion[tid] = unionCount;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        i64 totalIntersect = 0;
        int totalUnion = 0;
        for (uint i = 0; i < tgSize; ++i) {
            totalIntersect += redIntersect[i];
            totalUnion += redUnion[i];
        }
        clusterAcc[group] += totalIntersect;
        unionAcc[group] += totalUnion;
    }
}

// ---------------------------------------------------------------------------------
// VGA visual global measures: one level of the multi-source search (lanes = sources).
// Per level the host encodes pull, the downsweeps for k = nLevels-1 .. 1, combine and
// count, in this order, in one serial compute encoder.

// blocks[key][w] = OR of the frontier words of the nodes that have a block at key.
kernel void dmx_vis_global_pull(const device i64 *keyOffsets DMX_ATTR(buffer(0)),
                                const device int *keyNodes DMX_ATTR(buffer(1)),
                                const device u64 *frontier DMX_ATTR(buffer(2)),
                                device u64 *blocks DMX_ATTR(buffer(3)),
                                constant DmxVisGlobalParams &p DMX_ATTR(buffer(4)),
                                uint t DMX_ATTR(thread_position_in_grid)) {
    const uint lw = uint(p.laneWords);
    if (t >= p.nKeys * lw) {
        return;
    }
    const uint key = t / lw;
    const uint w = t % lw;
    u64 acc = 0;
    for (i64 e = keyOffsets[key], end = keyOffsets[key + 1]; e < end; ++e) {
        acc |= frontier[u64(keyNodes[e]) * lw + w];
    }
    blocks[t] = acc;
}

// A block of size 2^k at position q covers the blocks of size 2^(k-1) at q and
// q + 2^(k-1); gathered: block[k-1][q] |= block[k][q] | block[k][q - 2^(k-1)].
// Reads level k and writes level k-1 only, so the threads of one dispatch never conflict.
kernel void dmx_vis_global_downsweep(device u64 *blocks DMX_ATTR(buffer(0)),
                                     constant DmxVisGlobalParams &p DMX_ATTR(buffer(1)),
                                     uint t DMX_ATTR(thread_position_in_grid)) {
    const uint lw = uint(p.laneWords);
    const uint perOrientation = uint(p.nPositions) * lw;
    if (t >= DMX_METAL_ORIENTATIONS * perOrientation) {
        return;
    }
    const uint o = t / perOrientation;
    const uint rem = t % perOrientation;
    const uint q = rem / lw;
    const uint w = rem % lw;
    const uint halfSize = 1u << uint(p.k - 1);
    const u64 levelK = (u64(o) * u64(p.nLevels) + u64(p.k)) * u64(p.nPositions);
    const u64 levelBelow = (u64(o) * u64(p.nLevels) + u64(p.k - 1)) * u64(p.nPositions);
    u64 v = blocks[(levelK + q) * lw + w];
    if (q >= halfSize) {
        v |= blocks[(levelK + q - halfSize) * lw + w];
    }
    if (v != 0) {
        blocks[(levelBelow + q) * lw + w] |= v;
    }
}

// reached = OR over orientations of the size-1 block at the node's position;
// next frontier = reached & ~visited.
kernel void dmx_vis_global_combine(const device u64 *blocks DMX_ATTR(buffer(0)),
                                   const device int *nodePositions DMX_ATTR(buffer(1)),
                                   device u64 *frontier DMX_ATTR(buffer(2)),
                                   device u64 *visited DMX_ATTR(buffer(3)),
                                   device atomic_int *anyReached DMX_ATTR(buffer(4)),
                                   constant DmxVisGlobalParams &p DMX_ATTR(buffer(5)),
                                   uint t DMX_ATTR(thread_position_in_grid)) {
    const uint lw = uint(p.laneWords);
    if (t >= uint(p.nNodes) * lw) {
        return;
    }
    const uint v = t / lw;
    const uint w = t % lw;
    u64 reached = 0;
    for (uint o = 0; o < DMX_METAL_ORIENTATIONS; ++o) {
        const u64 level0 = u64(o) * u64(p.nLevels) * u64(p.nPositions);
        reached |= blocks[(level0 + u64(nodePositions[u64(o) * u64(p.nNodes) + v])) * lw + w];
    }
    const u64 next = reached & ~visited[t];
    visited[t] |= next;
    frontier[t] = next;
    if (next != 0) {
        atomic_store_explicit(anyReached, 1, memory_order_relaxed);
    }
}

// Threadgroup (chunk, w) of up to 64 threads: thread tid counts the lanes w * 64 + b,
// b = tid, tid + tgSize, ... < 64, over the chunk's nodes and adds them to levelCounts.
kernel void dmx_vis_global_count(const device u64 *frontier DMX_ATTR(buffer(0)),
                                 device atomic_int *levelCounts DMX_ATTR(buffer(1)),
                                 constant DmxVisGlobalParams &p DMX_ATTR(buffer(2)),
                                 uint2 group DMX_ATTR(threadgroup_position_in_grid),
                                 uint tid DMX_ATTR(thread_index_in_threadgroup),
                                 uint tgSize DMX_ATTR(threads_per_threadgroup)) {
    const uint lw = uint(p.laneWords);
    const uint w = group.y;
    const int begin = int(group.x) * DMX_METAL_COUNT_CHUNK;
    const int end = min(p.nNodes, begin + DMX_METAL_COUNT_CHUNK);
    for (uint b = tid; b < 64; b += tgSize) {
        const int lane = int(w * 64 + b);
        int count = 0;
        for (int v = begin; v < end; ++v) {
            count += int((frontier[u64(v) * lw + w] >> u64(b)) & u64(1));
        }
        if (count != 0 && lane < p.nSources) {
            atomic_fetch_add_explicit(&levelCounts[lane], count, memory_order_relaxed);
        }
    }
}

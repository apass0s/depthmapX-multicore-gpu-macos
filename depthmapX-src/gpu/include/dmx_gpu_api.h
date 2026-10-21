// depthmapX GPU module - plain C interface of depthmapX_gpu.dll
//
// The DLL is built with nvcc + MSVC and loaded at run time by salalib (see
// gpu/host/gpuloader.cpp), so depthmapX runs unchanged where the DLL, the NVIDIA
// driver or a suitable GPU is missing. Only plain C types cross this boundary.
//
// All functions return DMX_GPU_OK (0) on success. On failure the message for the
// calling thread is available from dmx_gpu_last_error().

#pragma once

#include <stdint.h>

#ifdef _WIN32
#ifdef DMX_GPU_BUILDING_DLL
#define DMX_GPU_API __declspec(dllexport)
#else
#define DMX_GPU_API __declspec(dllimport)
#endif
#else
#define DMX_GPU_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Bump whenever a signature or a struct below changes.
#define DMX_GPU_API_VERSION 2

enum {
    DMX_GPU_OK = 0,
    DMX_GPU_ERR_NO_DEVICE = 1,     // no CUDA device, or driver too old for this DLL
    DMX_GPU_ERR_UNSUPPORTED = 2,   // device has no kernel image for its architecture
    DMX_GPU_ERR_OUT_OF_MEMORY = 3, // not enough device memory for the request
    DMX_GPU_ERR_INVALID_ARG = 4,
    DMX_GPU_ERR_CUDA = 5           // any other CUDA runtime error
};

typedef struct dmx_gpu_device_info {
    char name[256];
    int cc_major;
    int cc_minor;
    int sm_count;
    uint64_t total_mem;
    uint64_t free_mem;
    int runtime_version; // e.g. 13040 for CUDA 13.4
    int driver_version;
} dmx_gpu_device_info;

DMX_GPU_API int dmx_gpu_api_version(void);
DMX_GPU_API const char *dmx_gpu_last_error(void);

// Selects device 0, creates the context and checks that a kernel can run on it.
DMX_GPU_API int dmx_gpu_probe(dmx_gpu_device_info *info);

// ---------------------------------------------------------------------------------
// VGA visual local measures (clustering coefficient, control, controllability).
//
// Pixels are numbered 0..n_pixels-1 in PixelRef order (x, then y). Each graph node
// owns one bitset row of words_per_row 64-bit words: bit p is set when pixel p occurs
// in the node's visibility bins. pixel_to_row maps a pixel to its node row, or -1 when
// the pixel is not a filled point with a node. Pixels that occur more than once in a
// node's bins (upstream counts every occurrence) are listed, once per extra occurrence,
// in dup_pixels[dup_offsets[row] .. dup_offsets[row + 1]).
//
// For each requested row u, with N(u) the set bits of row u:
//   cluster[u] = sum over v in N(u) with pixel_to_row[v] >= 0 of
//                (occurrences in v's bins of pixels that are in N(u))
//   union[u]   = |union over the same v of the pixel sets of v|

typedef struct dmx_vis_local dmx_vis_local;

DMX_GPU_API int dmx_vis_local_create(dmx_vis_local **out, const uint64_t *rows, int32_t n_rows,
                                     int32_t words_per_row, const int32_t *pixel_to_row, int32_t n_pixels,
                                     const int64_t *dup_offsets, const int32_t *dup_pixels);

// Computes rows first_row .. first_row + n_rows - 1 (blocking).
DMX_GPU_API int dmx_vis_local_run(dmx_vis_local *ctx, int32_t first_row, int32_t n_rows, int64_t *cluster_out,
                                  int32_t *union_out);

DMX_GPU_API void dmx_vis_local_destroy(dmx_vis_local *ctx);

// ---------------------------------------------------------------------------------
// VGA visual global measures: breadth-first search from many sources at once
// (one bit lane per source).
//
// Graph nodes are numbered 0..n_nodes-1. Every grid cell has a position in each of 4
// line orientations (0 horizontal, 1 vertical, 2 x+1/y+1 diagonal, 3 x+1/y-1
// diagonal); positions along a line are consecutive. A node "sees" whole runs of
// positions (its visibility bins). Each run [a, b] of length len in orientation o is
// covered by two blocks of size 2^k, k = floor(log2(len)), starting at a and at
// b - 2^k + 1. key_nodes[key_offsets[key] .. key_offsets[key + 1]) lists the nodes that
// have a block at key = (o * n_levels + k) * n_positions + start.
// node_positions[o * n_nodes + v] is node v's position in orientation o.
//
// dmx_vis_global_run searches from sources first_source .. first_source + n_sources - 1
// (n_sources <= dmx_vis_global_lanes) and writes, for level L and source i,
// level_counts[L * n_sources + i] = number of nodes first reached at depth L
// (level 0 = the source). Searching stops after max_depth levels (radius), or -1 for
// no limit. *n_levels_out receives the number of levels written. If more than
// level_capacity levels are needed it returns DMX_GPU_ERR_INVALID_ARG and sets
// *n_levels_out to the capacity required so far (call again with more).

typedef struct dmx_vis_global dmx_vis_global;

DMX_GPU_API int dmx_vis_global_create(dmx_vis_global **out, int32_t n_nodes, int32_t n_positions, int32_t n_levels,
                                      const int64_t *key_offsets, const int32_t *key_nodes,
                                      const int32_t *node_positions, int32_t max_lanes);

DMX_GPU_API int dmx_vis_global_lanes(const dmx_vis_global *ctx);

DMX_GPU_API int dmx_vis_global_run(dmx_vis_global *ctx, int32_t first_source, int32_t n_sources, int32_t max_depth,
                                   int32_t *level_counts, int32_t level_capacity, int32_t *n_levels_out);

DMX_GPU_API void dmx_vis_global_destroy(dmx_vis_global *ctx);

#ifdef __cplusplus
}
#endif

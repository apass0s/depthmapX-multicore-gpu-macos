// depthmapX GPU module (Metal) - kernel parameter blocks, shared by the Metal Shading
// Language kernels (dmx_metal_kernels.metal) and the C++ host code.
//
// Only 32-bit fields: int is 32 bits in both languages, so the layouts are identical.
// The host passes each block with setBytes; see dmx_metal_api.cpp for their meaning.

#ifndef DMX_METAL_SHARED_H
#define DMX_METAL_SHARED_H

#define DMX_METAL_ORIENTATIONS 4
#define DMX_METAL_COUNT_CHUNK 256 // nodes per threadgroup in vis_global_count

struct DmxVisLocalParams {
    int wordsPerRow;
    int firstRow;
    int sliceStart;
    int sliceWords;
    int addDuplicates; // 1 on the first slice when the map has duplicate pixels
};

struct DmxVisGlobalParams {
    int nNodes;
    int nLevels;     // block sizes 2^0 .. 2^(nLevels-1)
    int nPositions;  // positions per orientation
    int laneWords;   // 64-bit words of lanes per node (lanes / 64)
    int k;           // vis_global_downsweep: block level being pushed down
    int nSources;    // vis_global_count: lanes in use
    unsigned int nKeys; // 4 * nLevels * nPositions; the host keeps nKeys * laneWords < 2^32,
                        // so every grid index fits the 32-bit thread position
};

#endif

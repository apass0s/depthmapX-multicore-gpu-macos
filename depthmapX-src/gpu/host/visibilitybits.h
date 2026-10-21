// depthmapX GPU module - visibility graph of a PointMap as bitset rows

#pragma once

#include "salalib/ngraph.h"
#include "salalib/pixelref.h"
#include "salalib/pointdata.h"

#include <cstdint>
#include <string>
#include <vector>

namespace dmxaccel {

// Calls f(pixel) for every pixel of a node's bins, in the order and with the
// repetitions of Node::first()/next(). Unlike those, it keeps its own cursor, so it is
// safe to call on the same node from several threads.
template <typename F> void forEachNodePixel(const Node &node, F &&f) {
    for (int b = 0; b < 32; ++b) {
        const Bin &bin = node.bin(b);
        for (const PixelVec &vec : bin.m_pixel_vecs) {
            PixelRef pix = vec.start();
            const short end = vec.end().col(bin.m_dir);
            do {
                f(pix);
                pix.move(bin.m_dir);
            } while (pix.col(bin.m_dir) <= end);
        }
    }
}

// Pixels are numbered in PixelRef order (x, then y), so iterating a row's set bits in
// increasing order visits pixels in the order std::sort gives PixelRefs.
struct VisibilityBits {
    int32_t nPixels = 0;
    int32_t wordsPerRow = 0;
    size_t gridRows = 0;

    std::vector<int32_t> gridToPixel;   // x * gridRows + y -> pixel, or -1
    std::vector<PixelRef> rowPixel;     // node row -> its PixelRef
    std::vector<int32_t> pixelToRow;    // pixel -> node row, or -1 if not a filled point with a node
    std::vector<uint64_t> rows;         // nRows() * wordsPerRow
    std::vector<int32_t> iterationSize; // per row: pixels visited by the node iterator, repeats included
    std::vector<int32_t> firstWord;     // per row: first non-zero word (wordsPerRow if empty)
    std::vector<int32_t> lastWord;      // per row: last non-zero word (-1 if empty)
    std::vector<int64_t> dupOffsets;    // CSR of repeated pixels per row (nRows() + 1 entries)
    std::vector<int32_t> dupPixels;

    int32_t nRows() const { return int32_t(rowPixel.size()); }
    const uint64_t *row(int32_t r) const { return rows.data() + size_t(r) * size_t(wordsPerRow); }
    bool test(int32_t r, int32_t pixel) const { return (row(r)[pixel >> 6] >> (pixel & 63)) & 1u; }
    int32_t pixelOf(PixelRef p) const { return gridToPixel[size_t(p.x) * gridRows + size_t(p.y)]; }

    static uint64_t estimateRowBytes(PointMap &map);

    // Builds the rows for every filled point that has a node. Returns false, with a
    // reason, if the map has something this representation does not support.
    bool build(PointMap &map, std::string &reason);
};

} // namespace dmxaccel

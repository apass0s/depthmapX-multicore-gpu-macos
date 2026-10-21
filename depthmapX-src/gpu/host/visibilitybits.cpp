// depthmapX GPU module - visibility graph of a PointMap as bitset rows

#include "gpu/host/visibilitybits.h"

#include <algorithm>
#include <atomic>
#include <memory>

namespace dmxaccel {

uint64_t VisibilityBits::estimateRowBytes(PointMap &map) {
    // Upper bound: every grid cell numbered, one row per filled point.
    const uint64_t gridCells = uint64_t(map.getRows()) * uint64_t(map.getCols());
    const uint64_t words = (gridCells + 63) / 64;
    return uint64_t(map.getFilledPointCount()) * words * sizeof(uint64_t);
}

bool VisibilityBits::build(PointMap &map, std::string &reason) {
    const size_t rowsN = map.getRows(), colsN = map.getCols();
    gridRows = rowsN;

    // Node rows, in the column-major order the VGA modules visit points.
    rowPixel.clear();
    for (size_t i = 0; i < colsN; i++) {
        for (size_t j = 0; j < rowsN; j++) {
            PixelRef p(static_cast<short>(i), static_cast<short>(j));
            Point &pt = map.getPoint(p);
            if (pt.filled()) {
                if (!pt.hasNode()) {
                    reason = "filled point without a visibility node (graph not made?)";
                    return false;
                }
                rowPixel.push_back(p);
            }
        }
    }
    const int32_t n = nRows();

    // Mark every pixel that is filled or seen by some node.
    const size_t gridCells = rowsN * colsN;
    std::unique_ptr<std::atomic<uint8_t>[]> seen(new std::atomic<uint8_t>[gridCells]);
    for (size_t k = 0; k < gridCells; k++) {
        seen[k].store(0, std::memory_order_relaxed);
    }
    std::atomic<bool> outOfGrid(false);
#pragma omp parallel for schedule(dynamic, 64)
    for (int32_t r = 0; r < n; r++) {
        PixelRef p = rowPixel[size_t(r)];
        seen[size_t(p.x) * rowsN + size_t(p.y)].store(1, std::memory_order_relaxed);
        forEachNodePixel(map.getPoint(p).getNode(), [&](PixelRef q) {
            if (q.x < 0 || q.y < 0 || size_t(q.x) >= colsN || size_t(q.y) >= rowsN) {
                outOfGrid.store(true, std::memory_order_relaxed);
                return;
            }
            seen[size_t(q.x) * rowsN + size_t(q.y)].store(1, std::memory_order_relaxed);
        });
    }
    if (outOfGrid.load()) {
        reason = "visibility bins reference pixels outside the grid";
        return false;
    }

    gridToPixel.assign(gridCells, -1);
    int32_t next = 0;
    for (size_t k = 0; k < gridCells; k++) { // k = x * rows + y: PixelRef order
        if (seen[k].load(std::memory_order_relaxed)) {
            gridToPixel[k] = next++;
        }
    }
    seen.reset();
    nPixels = next;
    wordsPerRow = (nPixels + 63) / 64;

    pixelToRow.assign(size_t(nPixels), -1);
    for (int32_t r = 0; r < n; r++) {
        pixelToRow[size_t(pixelOf(rowPixel[size_t(r)]))] = r;
    }

    rows.assign(size_t(n) * size_t(wordsPerRow), 0);
    iterationSize.assign(size_t(n), 0);
    firstWord.assign(size_t(n), wordsPerRow);
    lastWord.assign(size_t(n), -1);
    std::vector<std::vector<int32_t>> repeatedPixels(static_cast<size_t>(n));
#pragma omp parallel for schedule(dynamic, 64)
    for (int32_t r = 0; r < n; r++) {
        uint64_t *bits = rows.data() + size_t(r) * size_t(wordsPerRow);
        int32_t count = 0, lo = wordsPerRow, hi = -1;
        forEachNodePixel(map.getPoint(rowPixel[size_t(r)]).getNode(), [&](PixelRef q) {
            const int32_t px = pixelOf(q);
            const int32_t w = px >> 6;
            const uint64_t mask = uint64_t(1) << (px & 63);
            count++;
            if (bits[w] & mask) {
                repeatedPixels[size_t(r)].push_back(px);
            } else {
                bits[w] |= mask;
                lo = std::min(lo, w);
                hi = std::max(hi, w);
            }
        });
        iterationSize[size_t(r)] = count;
        firstWord[size_t(r)] = lo;
        lastWord[size_t(r)] = hi;
    }

    dupOffsets.assign(size_t(n) + 1, 0);
    for (int32_t r = 0; r < n; r++) {
        dupOffsets[size_t(r) + 1] = dupOffsets[size_t(r)] + int64_t(repeatedPixels[size_t(r)].size());
    }
    dupPixels.clear();
    dupPixels.reserve(size_t(dupOffsets.back()));
    for (auto &d : repeatedPixels) {
        dupPixels.insert(dupPixels.end(), d.begin(), d.end());
    }
    return true;
}

} // namespace dmxaccel

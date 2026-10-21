// depthmapX GPU module - VGA visual global measures on the GPU or on all CPU cores
//
// Upstream (salalib/vgamodules/vgavisualglobal.cpp) runs one breadth-first search per
// source point and derives every attribute from the number of points first reached
// at each depth. Here the searches run in parallel:
//  - CPU: the upstream search, unchanged, one source per OpenMP thread with
//    thread-local scratch matrices (handles every map);
//  - GPU: a multi-source search over visibility runs (gpu/cuda/dmx_vis_global.cu), for
//    maps without merge links or context-filled points.
// The depth distributions are integers, and the attributes are computed from them by
// the same code as upstream, so values are identical.

#include "gpu/host/vgavisualglobalaccel.h"

#include "gpu/host/gpuloader.h"
#include "gpu/host/visibilitybits.h"

#include "salalib/vgamodules/vgavisualglobal.h"

#include "genlib/pafmath.h"
#include "genlib/simplematrix.h"
#include "genlib/stringutils.h"

#include <algorithm>
#include <atomic>
#include <memory>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

constexpr int32_t kCpuSourcesPerBatch = 512;

struct DepthStats {
    int total_depth = 0;
    int total_nodes = 0;
    std::vector<int> distribution; // points first reached at each depth, depth 0 = the source
};

class Progress {
  public:
    Progress(Communicator *comm, int records) : m_comm(comm) {
        if (m_comm) {
            qtimer(m_time, 0);
            m_comm->CommPostMessage(Communicator::NUM_RECORDS, records);
        }
    }
    void update(int done) {
        if (m_comm && qtimer(m_time, 500)) {
            if (m_comm->IsCancelled()) {
                throw Communicator::CancelledException();
            }
            m_comm->CommPostMessage(Communicator::CURRENT_RECORD, done);
        }
    }

  private:
    Communicator *m_comm;
    time_t m_time = 0;
};

// ----------------------------------------------------------------------------------
// CPU: upstream's search for one source, reading the map only.

void extractUnseen(Node &node, PixelRefVector &pixels, depthmapX::RowMatrix<int> &miscs,
                   depthmapX::RowMatrix<PixelRef> &extents) {
    for (int i = 0; i < 32; i++) {
        Bin &bin = node.bin(i);
        for (auto pixVec : bin.m_pixel_vecs) {
            for (PixelRef pix = pixVec.start(); pix.col(bin.m_dir) <= pixVec.end().col(bin.m_dir);) {
                int &misc = miscs(pix.y, pix.x);
                PixelRef &extent = extents(pix.y, pix.x);
                if (misc == 0) {
                    pixels.push_back(pix);
                    misc |= (1 << i);
                }
                if (!(bin.m_dir & PixelRef::DIAGONAL)) {
                    if (extent.col(bin.m_dir) >= pixVec.end().col(bin.m_dir))
                        break;
                    extent.col(bin.m_dir) = pixVec.end().col(bin.m_dir);
                }
                pix.move(bin.m_dir);
            }
        }
    }
}

DepthStats searchFrom(PointMap &map, PixelRef curs, double radius, depthmapX::RowMatrix<int> &miscs,
                      depthmapX::RowMatrix<PixelRef> &extents) {
    for (size_t ii = 0; ii < map.getCols(); ii++) {
        for (size_t jj = 0; jj < map.getRows(); jj++) {
            miscs(jj, ii) = 0;
            extents(jj, ii) = PixelRef(ii, jj);
        }
    }
    DepthStats s;
    std::vector<PixelRefVector> search_tree;
    search_tree.push_back(PixelRefVector());
    search_tree.back().push_back(curs);

    int level = 0;
    while (search_tree[level].size()) {
        search_tree.push_back(PixelRefVector());
        const PixelRefVector &searchTreeAtLevel = search_tree[level];
        s.distribution.push_back(0);
        for (auto currLvlIter = searchTreeAtLevel.rbegin(); currLvlIter != searchTreeAtLevel.rend(); currLvlIter++) {
            int &pmisc = miscs(currLvlIter->y, currLvlIter->x);
            Point &p = map.getPoint(*currLvlIter);
            if (p.filled() && pmisc != ~0) {
                s.total_depth += level;
                s.total_nodes += 1;
                s.distribution.back() += 1;
                if ((int)radius == -1 || (level < (int)radius && (!p.contextfilled() || currLvlIter->iseven()))) {
                    extractUnseen(p.getNode(), search_tree[level + 1], miscs, extents);
                    pmisc = ~0;
                    if (!p.getMergePixel().empty()) {
                        PixelRef mergePixel = p.getMergePixel();
                        int &p2misc = miscs(mergePixel.y, mergePixel.x);
                        Point &p2 = map.getPoint(mergePixel);
                        if (p2misc != ~0) {
                            extractUnseen(p2.getNode(), search_tree[level + 1], miscs, extents);
                            p2misc = ~0;
                        }
                    }
                } else {
                    pmisc = ~0;
                }
            }
            search_tree[level].pop_back();
        }
        level++;
    }
    return s;
}

// The same search, faster but with identical results:
//  - upstream remembers how far each pixel has been scanned ("extent") only along
//    horizontal and vertical runs, sharing one PixelRef between them; diagonal runs are
//    always rescanned in full. Here each of the 4 orientations has its own extent, so
//    diagonal rescans stop early too. A scan only stops at a pixel whose remaining run
//    an earlier scan already marked, so the pixels found, and the order they are found
//    in, are unchanged;
//  - scratch cells are reset lazily with a generation stamp instead of clearing the
//    whole grid before every search.
// It does not reproduce upstream's left-over scratch state; searchFrom does that.
class FastSearch {
  public:
    FastSearch(PointMap &map) : m_map(map), m_rows(map.getRows()), m_cells(map.getRows() * map.getCols()) {}

    DepthStats run(PixelRef curs, double radius) {
        if (++m_stamp == 0) { // wrapped: invalidate every cell explicitly
            for (auto &c : m_cells) {
                c.stamp = 0;
            }
            m_stamp = 1;
        }
        DepthStats s;
        if (m_tree.empty()) {
            m_tree.emplace_back();
        }
        m_tree[0].clear();
        m_tree[0].push_back(curs);

        size_t level = 0;
        while (m_tree[level].size()) {
            if (m_tree.size() <= level + 1) {
                m_tree.emplace_back();
            }
            m_tree[level + 1].clear();
            PixelRefVector &here = m_tree[level];
            s.distribution.push_back(0);
            for (auto it = here.rbegin(); it != here.rend(); ++it) {
                int &pmisc = cell(*it).misc;
                Point &p = m_map.getPoint(*it);
                if (p.filled() && pmisc != ~0) {
                    s.total_depth += int(level);
                    s.total_nodes += 1;
                    s.distribution.back() += 1;
                    if ((int)radius == -1 || (int(level) < (int)radius && (!p.contextfilled() || it->iseven()))) {
                        extractUnseen(p.getNode(), m_tree[level + 1]);
                        pmisc = ~0;
                        if (!p.getMergePixel().empty()) {
                            PixelRef mergePixel = p.getMergePixel();
                            int &p2misc = cell(mergePixel).misc;
                            Point &p2 = m_map.getPoint(mergePixel);
                            if (p2misc != ~0) {
                                extractUnseen(p2.getNode(), m_tree[level + 1]);
                                p2misc = ~0;
                            }
                        }
                    } else {
                        pmisc = ~0;
                    }
                }
            }
            here.clear();
            level++;
        }
        return s;
    }

  private:
    struct Cell {
        uint32_t stamp;
        int misc;
        short extent[4]; // furthest col(dir) scanned through this pixel, per orientation
    };

    Cell &cell(PixelRef p) {
        Cell &c = m_cells[size_t(p.x) * m_rows + size_t(p.y)];
        if (c.stamp != m_stamp) {
            c.stamp = m_stamp;
            c.misc = 0;
            c.extent[0] = c.extent[2] = c.extent[3] = p.x; // col() of horizontal and diagonals is x
            c.extent[1] = p.y;                           // col() of vertical is y
        }
        return c;
    }

    static int orientation(char dir) {
        switch (dir) {
        case PixelRef::HORIZONTAL:
            return 0;
        case PixelRef::VERTICAL:
            return 1;
        case PixelRef::POSDIAGONAL:
            return 2;
        case PixelRef::NEGDIAGONAL:
            return 3;
        default:
            return -1;
        }
    }

    void extractUnseen(Node &node, PixelRefVector &pixels) {
        for (int i = 0; i < 32; i++) {
            Bin &bin = node.bin(i);
            const int o = orientation(bin.m_dir);
            for (auto pixVec : bin.m_pixel_vecs) {
                const short end = pixVec.end().col(bin.m_dir);
                for (PixelRef pix = pixVec.start(); pix.col(bin.m_dir) <= end;) {
                    Cell &c = cell(pix);
                    if (c.misc == 0) {
                        pixels.push_back(pix);
                        c.misc |= (1 << i);
                    }
                    if (o >= 0) {
                        if (c.extent[o] >= end)
                            break;
                        c.extent[o] = end;
                    }
                    pix.move(bin.m_dir);
                }
            }
        }
    }

    PointMap &m_map;
    size_t m_rows;
    std::vector<Cell> m_cells;
    uint32_t m_stamp = 0;
    std::vector<PixelRefVector> m_tree;
};

void searchAllCpu(PointMap &map, const std::vector<PixelRef> &sources, double radius, std::vector<DepthStats> &out,
                  Progress &progress) {
    const int32_t n = int32_t(sources.size());
    for (int32_t start = 0; start < n; start += kCpuSourcesPerBatch) {
        const int32_t end = std::min(n, start + kCpuSourcesPerBatch);
#pragma omp parallel
        {
            FastSearch search(map);
#pragma omp for schedule(dynamic, 1)
            for (int32_t s = start; s < end; s++) {
                out[size_t(s)] = search.run(sources[size_t(s)], radius);
            }
        }
        progress.update(end);
    }
}

// ----------------------------------------------------------------------------------
// Visibility runs as range-update keys (see dmx_gpu_api.h), for the multi-source search.

struct LineIndex {
    // position of grid cell (x, y) along each orientation; cells of a line are consecutive
    std::vector<int32_t> pos[4];
    std::vector<int32_t> line[4];
};

LineIndex makeLineIndex(int cols, int rows) {
    LineIndex li;
    const size_t cells = size_t(cols) * size_t(rows);
    auto at = [rows](int x, int y) { return size_t(x) * size_t(rows) + size_t(y); };
    for (int o = 0; o < 4; o++) {
        li.pos[o].assign(cells, -1);
        li.line[o].assign(cells, -1);
    }
    int32_t p = 0;
    for (int y = 0; y < rows; y++) // horizontal lines: fixed y, x increasing
        for (int x = 0; x < cols; x++, p++) {
            li.pos[0][at(x, y)] = p;
            li.line[0][at(x, y)] = y;
        }
    p = 0;
    for (int x = 0; x < cols; x++) // vertical lines: fixed x, y increasing
        for (int y = 0; y < rows; y++, p++) {
            li.pos[1][at(x, y)] = p;
            li.line[1][at(x, y)] = x;
        }
    p = 0;
    for (int d = -(rows - 1); d <= cols - 1; d++) // x+1/y+1 diagonals: fixed x - y, x increasing
        for (int x = std::max(0, d); x < cols && x - d < rows; x++, p++) {
            li.pos[2][at(x, x - d)] = p;
            li.line[2][at(x, x - d)] = d;
        }
    p = 0;
    for (int s = 0; s <= cols + rows - 2; s++) // x+1/y-1 diagonals: fixed x + y, x increasing
        for (int x = std::max(0, s - (rows - 1)); x < cols && x <= s; x++, p++) {
            li.pos[3][at(x, s - x)] = p;
            li.line[3][at(x, s - x)] = s;
        }
    return li;
}

int orientationOf(char dir) {
    switch (dir) {
    case PixelRef::HORIZONTAL:
        return 0;
    case PixelRef::VERTICAL:
        return 1;
    case PixelRef::POSDIAGONAL:
        return 2;
    case PixelRef::NEGDIAGONAL:
        return 3;
    default:
        return -1;
    }
}

int floorLog2(int32_t v) {
    int k = 0;
    while ((int64_t(2) << k) <= v) {
        k++;
    }
    return k;
}

struct RunGraph {
    int32_t nNodes = 0;
    int32_t nPositions = 0;
    int32_t nLevels = 0;
    std::vector<int64_t> keyOffsets;
    std::vector<int32_t> keyNodes;
    std::vector<int32_t> nodePositions; // [o * nNodes + v]
};

// Nodes are the filled points in VGA visiting order (node index = source index).
bool buildRunGraph(PointMap &map, const std::vector<PixelRef> &nodes, RunGraph &g, std::string &reason) {
    const int cols = int(map.getCols()), rows = int(map.getRows());
    const LineIndex li = makeLineIndex(cols, rows);
    auto at = [rows](PixelRef p) { return size_t(p.x) * size_t(rows) + size_t(p.y); };
    g.nNodes = int32_t(nodes.size());
    g.nPositions = int32_t(size_t(cols) * size_t(rows));

    struct Run {
        int32_t node, orientation, start, length;
    };
    std::vector<std::vector<Run>> runs(static_cast<size_t>(g.nNodes));
    std::atomic<bool> unsupported(false);
    int32_t maxLength = 1;
#pragma omp parallel
    {
        int32_t localMax = 1;
#pragma omp for schedule(dynamic, 64)
        for (int32_t v = 0; v < g.nNodes; v++) {
            Node &node = map.getPoint(nodes[size_t(v)]).getNode();
            for (int i = 0; i < 32; i++) {
                const Bin &bin = node.bin(i);
                if (bin.m_pixel_vecs.empty()) {
                    continue;
                }
                const int o = orientationOf(bin.m_dir);
                if (o < 0) {
                    unsupported.store(true);
                    continue;
                }
                for (const PixelVec &vec : bin.m_pixel_vecs) {
                    const PixelRef s = vec.start(), e = vec.end();
                    const int32_t length = int32_t(e.col(bin.m_dir)) - int32_t(s.col(bin.m_dir)) + 1;
                    if (length <= 0) {
                        continue; // upstream's loop visits nothing
                    }
                    if (s.x < 0 || s.y < 0 || s.x >= cols || s.y >= rows || e.x < 0 || e.y < 0 || e.x >= cols ||
                        e.y >= rows || li.line[o][at(s)] != li.line[o][at(e)] ||
                        li.pos[o][at(e)] - li.pos[o][at(s)] != length - 1) {
                        unsupported.store(true);
                        continue;
                    }
                    runs[size_t(v)].push_back(Run{v, o, li.pos[o][at(s)], length});
                    localMax = std::max(localMax, length);
                }
            }
        }
#pragma omp critical
        maxLength = std::max(maxLength, localMax);
    }
    if (unsupported.load()) {
        reason = "visibility bins with an unexpected layout";
        return false;
    }
    g.nLevels = floorLog2(maxLength) + 1;

    const size_t nKeys = size_t(4) * size_t(g.nLevels) * size_t(g.nPositions);
    auto keysOf = [&](const Run &r, size_t out[2]) {
        const int k = floorLog2(r.length);
        const size_t base = (size_t(r.orientation) * g.nLevels + size_t(k)) * size_t(g.nPositions);
        out[0] = base + size_t(r.start);
        out[1] = base + size_t(r.start + r.length - (int32_t(1) << k));
        return out[0] == out[1] ? 1 : 2;
    };
    g.keyOffsets.assign(nKeys + 1, 0);
    for (const auto &nodeRuns : runs) {
        for (const Run &r : nodeRuns) {
            size_t k[2];
            const int c = keysOf(r, k);
            for (int i = 0; i < c; i++) {
                g.keyOffsets[k[i] + 1]++;
            }
        }
    }
    for (size_t k = 0; k < nKeys; k++) {
        g.keyOffsets[k + 1] += g.keyOffsets[k];
    }
    g.keyNodes.assign(size_t(g.keyOffsets[nKeys]), 0);
    std::vector<int64_t> fill(g.keyOffsets.begin(), g.keyOffsets.end() - 1);
    for (const auto &nodeRuns : runs) {
        for (const Run &r : nodeRuns) {
            size_t k[2];
            const int c = keysOf(r, k);
            for (int i = 0; i < c; i++) {
                g.keyNodes[size_t(fill[k[i]]++)] = r.node;
            }
        }
    }

    g.nodePositions.assign(size_t(4) * size_t(g.nNodes), 0);
    for (int o = 0; o < 4; o++) {
        for (int32_t v = 0; v < g.nNodes; v++) {
            g.nodePositions[size_t(o) * size_t(g.nNodes) + size_t(v)] = li.pos[o][at(nodes[size_t(v)])];
        }
    }
    return true;
}

bool searchAllGpu(const RunGraph &g, double radius, std::vector<DepthStats> &out, Progress &progress,
                  std::string &reason) {
    auto &gpu = dmxaccel::GpuModule::instance();
    dmx_vis_global *ctx = nullptr;
    if (gpu.visGlobalCreate(&ctx, g.nNodes, g.nPositions, g.nLevels, g.keyOffsets.data(),
                            g.keyNodes.empty() ? nullptr : g.keyNodes.data(), g.nodePositions.data(), 1024) !=
        DMX_GPU_OK) {
        reason = gpu.lastError();
        return false;
    }
    std::unique_ptr<dmx_vis_global, decltype(gpu.visGlobalDestroy)> guard(ctx, gpu.visGlobalDestroy);
    const int32_t lanes = gpu.visGlobalLanes(ctx);
    const int32_t maxDepth = ((int)radius == -1) ? -1 : (int)radius;

    int32_t capacity = 64;
    std::vector<int32_t> counts;
    for (int32_t first = 0; first < g.nNodes; first += lanes) {
        const int32_t n = std::min(lanes, g.nNodes - first);
        int32_t levels = 0;
        for (;;) {
            counts.assign(size_t(capacity) * size_t(n), 0);
            const int rc = gpu.visGlobalRun(ctx, first, n, maxDepth, counts.data(), capacity, &levels);
            if (rc == DMX_GPU_OK) {
                break;
            }
            if (rc == DMX_GPU_ERR_INVALID_ARG && levels > capacity) {
                capacity = std::max(levels, capacity * 4);
                continue;
            }
            reason = gpu.lastError();
            return false;
        }
        for (int32_t i = 0; i < n; i++) {
            DepthStats &s = out[size_t(first + i)];
            for (int32_t level = 0; level < levels; level++) {
                const int c = counts[size_t(level) * size_t(n) + size_t(i)];
                s.distribution.push_back(c);
                s.total_depth += level * c;
                s.total_nodes += c;
            }
        }
        progress.update(first + n);
    }
    return true;
}

// The GPU algorithm (gpu/cuda/dmx_vis_global.cu) on CPU threads: 64 * laneWords
// searches at a time, each OpenMP loop playing the role of one kernel.
bool searchAllCpuRuns(const RunGraph &g, double radius, std::vector<DepthStats> &out, Progress &progress) {
    const size_t nKeys = size_t(4) * size_t(g.nLevels) * size_t(g.nPositions);
    const uint64_t budget = dmxaccel::physicalMemoryBytes() / 8;
    int laneWords = 16;
    while (laneWords > 1 && uint64_t(nKeys + 2 * size_t(g.nNodes)) * laneWords * 8 > budget) {
        laneWords /= 2;
    }
    if (budget != 0 && uint64_t(nKeys + 2 * size_t(g.nNodes)) * laneWords * 8 > budget) {
        return false;
    }
    const int32_t lanes = 64 * laneWords;
    const int32_t maxDepth = ((int)radius == -1) ? -1 : (int)radius;
    const size_t lw = size_t(laneWords);
    std::vector<uint64_t> blocks(nKeys * lw), frontier(size_t(g.nNodes) * lw), visited(size_t(g.nNodes) * lw);
    const int threads = dmxaccel::cpuThreadCount();
    std::vector<std::vector<int32_t>> threadCounts(static_cast<size_t>(threads), std::vector<int32_t>(static_cast<size_t>(lanes)));

    for (int32_t first = 0; first < g.nNodes; first += lanes) {
        const int32_t n = std::min(lanes, g.nNodes - first);
        std::fill(frontier.begin(), frontier.end(), 0);
        std::fill(visited.begin(), visited.end(), 0);
        for (int32_t lane = 0; lane < n; lane++) {
            const size_t word = size_t(first + lane) * lw + size_t(lane / 64);
            frontier[word] = visited[word] = uint64_t(1) << (lane % 64);
            out[size_t(first + lane)].distribution.push_back(1);
            out[size_t(first + lane)].total_nodes = 1;
        }
        for (int level = 1; maxDepth < 0 || level <= maxDepth; level++) {
            // pull
#pragma omp parallel for schedule(dynamic, 1024)
            for (int64_t key = 0; key < int64_t(nKeys); key++) {
                uint64_t *dst = blocks.data() + size_t(key) * lw;
                std::fill(dst, dst + lw, 0);
                for (int64_t e = g.keyOffsets[size_t(key)]; e < g.keyOffsets[size_t(key) + 1]; e++) {
                    const uint64_t *src = frontier.data() + size_t(g.keyNodes[size_t(e)]) * lw;
                    for (size_t w = 0; w < lw; w++) {
                        dst[w] |= src[w];
                    }
                }
            }
            // downsweep
            for (int k = g.nLevels - 1; k >= 1; k--) {
                const int64_t half = int64_t(1) << (k - 1);
#pragma omp parallel for schedule(static)
                for (int64_t t = 0; t < int64_t(4) * g.nPositions; t++) {
                    const int64_t o = t / g.nPositions, p = t % g.nPositions;
                    const size_t levelK = (size_t(o) * g.nLevels + size_t(k)) * size_t(g.nPositions);
                    const size_t below = (size_t(o) * g.nLevels + size_t(k - 1)) * size_t(g.nPositions);
                    const uint64_t *a = blocks.data() + (levelK + size_t(p)) * lw;
                    const uint64_t *b = p >= half ? blocks.data() + (levelK + size_t(p - half)) * lw : nullptr;
                    uint64_t *dst = blocks.data() + (below + size_t(p)) * lw;
                    for (size_t w = 0; w < lw; w++) {
                        dst[w] |= a[w] | (b ? b[w] : 0);
                    }
                }
            }
            // combine and count
            int anyReached = 0;
#pragma omp parallel reduction(| : anyReached)
            {
#ifdef _OPENMP
                std::vector<int32_t> &counts = threadCounts[size_t(omp_get_thread_num())];
#else
                std::vector<int32_t> &counts = threadCounts[0];
#endif
                std::fill(counts.begin(), counts.end(), 0);
#pragma omp for schedule(static)
                for (int32_t v = 0; v < g.nNodes; v++) {
                    for (size_t w = 0; w < lw; w++) {
                        uint64_t reached = 0;
                        for (int o = 0; o < 4; o++) {
                            const size_t level0 = size_t(o) * size_t(g.nLevels) * size_t(g.nPositions);
                            const size_t pos = size_t(g.nodePositions[size_t(o) * size_t(g.nNodes) + size_t(v)]);
                            reached |= blocks[(level0 + pos) * lw + w];
                        }
                        const size_t i = size_t(v) * lw + w;
                        const uint64_t next = reached & ~visited[i];
                        visited[i] |= next;
                        frontier[i] = next;
                        for (uint64_t bits = next; bits; bits &= bits - 1) {
                            counts[w * 64 + size_t(dmxaccel::lowestBit64(bits))]++;
                            anyReached = 1;
                        }
                    }
                }
            }
            if (!anyReached) {
                break;
            }
            for (int32_t lane = 0; lane < n; lane++) {
                int32_t c = 0;
                for (const auto &counts : threadCounts) {
                    c += counts[size_t(lane)];
                }
                DepthStats &s = out[size_t(first + lane)];
                s.distribution.push_back(c);
                s.total_depth += level * c;
                s.total_nodes += c;
            }
        }
        progress.update(first + n);
    }
    return true;
}

// Searches up to `radius` levels see exactly the first levels of a longer search (the
// level sets of an unweighted breadth-first search do not depend on where it stops), so
// a radius' statistics are a prefix of the longest search's level counts. Integers only.
DepthStats truncateToRadius(const DepthStats &full, double radius) {
    if ((int)radius == -1) {
        return full;
    }
    DepthStats s;
    const size_t levels = std::min(full.distribution.size(), size_t((int)radius) + 1);
    s.distribution.assign(full.distribution.begin(), full.distribution.begin() + std::ptrdiff_t(levels));
    for (size_t level = 0; level < levels; level++) {
        s.total_depth += int(level) * s.distribution[level];
        s.total_nodes += s.distribution[level];
    }
    return s;
}

// Column set-up, attribute values, left-over point state and displayed attribute for one
// radius, exactly as VGAVisualGlobal::run produces them.
void writeRadius(PointMap &map, double m_radius, const std::vector<PixelRef> &sources,
                 const std::vector<DepthStats> &stats, bool simple_version) {
    AttributeTable &attributes = map.getAttributeTable();

    int entropy_col = -1, rel_entropy_col = -1, integ_dv_col = -1, integ_pv_col = -1, integ_tk_col = -1,
        depth_col = -1, count_col = -1;
    std::string radius_text;
    if (m_radius != -1) {
        radius_text = std::string(" R") + dXstring::formatString(int(m_radius), "%d");
    }

    // Same column set-up as VGAVisualGlobal::run.
#ifndef _COMPILE_dX_SIMPLE_VERSION
    if (!simple_version) {
        std::string entropy_col_text = std::string("Visual Entropy") + radius_text;
        entropy_col = attributes.insertOrResetColumn(entropy_col_text.c_str());
    }
#endif

    std::string integ_dv_col_text = std::string("Visual Integration [HH]") + radius_text;
    integ_dv_col = attributes.insertOrResetColumn(integ_dv_col_text.c_str());

#ifndef _COMPILE_dX_SIMPLE_VERSION
    if (!simple_version) {
        std::string integ_pv_col_text = std::string("Visual Integration [P-value]") + radius_text;
        integ_pv_col = attributes.insertOrResetColumn(integ_pv_col_text.c_str());
        std::string integ_tk_col_text = std::string("Visual Integration [Tekl]") + radius_text;
        integ_tk_col = attributes.insertOrResetColumn(integ_tk_col_text.c_str());
        std::string depth_col_text = std::string("Visual Mean Depth") + radius_text;
        depth_col = attributes.insertOrResetColumn(depth_col_text.c_str());
        std::string count_col_text = std::string("Visual Node Count") + radius_text;
        count_col = attributes.insertOrResetColumn(count_col_text.c_str());
        std::string rel_entropy_col_text = std::string("Visual Relativised Entropy") + radius_text;
        rel_entropy_col = attributes.insertOrResetColumn(rel_entropy_col_text.c_str());
    }
#endif

    // Write results exactly as VGAVisualGlobal::run does.
    for (size_t s = 0; s < sources.size(); s++) {
        const int total_depth = stats[s].total_depth;
        const int total_nodes = stats[s].total_nodes;
        const std::vector<int> &distribution = stats[s].distribution;
        AttributeRow &row = attributes.getRow(AttributeKey(sources[s]));
        // only set to single float precision after divide
        // note -- total_nodes includes this one -- mean depth as per p.108 Social Logic of Space
        if (!simple_version) {
            row.setValue(count_col, float(total_nodes)); // note: total nodes includes this one
        }
        // ERROR !!!!!!
        if (total_nodes > 1) {
            double mean_depth = double(total_depth) / double(total_nodes - 1);
            if (!simple_version) {
                row.setValue(depth_col, float(mean_depth));
            }
            // total nodes > 2 to avoid divide by 0 (was > 3)
            if (total_nodes > 2 && mean_depth > 1.0) {
                double ra = 2.0 * (mean_depth - 1.0) / double(total_nodes - 2);
                // d-value / p-values from Depthmap 4 manual, note: node_count includes this one
                double rra_d = ra / dvalue(total_nodes);
                double rra_p = ra / pvalue(total_nodes);
                double integ_tk = teklinteg(total_nodes, total_depth);
                row.setValue(integ_dv_col, float(1.0 / rra_d));
                if (!simple_version) {
                    row.setValue(integ_pv_col, float(1.0 / rra_p));
                }
                if (total_depth - total_nodes + 1 > 1) {
                    if (!simple_version) {
                        row.setValue(integ_tk_col, float(integ_tk));
                    }
                } else {
                    if (!simple_version) {
                        row.setValue(integ_tk_col, -1.0f);
                    }
                }
            } else {
                row.setValue(integ_dv_col, (float)-1);
                if (!simple_version) {
                    row.setValue(integ_pv_col, (float)-1);
                    row.setValue(integ_tk_col, (float)-1);
                }
            }
            double entropy = 0.0, rel_entropy = 0.0, factorial = 1.0;
            // n.b., this distribution contains the root node itself in distribution[0]
            // -> chopped from entropy to avoid divide by zero if only one node
            for (size_t k = 1; k < distribution.size(); k++) {
                if (distribution[k] > 0) {
                    double prob = double(distribution[k]) / double(total_nodes - 1);
                    entropy -= prob * pafLog2(prob);
                    // Formula from Turner 2001, "Depthmap"
                    factorial *= double(k + 1);
                    double q = (pow(mean_depth, double(k)) / double(factorial)) * exp(-mean_depth);
                    rel_entropy += (float)prob * pafLog2(prob / q);
                }
            }
            if (!simple_version) {
                row.setValue(entropy_col, float(entropy));
                row.setValue(rel_entropy_col, float(rel_entropy));
            }
        } else {
            if (!simple_version) {
                row.setValue(depth_col, (float)-1);
                row.setValue(entropy_col, (float)-1);
                row.setValue(rel_entropy_col, (float)-1);
            }
        }
    }

    // Upstream leaves its scratch state of the last search in every point; reproduce it.
    depthmapX::RowMatrix<int> miscs(map.getRows(), map.getCols());
    depthmapX::RowMatrix<PixelRef> extents(map.getRows(), map.getCols());
    if (!sources.empty()) {
        searchFrom(map, sources.back(), m_radius, miscs, extents);
    }
    for (size_t i = 0; i < map.getCols(); i++) {
        for (size_t j = 0; j < map.getRows(); j++) {
            PixelRef curs = PixelRef(static_cast<short>(i), static_cast<short>(j));
            map.getPoint(curs).m_misc = miscs(j, i);
            map.getPoint(curs).m_extent = extents(j, i);
        }
    }
    map.setDisplayedAttribute(integ_dv_col);
}

std::string radiusList(const std::vector<double> &radii) {
    std::string out;
    for (double r : radii) {
        out += (out.empty() ? "" : ",") + ((int)r == -1 ? std::string("n") : std::to_string((int)r));
    }
    return out;
}

} // namespace

bool VGAVisualGlobalAccel::run(Communicator *comm, PointMap &map, bool simple_version) {
    using dmxaccel::ComputeMode;

    // Nothing is searched when only gates are analysed: upstream is already instant.
    if (m_gates_only) {
        bool ok = true;
        for (double radius : m_radii) {
            ok = VGAVisualGlobal(radius, m_gates_only).run(comm, map, simple_version) && ok;
        }
        return ok;
    }

    // Sources in upstream visiting order; skipped points keep their reset values.
    std::vector<PixelRef> filled, sources;
    bool gpuCompatible = true;
    for (size_t i = 0; i < map.getCols(); i++) {
        for (size_t j = 0; j < map.getRows(); j++) {
            PixelRef curs = PixelRef(static_cast<short>(i), static_cast<short>(j));
            Point &p = map.getPoint(curs);
            if (!p.filled()) {
                continue;
            }
            filled.push_back(curs);
            if (!p.hasNode() || p.contextfilled() || !p.getMergePixel().empty()) {
                gpuCompatible = false;
            }
            if (p.contextfilled() && !curs.iseven()) {
                continue;
            }
            sources.push_back(curs);
        }
    }

    // One search reaching the largest radius serves every radius (multi-source path).
    double deepest = 0.0;
    for (double radius : m_radii) {
        deepest = ((int)radius == -1 || (int)deepest == -1) ? -1.0 : std::max(deepest, radius);
    }
    const std::string radiiText = m_radii.size() > 1 ? ", radii " + radiusList(m_radii) : std::string();

    Progress progress(comm, map.getFilledPointCount());
    std::vector<DepthStats> stats;
    std::string path;
    bool done = false;

    // The multi-source search (GPU or CPU) needs a map without merge links or
    // context-filled points; then every filled point is a source, in the same order.
    RunGraph g;
    std::string runsReason;
    const bool haveRuns = gpuCompatible
                              ? buildRunGraph(map, filled, g, runsReason)
                              : (runsReason = "map has merge links, context-filled points or points without a node",
                                 false);

    if (m_settings.mode == ComputeMode::Gpu) {
        auto &gpu = dmxaccel::GpuModule::instance();
        std::string reason;
        if (!gpu.available()) {
            path = "GPU unavailable: " + gpu.unavailableReason();
        } else if (int(filled.size()) < m_settings.gpuMinNodes) {
            path = "map below the GPU threshold of " + std::to_string(m_settings.gpuMinNodes) + " points";
        } else if (!haveRuns) {
            path = runsReason;
        } else {
            stats.assign(filled.size(), DepthStats());
            if (searchAllGpu(g, deepest, stats, progress, reason)) {
                done = true;
                dmxaccel::addReport("VGA visual global: GPU (" + gpu.deviceDescription() + "), " +
                                    std::to_string(filled.size()) + " points" + radiiText);
            } else {
                path = "GPU failed: " + reason;
            }
        }
    }
    if (!done && haveRuns) {
        stats.assign(filled.size(), DepthStats());
        if (searchAllCpuRuns(g, deepest, stats, progress)) {
            done = true;
            dmxaccel::addReport("VGA visual global: CPU, " + std::to_string(dmxaccel::cpuThreadCount()) +
                                " threads, multi-source, " + std::to_string(filled.size()) + " points" +
                                radiiText + (path.empty() ? "" : " (" + path + ")"));
        }
    }

    if (done) {
        // without context-filled points, sources == filled
        for (double radius : m_radii) {
            std::vector<DepthStats> radiusStats(stats.size());
            for (size_t s = 0; s < stats.size(); s++) {
                radiusStats[s] = truncateToRadius(stats[s], radius);
            }
            writeRadius(map, radius, sources, radiusStats, simple_version);
        }
        return true;
    }

    // Upstream's search: its expansion rule depends on the radius (context-filled
    // points), so one search per radius.
    for (double radius : m_radii) {
        stats.assign(sources.size(), DepthStats());
        searchAllCpu(map, sources, radius, stats, progress);
        writeRadius(map, radius, sources, stats, simple_version);
    }
    dmxaccel::addReport("VGA visual global: CPU, " + std::to_string(dmxaccel::cpuThreadCount()) +
                        " threads, per source, " + std::to_string(filled.size()) + " points" + radiiText + " (" +
                        (path.empty() ? runsReason : path) + ")");
    return true;
}

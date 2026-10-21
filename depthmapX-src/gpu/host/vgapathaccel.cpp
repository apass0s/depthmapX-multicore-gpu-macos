// depthmapX GPU module - VGA metric and angular measures on all CPU cores
//
// Upstream (salalib/vgamodules/vgametric.cpp, vgaangular.cpp) runs, per source point,
// a lazy Dijkstra search over a std::set ordered by (key, pixel), keeping the search
// state (m_misc, m_dist, m_cumangle) in the shared Point objects, so it cannot run in
// parallel. Here the same search keeps its state in per-thread scratch cells and an
// indexed binary heap:
//  - the heap holds each pixel's smallest pending entry and pops entries in the same
//    (key, pixel) order; larger stale entries, which upstream pops and skips, are
//    dropped (see push() for the exact std::set insert semantics kept);
//  - every comparison and floating-point expression is upstream's, with the same types,
//    using the same dist()/angle() functions.
// Results are therefore identical. The left-over point state of the last search is
// copied back into the Points, as upstream leaves it.

#include "gpu/host/vgapathaccel.h"

#include "salalib/vgamodules/vgaangular.h"
#include "salalib/vgamodules/vgametric.h"

#include "genlib/stringutils.h"

#include <algorithm>
#include <cmath>

namespace {

constexpr int32_t kSourcesPerBatch = 256;

struct Progress {
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
    Communicator *m_comm;
    time_t m_time = 0;
};

struct Totals {
    float total_depth = 0.0f; // metric only
    float total_angle = 0.0f;
    float euclid_depth = 0.0f; // metric only
    int total_nodes = 0;
};

// Read-only facts about the map shared by all searches. Cells are numbered
// x * rows + y, which orders them like PixelRef's operator<.
struct PathMap {
    PointMap &map;
    size_t rows, cols;
    double spacing;

    enum : uint8_t { FILLED = 1, EXPANDABLE = 2, MERGED = 4 };
    std::vector<uint8_t> flags; // EXPANDABLE: blocked() || blockedAdjacent()
    std::vector<int32_t> mergeCell; // merge partner of MERGED cells

    // A visibility run of a node, flattened: the pixels upstream's loop
    // `for (pix = start; pix.col(dir) <= end.col(dir); pix.move(dir))` visits.
    struct Run {
        int32_t start;  // first cell
        int32_t stride; // cell step
        int32_t length; // pixels visited
        int16_t x, y;   // first pixel
        int16_t dx, dy; // pixel step
    };
    std::vector<int64_t> runOffsets; // per cell, into runs
    std::vector<Run> runs;           // in upstream's bin / vector order

    explicit PathMap(PointMap &m) : map(m), rows(m.getRows()), cols(m.getCols()), spacing(m.getSpacing()) {
        const size_t cells = rows * cols;
        flags.assign(cells, 0);
        mergeCell.assign(cells, -1);
        runOffsets.assign(cells + 1, 0);
        for (size_t c = 0; c < cells; c++) {
            const PixelRef p(static_cast<short>(c / rows), static_cast<short>(c % rows));
            Point &pt = map.getPoint(p);
            flags[c] = uint8_t((pt.filled() ? FILLED : 0) | ((pt.blocked() || map.blockedAdjacent(p)) ? EXPANDABLE : 0));
            if (pt.filled() && !pt.getMergePixel().empty()) {
                flags[c] |= MERGED;
                mergeCell[c] = int32_t(index(pt.getMergePixel()));
            }
            if (pt.filled() && pt.hasNode()) {
                Node &node = pt.getNode();
                for (int i = 0; i < 32; i++) {
                    const Bin &bin = node.bin(i);
                    for (const PixelVec &vec : bin.m_pixel_vecs) {
                        PixelRef next = vec.start();
                        next.move(bin.m_dir);
                        Run r;
                        r.x = vec.start().x;
                        r.y = vec.start().y;
                        r.dx = int16_t(next.x - r.x);
                        r.dy = int16_t(next.y - r.y);
                        r.start = int32_t(index(vec.start()));
                        r.stride = int32_t(r.dx) * int32_t(rows) + int32_t(r.dy);
                        r.length = 0;
                        for (PixelRef pix = vec.start(); pix.col(bin.m_dir) <= vec.end().col(bin.m_dir);
                             pix.move(bin.m_dir)) {
                            r.length++;
                        }
                        if (r.length > 0) {
                            runs.push_back(r);
                        }
                    }
                }
            }
            runOffsets[c + 1] = int64_t(runs.size());
        }
    }
    size_t index(PixelRef p) const { return size_t(p.x) * rows + size_t(p.y); }
    PixelRef pixel(int32_t cell) const {
        return PixelRef(static_cast<short>(size_t(cell) / rows), static_cast<short>(size_t(cell) % rows));
    }
};

template <bool Angular> class PathSearch {
  public:
    explicit PathSearch(const PathMap &pm)
        : m_pm(pm), m_cells(pm.rows * pm.cols), m_done((pm.rows * pm.cols + 63) / 64, 0) {}

    Totals run(PixelRef cursPixel, double radius) {
        newSearch();
        Totals t;
        const int32_t curs = int32_t(m_pm.index(cursPixel));
        push(0.0f, curs, -1);
        if (Angular) {
            cell(curs).cum = 0.0f;
        }
        while (!m_heap.empty()) {
            const Entry here = pop();
            if (Angular) {
                if (radius != -1.0 && here.key > radius) {
                    break;
                }
            } else {
                if (radius != -1.0 && (here.key * m_pm.spacing) > radius) {
                    break;
                }
            }
            const uint8_t f = m_pm.flags[size_t(here.cell)];
            if ((f & PathMap::FILLED) && !isDone(here.cell)) {
                extract(here);
                setDone(here.cell);
                if (f & PathMap::MERGED) {
                    const int32_t merge = m_pm.mergeCell[size_t(here.cell)];
                    if (!isDone(merge)) {
                        cell(merge).cum = cell(here.cell).cum;
                        extract(Entry{here.key, merge, -1});
                        setDone(merge);
                    }
                }
                if (Angular) {
                    t.total_angle += cell(here.cell).cum;
                } else {
                    t.total_depth += float(here.key * m_pm.spacing);
                    t.total_angle += cell(here.cell).cum;
                    t.euclid_depth += float(m_pm.spacing * dist(m_pm.pixel(here.cell), cursPixel));
                }
                t.total_nodes += 1;
            }
        }
        return t;
    }

    // Upstream's m_misc / m_dist / m_cumangle after the last search.
    void copyStateTo(PointMap &map) {
        for (size_t c = 0; c < m_cells.size(); c++) {
            const Cell &s = cell(int32_t(c));
            Point &pt = map.getPoint(m_pm.pixel(int32_t(c)));
            pt.m_misc = isDone(int32_t(c)) ? ~0 : 0;
            pt.m_dist = s.dist;
            pt.m_cumangle = s.cum;
        }
    }

  private:
    static constexpr int32_t kNotQueued = -1;

    struct Cell {
        uint32_t stamp;
        float dist;   // metric key (upstream m_dist)
        float cum;    // cumulative angle; angular key (upstream m_cumangle)
        int32_t slot; // position in m_heap, or kNotQueued
    };
    struct Entry {
        float key;
        int32_t cell;
        int32_t last; // -1 = NoPixel
    };

    static bool before(const Entry &a, const Entry &b) { // upstream's operator< on the triples
        return (a.key < b.key) || (a.key == b.key && a.cell < b.cell);
    }

    void newSearch() {
        m_heap.clear();
        std::fill(m_done.begin(), m_done.end(), 0);
        if (++m_stamp == 0) {
            for (auto &c : m_cells) {
                c.stamp = 0;
            }
            m_stamp = 1;
        }
    }

    // Upstream's m_misc == ~0 (processed), one bit per cell: small enough to stay in L1,
    // so the scan skips processed pixels without touching their cells. Kept apart from
    // the heap slot because a merge partner can be marked while it is still queued.
    bool isDone(int32_t index) const { return (m_done[size_t(index) >> 6] >> (index & 63)) & 1u; }
    void setDone(int32_t index) { m_done[size_t(index) >> 6] |= uint64_t(1) << (index & 63); }

    Cell &cell(int32_t index) {
        Cell &c = m_cells[size_t(index)];
        if (c.stamp != m_stamp) { // upstream's reset of every point before a search
            c.stamp = m_stamp;
            c.dist = Angular ? 0.0f : -1.0f;
            c.cum = Angular ? -1.0f : 0.0f;
            c.slot = kNotQueued;
        }
        return c;
    }

    // Node::extractMetric / extractAngular. Pixels that cannot pass upstream's strict
    // "<" improvement test are skipped before the costly sqrt/acos. Every filter below
    // only skips a pixel when the test provably fails, with margins far above the
    // rounding involved; anything else gets upstream's exact computation:
    //  - the added distance or angle is never negative, so a pixel whose key is already
    //    <= the current key always fails;
    //  - angular: acos(x) >= sqrt(2 (1 - x)), so if that lower bound already exceeds the
    //    room left (pixel key - current key) the test fails;
    //  - metric: if the squared distance clearly exceeds the squared room, so does the
    //    distance.
    void extract(const Entry &curs) {
        if (!(curs.key == 0.0f || (m_pm.flags[size_t(curs.cell)] & PathMap::EXPANDABLE))) {
            return;
        }
        const PixelRef cursPixel = m_pm.pixel(curs.cell);
        const PixelRef lastPixel = curs.last < 0 ? NoPixel : m_pm.pixel(curs.last);
        // upstream's angle(a, cursPixel, lastPixel) terms that do not depend on a
        const int bcx = cursPixel.x - lastPixel.x, bcy = cursPixel.y - lastPixel.y;
        const double lastLength = sqrt(sqr(bcx) + sqr(bcy));
        for (int64_t r = m_pm.runOffsets[size_t(curs.cell)]; r < m_pm.runOffsets[size_t(curs.cell) + 1]; r++) {
            const PathMap::Run &run = m_pm.runs[size_t(r)];
            int32_t index = run.start;
            PixelRef pix(run.x, run.y);
            for (int32_t k = 0; k < run.length; k++, index += run.stride, pix.x += run.dx, pix.y += run.dy) {
                if (isDone(index)) {
                    continue;
                }
                Cell &pt = cell(index);
                if (Angular) {
                    if (pt.cum != -1.0f) {
                        if (pt.cum <= curs.key) {
                            continue;
                        }
                        if (lastPixel != NoPixel) {
                            const int abx = pix.x - cursPixel.x, aby = pix.y - cursPixel.y;
                            const double x = double(abx * bcx + aby * bcy) /
                                             (sqrt(sqr(abx) + sqr(aby)) * lastLength + 1e-12);
                            const double room =
                                double(pt.cum) - double(curs.key) + 1e-6 * (std::fabs(double(pt.cum)) + 4.0);
                            if (sqrt(std::max(0.0, 2.0 * (1.0 - x))) / (M_PI * 0.5) >= room) {
                                continue;
                            }
                        }
                    }
                    float ang = (lastPixel == NoPixel) ? 0.0f
                                                       : (float)(angle(pix, cursPixel, lastPixel) / (M_PI * 0.5));
                    if (pt.cum == -1.0 || curs.key + ang < pt.cum) {
                        pt.cum = cell(curs.cell).cum + ang;
                        push(pt.cum, index, curs.cell);
                    }
                } else {
                    if (pt.dist != -1.0f) {
                        if (pt.dist <= curs.key) {
                            continue;
                        }
                        const double room = (double(pt.dist) - double(curs.key)) * (1.0 + 1e-9);
                        if (sqr(pix.x - cursPixel.x) + sqr(pix.y - cursPixel.y) > room * room) {
                            continue;
                        }
                    }
                    if (pt.dist == -1.0 || (curs.key + dist(pix, cursPixel) < pt.dist)) {
                        pt.dist = curs.key + (float)dist(pix, cursPixel);
                        pt.cum = cell(curs.cell).cum +
                                 (lastPixel == NoPixel ? 0.0f
                                                       : (float)(angle(pix, cursPixel, lastPixel) / (M_PI * 0.5)));
                        push(pt.dist, index, curs.cell);
                    }
                }
            }
        }
    }

    // std::set::insert of (key, pixel, last). The heap keeps only the pixel's smallest
    // pending entry, which is the only one upstream ever acts on:
    //  - smaller key: replaces the entry (the old one would be popped later and skipped);
    //  - equal key: std::set keeps the existing element, including its old `last`. This
    //    happens in metric analysis, which tests the improvement in double but stores
    //    the key as float;
    //  - larger key: the existing smaller entry pops first, so nothing changes.
    void push(float key, int32_t index, int32_t last) {
        Cell &c = cell(index);
        if (c.slot == kNotQueued) {
            c.slot = int32_t(m_heap.size());
            m_heap.push_back(Entry{key, index, last});
        } else if (key < m_heap[size_t(c.slot)].key) {
            m_heap[size_t(c.slot)] = Entry{key, index, last};
        } else {
            return;
        }
        siftUp(size_t(c.slot));
    }

    Entry pop() {
        const Entry top = m_heap.front();
        m_cells[size_t(top.cell)].slot = kNotQueued;
        const Entry tail = m_heap.back();
        m_heap.pop_back();
        if (!m_heap.empty()) {
            m_heap[0] = tail;
            siftDown(0);
        }
        return top;
    }

    // Heap entries' cells are always current (they were pushed in this search), so the
    // slot updates below index m_cells directly.
    void siftUp(size_t i) {
        const Entry e = m_heap[i];
        while (i > 0) {
            const size_t parent = (i - 1) / 2;
            if (!before(e, m_heap[parent])) {
                break;
            }
            m_heap[i] = m_heap[parent];
            m_cells[size_t(m_heap[i].cell)].slot = int32_t(i);
            i = parent;
        }
        m_heap[i] = e;
        m_cells[size_t(e.cell)].slot = int32_t(i);
    }

    void siftDown(size_t i) {
        const Entry e = m_heap[i];
        const size_t n = m_heap.size();
        for (;;) {
            size_t child = 2 * i + 1;
            if (child >= n) {
                break;
            }
            if (child + 1 < n && before(m_heap[child + 1], m_heap[child])) {
                child++;
            }
            if (!before(m_heap[child], e)) {
                break;
            }
            m_heap[i] = m_heap[child];
            m_cells[size_t(m_heap[i].cell)].slot = int32_t(i);
            i = child;
        }
        m_heap[i] = e;
        m_cells[size_t(e.cell)].slot = int32_t(i);
    }

    const PathMap &m_pm;
    std::vector<Cell> m_cells;
    std::vector<uint64_t> m_done;
    uint32_t m_stamp = 0;
    std::vector<Entry> m_heap;
};

template <bool Angular>
std::vector<Totals> searchAllCpu(const PathMap &pm, const std::vector<PixelRef> &sources, double radius,
                                 Progress &progress) {
    std::vector<Totals> out(sources.size());
    const int32_t n = int32_t(sources.size());
    for (int32_t start = 0; start < n; start += kSourcesPerBatch) {
        const int32_t end = std::min(n, start + kSourcesPerBatch);
#pragma omp parallel
        {
            PathSearch<Angular> search(pm);
#pragma omp for schedule(dynamic, 1)
            for (int32_t s = start; s < end; s++) {
                out[size_t(s)] = search.run(sources[size_t(s)], radius);
            }
        }
        progress.update(end);
    }
    return out;
}

std::vector<PixelRef> filledPoints(PointMap &map) {
    std::vector<PixelRef> out;
    for (size_t i = 0; i < map.getCols(); i++) {
        for (size_t j = 0; j < map.getRows(); j++) {
            PixelRef curs = PixelRef(static_cast<short>(i), static_cast<short>(j));
            if (map.getPoint(curs).filled()) {
                out.push_back(curs);
            }
        }
    }
    return out;
}

} // namespace

bool VGAMetricAccel::run(Communicator *comm, PointMap &map, bool simple_version) {
    if (m_gates_only) { // upstream does no search at all
        return VGAMetric(m_radius, m_gates_only).run(comm, map, simple_version);
    }

    // Same column set-up as VGAMetric::run.
    std::string radius_text;
    if (m_radius != -1.0) {
        if (m_radius > 100.0) {
            radius_text = std::string(" R") + dXstring::formatString(m_radius, "%.f");
        } else if (map.getRegion().width() < 1.0) {
            radius_text = std::string(" R") + dXstring::formatString(m_radius, "%.4f");
        } else {
            radius_text = std::string(" R") + dXstring::formatString(m_radius, "%.2f");
        }
    }
    AttributeTable &attributes = map.getAttributeTable();

    // n.b. these must be entered in alphabetical order to preserve col indexing:
    std::string penn_col_text = std::string("Metric Mean Penn Distance") + radius_text;
    int penn_col = attributes.insertOrResetColumn(penn_col_text.c_str());
    std::string mspa_col_text = std::string("Metric Mean Shortest-Path Angle") + radius_text;
    int mspa_col = attributes.insertOrResetColumn(mspa_col_text.c_str());
    std::string mspl_col_text = std::string("Metric Mean Shortest-Path Distance") + radius_text;
    int mspl_col = attributes.insertOrResetColumn(mspl_col_text.c_str());
    std::string dist_col_text = std::string("Metric Mean Straight-Line Distance") + radius_text;
    int dist_col = attributes.insertOrResetColumn(dist_col_text.c_str());
    std::string count_col_text = std::string("Metric Node Count") + radius_text;
    int count_col = attributes.insertOrResetColumn(count_col_text.c_str());

    const std::vector<PixelRef> sources = filledPoints(map);
    const PathMap pm(map);
    Progress progress(comm, map.getFilledPointCount());

    const std::vector<Totals> totals = searchAllCpu<false>(pm, sources, m_radius, progress);
    dmxaccel::addReport("VGA metric: CPU, " + std::to_string(dmxaccel::cpuThreadCount()) + " threads, " +
                        std::to_string(sources.size()) + " points");

    // Write results exactly as VGAMetric::run does.
    for (size_t s = 0; s < sources.size(); s++) {
        const float total_depth = totals[s].total_depth, total_angle = totals[s].total_angle,
                    euclid_depth = totals[s].euclid_depth;
        const int total_nodes = totals[s].total_nodes;
        AttributeRow &row = attributes.getRow(AttributeKey(sources[s]));
        row.setValue(mspa_col, float(double(total_angle) / double(total_nodes)));
        row.setValue(mspl_col, float(double(total_depth) / double(total_nodes)));
        row.setValue(dist_col, float(double(euclid_depth) / double(total_nodes)));
        // This is the legacy Mean Penn Distance formula. The root contributes zero to both
        // totals and is included in total_nodes for compatibility with historic output.
        row.setValue(penn_col,
                     float(std::max(0.0, double(total_depth - euclid_depth)) / double(total_nodes)));
        row.setValue(count_col, float(total_nodes));
    }

    if (!sources.empty()) {
        PathSearch<false> last(pm);
        last.run(sources.back(), m_radius);
        last.copyStateTo(map);
    }
    map.overrideDisplayedAttribute(-2);
    map.setDisplayedAttribute(mspl_col);
    return true;
}

bool VGAAngularAccel::run(Communicator *comm, PointMap &map, bool simple_version) {
    if (m_gates_only) { // upstream does no search at all
        return VGAAngular(m_radius, m_gates_only).run(comm, map, simple_version);
    }

    // Same column set-up as VGAAngular::run.
    std::string radius_text;
    if (m_radius != -1.0) {
        if (map.getRegion().width() > 100.0) {
            radius_text = std::string(" R") + dXstring::formatString(m_radius, "%.f");
        } else if (map.getRegion().width() < 1.0) {
            radius_text = std::string(" R") + dXstring::formatString(m_radius, "%.4f");
        } else {
            radius_text = std::string(" R") + dXstring::formatString(m_radius, "%.2f");
        }
    }

    AttributeTable &attributes = map.getAttributeTable();

    std::string mean_depth_col_text = std::string("Angular Mean Depth") + radius_text;
    int mean_depth_col = attributes.getOrInsertColumn(mean_depth_col_text.c_str());
    std::string total_detph_col_text = std::string("Angular Total Depth") + radius_text;
    int total_depth_col = attributes.getOrInsertColumn(total_detph_col_text.c_str());
    std::string count_col_text = std::string("Angular Node Count") + radius_text;
    int count_col = attributes.getOrInsertColumn(count_col_text.c_str());

    total_depth_col = attributes.getOrInsertColumn(total_detph_col_text.c_str());

    const std::vector<PixelRef> sources = filledPoints(map);
    const PathMap pm(map);
    Progress progress(comm, map.getFilledPointCount());

    const std::vector<Totals> totals = searchAllCpu<true>(pm, sources, m_radius, progress);
    dmxaccel::addReport("VGA angular: CPU, " + std::to_string(dmxaccel::cpuThreadCount()) + " threads, " +
                        std::to_string(sources.size()) + " points");

    // Write results exactly as VGAAngular::run does.
    for (size_t s = 0; s < sources.size(); s++) {
        const float total_angle = totals[s].total_angle;
        const int total_nodes = totals[s].total_nodes;
        AttributeRow &row = map.getAttributeTable().getRow(AttributeKey(sources[s]));
        if (total_nodes > 0) {
            row.setValue(mean_depth_col, float(double(total_angle) / double(total_nodes)));
        }
        row.setValue(total_depth_col, total_angle);
        row.setValue(count_col, float(total_nodes));
    }

    if (!sources.empty()) {
        PathSearch<true> last(pm);
        last.run(sources.back(), m_radius);
        last.copyStateTo(map);
    }
    map.setDisplayedAttribute(-2);
    map.setDisplayedAttribute(mean_depth_col);
    return true;
}

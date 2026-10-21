// depthmapX GPU module - segment angular analysis on all CPU cores
//
// Upstream (salalib/segmmodules/segmangular.cpp) runs one search per segment over a
// std::vector kept sorted with insert_sorted (upper_bound) and popped from the front.
// All SegmentData in it compare equal (metricdepth is always 0), so the vector is
// ordered by the float angle and, among equal angles, by insertion order. A binary
// heap on (angle, insertion number) pops the same entries in the same order, so every
// sum is accumulated identically; the searches then run in parallel, each with its own
// state. Results are identical to upstream.

#include "gpu/host/segmangularaccel.h"

#include "genlib/stringutils.h"

#include <algorithm>
#include <queue>

namespace {

constexpr int kSourcesPerBatch = 256;

struct Totals {
    std::vector<double> total_depth; // per radius bin
    std::vector<int> node_count;
};

class AngularSearch {
  public:
    AngularSearch(ShapeGraph &map, const std::vector<double> &radii)
        : m_connections(map.getConnections()), m_radii(radii), m_covered(map.getShapeCount(), 0) {}

    Totals run(int source) {
        if (++m_stamp == 0) {
            std::fill(m_covered.begin(), m_covered.end(), 0);
            m_stamp = 1;
        }
        Totals t;
        t.total_depth.assign(m_radii.size(), 0.0);
        t.node_count.assign(m_radii.size(), 0);
        uint64_t seq = 0;
        m_queue = Queue();
        m_queue.push(Entry{0.0f, seq++, source, 0, 0});
        // node_count includes this one, but will be added in next algo:
        while (!m_queue.empty()) {
            const Entry here = m_queue.top();
            m_queue.pop();
            if (covered(here.ref)) {
                continue;
            }
            m_covered[size_t(here.ref)] = m_stamp;
            double depth_to_line = here.angle;
            t.total_depth[here.coverage] += depth_to_line;
            t.node_count[here.coverage] += 1;
            const Connector &line = m_connections[size_t(here.ref)];
            if (here.dir != -1) {
                expand(line.m_forward_segconns, depth_to_line, here.coverage, seq);
            }
            if (here.dir != 1) {
                expand(line.m_back_segconns, depth_to_line, here.coverage, seq);
            }
        }
        return t;
    }

  private:
    struct Entry {
        float angle;
        uint64_t seq; // insertion order: upstream's upper_bound insert keeps equal angles FIFO
        int ref;
        char dir;
        unsigned int coverage;
    };
    struct Later {
        bool operator()(const Entry &a, const Entry &b) const {
            return a.angle > b.angle || (a.angle == b.angle && a.seq > b.seq);
        }
    };
    using Queue = std::priority_queue<Entry, std::vector<Entry>, Later>;

    bool covered(int ref) const { return m_covered[size_t(ref)] == m_stamp; }

    void expand(const std::map<SegmentRef, float> &segconns, double depth_to_line, unsigned int coverage,
                uint64_t &seq) {
        for (auto &segconn : segconns) {
            if (!covered(segconn.first.ref)) {
                double angle = depth_to_line + segconn.second;
                size_t rbin = coverage;
                while (rbin != m_radii.size() && m_radii[rbin] != -1 && angle > m_radii[rbin]) {
                    rbin++;
                }
                if (rbin != m_radii.size()) {
                    m_queue.push(Entry{float(angle), seq++, segconn.first.ref, segconn.first.dir, (unsigned int)rbin});
                }
            }
        }
    }

    const std::vector<Connector> &m_connections;
    const std::vector<double> &m_radii;
    std::vector<uint32_t> m_covered;
    uint32_t m_stamp = 0;
    Queue m_queue;
};

} // namespace

bool SegmentAngularAccel::run(Communicator *comm, ShapeGraph &map, bool) {

    if (map.getMapType() != ShapeMap::SEGMENTMAP) {
        return false;
    }

    AttributeTable &attributes = map.getAttributeTable();

    time_t atime = 0;
    if (comm) {
        qtimer(atime, 0);
        comm->CommPostMessage(Communicator::NUM_RECORDS, map.getConnections().size());
    }

    // Same radius ordering and column set-up as SegmentAngular::run.
    bool radius_n = false;
    std::vector<double> radii;
    for (double radius : m_radius_set) {
        if (radius < 0) {
            radius_n = true;
        } else {
            radii.push_back(radius);
        }
    }
    if (radius_n) {
        radii.push_back(-1.0);
    }

    std::vector<int> depth_col, count_col, total_col;
    for (int radius : radii) {
        std::string radius_text = makeRadiusText(Options::RADIUS_ANGULAR, radius);
        std::string depth_col_text = std::string("Angular Mean Depth") + radius_text;
        attributes.insertOrResetColumn(depth_col_text.c_str());
        std::string count_col_text = std::string("Angular Node Count") + radius_text;
        attributes.insertOrResetColumn(count_col_text.c_str());
        std::string total_col_text = std::string("Angular Total Depth") + radius_text;
        attributes.insertOrResetColumn(total_col_text.c_str());
    }

    for (int radius : radii) {
        std::string radius_text = makeRadiusText(Options::RADIUS_ANGULAR, radius);
        std::string depth_col_text = std::string("Angular Mean Depth") + radius_text;
        depth_col.push_back(attributes.getColumnIndex(depth_col_text.c_str()));
        std::string count_col_text = std::string("Angular Node Count") + radius_text;
        count_col.push_back(attributes.getColumnIndex(count_col_text.c_str()));
        std::string total_col_text = std::string("Angular Total Depth") + radius_text;
        total_col.push_back(attributes.getColumnIndex(total_col_text.c_str()));
    }

    // Upstream searches from segment i for the i-th row of the attribute table.
    std::vector<AttributeRow *> rows;
    for (auto &iter : attributes) {
        rows.push_back(&iter.getRow());
    }
    const int n = int(rows.size());
    std::vector<Totals> totals(rows.size());
    for (int start = 0; start < n; start += kSourcesPerBatch) {
        const int end = std::min(n, start + kSourcesPerBatch);
#pragma omp parallel
        {
            AngularSearch search(map, radii);
#pragma omp for schedule(dynamic, 1)
            for (int i = start; i < end; i++) {
                totals[size_t(i)] = search.run(i);
            }
        }
        if (comm) {
            if (qtimer(atime, 500)) {
                if (comm->IsCancelled()) {
                    throw Communicator::CancelledException();
                }
                comm->CommPostMessage(Communicator::CURRENT_RECORD, end);
            }
        }
    }
    dmxaccel::addReport("Segment angular: CPU, " + std::to_string(dmxaccel::cpuThreadCount()) + " threads, " +
                        std::to_string(n) + " segments");

    // Write results exactly as SegmentAngular::run does.
    for (int i = 0; i < n; i++) {
        AttributeRow &row = *rows[size_t(i)];
        const std::vector<double> &total_depth = totals[size_t(i)].total_depth;
        const std::vector<int> &node_count = totals[size_t(i)].node_count;
        int curs_node_count = 0;
        double curs_total_depth = 0.0;
        for (size_t r = 0; r < radii.size(); r++) {
            curs_node_count += node_count[r];
            curs_total_depth += total_depth[r];
            row.setValue(count_col[r], float(curs_node_count));
            if (curs_node_count > 1) {
                // note -- node_count includes this one -- mean depth as per p.108 Social Logic of Space
                double mean_depth = curs_total_depth / double(curs_node_count - 1);
                row.setValue(depth_col[r], float(mean_depth));
                row.setValue(total_col[r], float(curs_total_depth));
            } else {
                row.setValue(depth_col[r], -1);
                row.setValue(total_col[r], -1);
            }
        }
    }

    map.setDisplayedAttribute(-2); // <- override if it's already showing
    map.setDisplayedAttribute(depth_col.back());

    return true;
}

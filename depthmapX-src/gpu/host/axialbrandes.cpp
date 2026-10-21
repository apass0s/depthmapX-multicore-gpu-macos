// depthmapX GPU module - exact axial choice (Brandes betweenness) on all CPU cores
//
// Brandes (2001), unweighted: for every source s a breadth-first search counts shortest
// paths (sigma), then lines are taken in reverse search order and each passes
// delta[v] += sigma[v] / sigma[w] * (1 + delta[w]) to its predecessors v; delta[w] is
// added to w's betweenness. Matches the rapido build's "Choice [Brandes]" value for value
// on the example axial map (ordered pairs, end points excluded).

#include "gpu/host/axialbrandes.h"
#include "gpu/host/shaperows.h"

#include "genlib/stringutils.h"

#include <algorithm>

namespace {

struct Contribution {
    std::vector<std::pair<int, double>> delta; // (line, delta), lines other than the source
};

class BrandesSearch {
  public:
    BrandesSearch(const std::vector<Connector> &connections)
        : m_connections(connections), m_dist(connections.size(), -1), m_sigma(connections.size(), 0.0),
          m_delta(connections.size(), 0.0) {}

    void run(int s, int radius, Contribution &out) {
        // reset only what the previous search touched
        for (int v : m_order) {
            m_dist[size_t(v)] = -1;
            m_sigma[size_t(v)] = 0.0;
            m_delta[size_t(v)] = 0.0;
        }
        m_order.clear();
        m_order.push_back(s);
        m_dist[size_t(s)] = 0;
        m_sigma[size_t(s)] = 1.0;
        for (size_t qi = 0; qi < m_order.size(); qi++) {
            const int v = m_order[qi];
            if (radius != -1 && m_dist[size_t(v)] >= radius) {
                continue;
            }
            for (int w : m_connections[size_t(v)].m_connections) {
                if (m_dist[size_t(w)] < 0) {
                    m_dist[size_t(w)] = m_dist[size_t(v)] + 1;
                    m_order.push_back(w);
                }
                if (m_dist[size_t(w)] == m_dist[size_t(v)] + 1) {
                    m_sigma[size_t(w)] += m_sigma[size_t(v)];
                }
            }
        }
        out.delta.clear();
        for (size_t k = m_order.size(); k-- > 0;) {
            const int w = m_order[k];
            // predecessors of w: neighbours one level closer to s (in connection order)
            if (w != s) {
                for (int v : m_connections[size_t(w)].m_connections) {
                    if (m_dist[size_t(v)] >= 0 && m_dist[size_t(v)] + 1 == m_dist[size_t(w)] &&
                        !(radius != -1 && m_dist[size_t(v)] >= radius)) {
                        m_delta[size_t(v)] += m_sigma[size_t(v)] / m_sigma[size_t(w)] * (1.0 + m_delta[size_t(w)]);
                    }
                }
                out.delta.emplace_back(w, m_delta[size_t(w)]);
            }
        }
    }

  private:
    const std::vector<Connector> &m_connections;
    std::vector<int> m_dist;
    std::vector<double> m_sigma, m_delta;
    std::vector<int> m_order;
};

} // namespace

bool AxialBrandesChoice::run(Communicator *comm, ShapeGraph &map, bool) {
    AttributeTable &attributes = map.getAttributeTable();
    const std::vector<Connector> &connections = map.getConnections();
    const int n = int(connections.size());
    const std::vector<AttributeRow *> shapeRows = dmxaccel::rowsByShapeIndex(map);

    // radii as AxialIntegration orders them: finite ones ascending, then n
    bool radius_n = false;
    std::vector<int> radii;
    for (double radius : m_radius_set) {
        if (radius < 0) {
            radius_n = true;
        } else {
            radii.push_back(static_cast<int>(radius));
        }
    }
    if (radius_n) {
        radii.push_back(-1);
    }

    time_t atime = 0;
    if (comm) {
        qtimer(atime, 0);
        comm->CommPostMessage(Communicator::NUM_RECORDS, n * int(radii.size()));
    }

    const int threads = dmxaccel::cpuThreadCount();
    const int batchSize = 16 * threads;
    std::vector<Contribution> results(static_cast<size_t>(batchSize));
    int done = 0;
    for (int radius : radii) {
        std::string radius_text;
        if (radius != -1) {
            radius_text = dXstring::formatString(radius, " R%d");
        }
        const int col = int(attributes.insertOrResetColumn(std::string("Choice [Brandes]") + radius_text));

        std::vector<double> betweenness(size_t(n), 0.0);
        for (int start = 0; start < n; start += batchSize) {
            const int count = std::min(n, start + batchSize) - start;
#pragma omp parallel
            {
                BrandesSearch search(connections);
#pragma omp for schedule(dynamic, 1)
                for (int k = 0; k < count; k++) {
                    search.run(start + k, radius, results[size_t(k)]);
                }
            }
            for (int k = 0; k < count; k++) { // in source order: independent of the thread count
                for (const auto &d : results[size_t(k)].delta) {
                    betweenness[size_t(d.first)] += d.second;
                }
            }
            done += count;
            if (comm) {
                if (qtimer(atime, 500)) {
                    if (comm->IsCancelled()) {
                        throw Communicator::CancelledException();
                    }
                    comm->CommPostMessage(Communicator::CURRENT_RECORD, done);
                }
            }
        }
        for (int v = 0; v < n; v++) {
            shapeRows[size_t(v)]->setValue(size_t(col), float(betweenness[size_t(v)]));
        }
    }
    dmxaccel::addReport("Axial Brandes choice: CPU, " + std::to_string(threads) + " threads, " + std::to_string(n) +
                        " lines");
    return true;
}

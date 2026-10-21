// depthmapX GPU module - segment topological / metric analysis on all CPU cores
//
// Upstream (salalib/segmmodules/segmtopological.cpp, segmmetric.cpp; the two differ in
// a few lines, marked below) runs one bucketed search per segment. The search state is
// per source (its audit trail is not reset between sources, but entries from earlier
// searches are never read), so each thread keeps its own. Choice is shared:
//  - choice adds 1 per path node: integer-valued, so per-thread sums are exact;
//  - weighted choice adds root length * length: order-dependent rounding, so each
//    source logs its additions and they are replayed in upstream's source order.
// Results are therefore identical to upstream.

#include "gpu/host/segmtopometaccel.h"
#include "gpu/host/shaperows.h"

#include "salalib/segmmodules/segmhelpers.h"

#include "genlib/stringutils.h"

#include <algorithm>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

struct SourceResult {
    double total = 0.0, wtotal = 0.0, wtotaldepth = 0.0, totalsegdepth = 0.0, totalmetdepth = 0.0;
    double rootseglength = 0.0;
    // Weighted choice additions in upstream order. Each walk back to the root adds the
    // same value to every segment on it, so a walk is stored as ~index of its value in
    // walk_values, followed by the segments it adds to.
    std::vector<int32_t> walk_log;
    std::vector<double> walk_values;
};

struct Setup {
    const std::vector<Connector> &connections;
    size_t n;
    bool metric;
    double radius;
    bool sel_only;
    std::vector<int> axialrefs;
    std::vector<float> seglengths;
    float maxseglength;
};

class TopoMetSearch {
  public:
    explicit TopoMetSearch(const Setup &s)
        : m_s(s), m_seen(s.n), m_audittrail(s.n), m_choice(s.sel_only ? 0 : s.n, 0.0), m_list(512) {}

    std::vector<double> &choice() { return m_choice; }

    void run(size_t cursor, SourceResult &out) {
        const Setup &s = m_s;
        const int maxbin = s.metric ? 512 : 2;
        for (size_t i = 0; i < s.n; i++) {
            m_seen[i] = 0xffffffff;
        }
        for (auto &l : m_list) {
            l.clear();
        }
        std::vector<TopoMetSegmentRef> &audittrail = m_audittrail;
        int bin = 0;
        m_list[size_t(bin)].push_back(int(cursor));
        double rootseglength = s.seglengths[cursor];
        audittrail[cursor] = TopoMetSegmentRef(int(cursor), Connector::SEG_CONN_ALL, rootseglength * 0.5, -1);
        int open = 1;
        unsigned int segdepth = 0;
        double total = 0.0, wtotal = 0.0, wtotaldepth = 0.0, totalsegdepth = 0.0, totalmetdepth = 0.0;
        out.walk_log.clear();
        out.walk_values.clear();
        while (open != 0) {
            while (m_list[size_t(bin)].size() == 0) {
                bin++;
                segdepth += 1;
                if (bin == maxbin) {
                    bin = 0;
                }
            }
            //
            TopoMetSegmentRef &here = audittrail[size_t(m_list[size_t(bin)].back())];
            m_list[size_t(bin)].pop_back();
            open--;
            //
            if (here.done) {
                continue;
            } else {
                here.done = true;
            }
            //
            double len = s.seglengths[size_t(here.ref)];
            totalsegdepth += segdepth;
            totalmetdepth += here.dist - len * 0.5; // preloaded with length ahead
            wtotal += len;
            if (s.metric) {
                wtotaldepth += len * (here.dist - len * 0.5);
            } else {
                wtotaldepth += len * segdepth;
            }
            total += 1;
            //
            const Connector &axline = s.connections.at(size_t(here.ref));
            int connected_cursor = -2;
            auto iter = axline.m_back_segconns.begin();
            bool backsegs = true;
            while (connected_cursor != -1) {
                if (backsegs && iter == axline.m_back_segconns.end()) {
                    iter = axline.m_forward_segconns.begin();
                    backsegs = false;
                }
                if (!backsegs && iter == axline.m_forward_segconns.end()) {
                    break;
                }
                connected_cursor = iter->first.ref;
                if (m_seen[size_t(connected_cursor)] > segdepth && static_cast<size_t>(connected_cursor) != cursor) {
                    bool seenalready = (m_seen[size_t(connected_cursor)] == 0xffffffff) ? false : true;
                    float length = s.seglengths[size_t(connected_cursor)];
                    int axialref = s.axialrefs[size_t(connected_cursor)];
                    audittrail[size_t(connected_cursor)] =
                        TopoMetSegmentRef(connected_cursor, here.dir, here.dist + length, here.ref);
                    m_seen[size_t(connected_cursor)] = segdepth;
                    if (s.radius == -1 || here.dist + length < s.radius) {
                        // puts in a suitable bin ahead of us...
                        open++;
                        //
                        if (s.metric) {
                            // better to divide by 511 but have 512 bins...
                            m_list[size_t((bin + int(floor(0.5 + 511 * length / s.maxseglength))) % 512)].push_back(
                                connected_cursor);
                        } else if (s.axialrefs[size_t(here.ref)] == axialref) {
                            m_list[size_t(bin)].push_back(connected_cursor);
                        } else {
                            m_list[size_t((bin + 1) % 2)].push_back(connected_cursor);
                            m_seen[size_t(connected_cursor)] = segdepth + 1;
                        }
                    }
                    if (!s.sel_only && connected_cursor > int(cursor) &&
                        !seenalready) { // only one way paths, saves doing this twice
                        int subcur = connected_cursor;
                        out.walk_log.push_back(~int32_t(out.walk_values.size()));
                        out.walk_values.push_back(rootseglength * length);
                        while (subcur != -1) {
                            // in this method of choice, start and end lines are included
                            m_choice[size_t(subcur)] += 1;
                            out.walk_log.push_back(subcur);
                            subcur = audittrail[size_t(subcur)].previous;
                        }
                    }
                }
                iter++;
            }
        }
        out.total = total;
        out.wtotal = wtotal;
        out.wtotaldepth = wtotaldepth;
        out.totalsegdepth = totalsegdepth;
        out.totalmetdepth = totalmetdepth;
        out.rootseglength = rootseglength;
    }

  private:
    const Setup &m_s;
    std::vector<unsigned int> m_seen;
    std::vector<TopoMetSegmentRef> m_audittrail;
    std::vector<double> m_choice; // this thread's choice sums (integer-valued)
    std::vector<std::vector<int>> m_list;
};

} // namespace

bool SegmentTopoMetAccel::run(Communicator *comm, ShapeGraph &map, bool) {
    AttributeTable &attributes = map.getAttributeTable();

    bool retvar = true;

    time_t atime = 0;
    if (comm) {
        qtimer(atime, 0);
        comm->CommPostMessage(Communicator::NUM_RECORDS,
                              (m_sel_only ? map.getSelSet().size() : map.getConnections().size()));
    }

    // record axial line refs for topological analysis
    std::vector<int> axialrefs;
    // quick through to find the longest seg length
    std::vector<float> seglengths;
    float maxseglength = 0.0f;
    const std::vector<AttributeRow *> shapeRows = dmxaccel::rowsByShapeIndex(map);
    for (size_t cursor = 0; cursor < map.getShapeCount(); cursor++) {
        AttributeRow &row = *shapeRows[cursor];
        axialrefs.push_back(row.getValue(attributes.getColumnIndex("Axial Line Ref")));
        seglengths.push_back(row.getValue(attributes.getColumnIndex("Segment Length")));
        if (seglengths.back() > maxseglength) {
            maxseglength = seglengths.back();
        }
    }

    // Same column set-up as SegmentTopological::run / SegmentMetric::run.
    std::string prefix, suffix;
    prefix = m_metric ? "Metric " : "Topological ";
    if (m_radius != -1.0) {
        suffix = dXstring::formatString(m_radius, " R%.f metric");
    }
    std::string choicecol = prefix + "Choice" + suffix;
    std::string wchoicecol = prefix + "Choice [SLW]" + suffix;
    std::string meandepthcol = prefix + "Mean Depth" + suffix;
    std::string wmeandepthcol = prefix + std::string("Mean Depth [SLW]") + suffix;
    std::string totaldcol = prefix + "Total Depth" + suffix;
    std::string totalcol = prefix + "Total Nodes" + suffix;
    std::string wtotalcol = prefix + "Total Length" + suffix;
    //
    if (!m_sel_only) {
        attributes.insertOrResetColumn(choicecol.c_str());
        attributes.insertOrResetColumn(wchoicecol.c_str());
    }
    attributes.insertOrResetColumn(meandepthcol.c_str());
    attributes.insertOrResetColumn(wmeandepthcol.c_str());
    attributes.insertOrResetColumn(totaldcol.c_str());
    attributes.insertOrResetColumn(totalcol.c_str());
    attributes.insertOrResetColumn(wtotalcol.c_str());

    const size_t n = map.getShapeCount();
    const int threads = dmxaccel::cpuThreadCount();
    Setup setup{map.getConnections(), n, m_metric, m_radius, m_sel_only, axialrefs, seglengths, maxseglength};

    std::vector<size_t> sources;
    for (size_t cursor = 0; cursor < n; cursor++) {
        if (m_sel_only && !shapeRows[cursor]->isSelected()) {
            continue;
        }
        sources.push_back(cursor);
    }

    std::vector<double> wchoice(m_sel_only ? 0 : n, 0.0);
    std::vector<std::vector<double>> threadChoice(static_cast<size_t>(threads));
    // weighted choice logs are kept until replayed, so batches stay small
    const int batchSize = 4 * threads;
    std::vector<SourceResult> results(static_cast<size_t>(batchSize));

    for (size_t start = 0; start < sources.size(); start += size_t(batchSize)) {
        const int count = int(std::min(sources.size() - start, size_t(batchSize)));
#pragma omp parallel
        {
#ifdef _OPENMP
            const int tid = omp_get_thread_num();
#else
            const int tid = 0;
#endif
            TopoMetSearch search(setup);
            if (!threadChoice[size_t(tid)].empty()) {
                search.choice().swap(threadChoice[size_t(tid)]); // continue this thread's sums
            }
#pragma omp for schedule(dynamic, 1)
            for (int i = 0; i < count; i++) {
                search.run(sources[start + size_t(i)], results[size_t(i)]);
            }
            threadChoice[size_t(tid)].swap(search.choice());
        }
        // Replay weighted choice in upstream's order of sources and additions.
        if (!m_sel_only) {
            for (int i = 0; i < count; i++) {
                const SourceResult &res = results[size_t(i)];
                double value = 0.0;
                for (int32_t entry : res.walk_log) {
                    if (entry < 0) {
                        value = res.walk_values[size_t(~entry)];
                    } else {
                        wchoice[size_t(entry)] += value;
                    }
                }
            }
        }
        for (int i = 0; i < count; i++) {
            const SourceResult &res = results[size_t(i)];
            AttributeRow &row = *shapeRows[sources[start + size_t(i)]];
            // also put in mean depth:
            if (m_metric) {
                row.setValue(meandepthcol.c_str(), res.totalmetdepth / (res.total - 1));
                row.setValue(totaldcol.c_str(), res.totalmetdepth);
            } else {
                row.setValue(meandepthcol.c_str(), res.totalsegdepth / (res.total - 1));
                row.setValue(totaldcol.c_str(), res.totalsegdepth);
            }
            row.setValue(wmeandepthcol.c_str(), res.wtotaldepth / (res.wtotal - res.rootseglength));
            row.setValue(totalcol.c_str(), res.total);
            row.setValue(wtotalcol.c_str(), res.wtotal);
        }
        if (comm) {
            if (qtimer(atime, 500)) {
                if (comm->IsCancelled()) {
                    throw Communicator::CancelledException();
                }
            }
            comm->CommPostMessage(Communicator::CURRENT_RECORD, int(start) + count);
        }
    }
    dmxaccel::addReport(std::string("Segment ") + (m_metric ? "metric" : "topological") + ": CPU, " +
                        std::to_string(threads) + " threads, " + std::to_string(n) + " segments");

    if (!m_sel_only) {
        std::vector<double> choice(n, 0.0); // integer-valued sums: exact in any order
        for (const auto &c : threadChoice) {
            for (size_t i = 0; i < c.size(); i++) {
                choice[i] += c[i];
            }
        }
        // note, I've stopped sel only from calculating choice values:
        for (size_t cursor = 0; cursor < n; cursor++) {
            AttributeRow &row = *shapeRows[cursor];
            row.setValue(choicecol.c_str(), choice[cursor]);
            row.setValue(wchoicecol.c_str(), wchoice[cursor]);
        }
    }

    if (!m_sel_only) {
        map.setDisplayedAttribute(attributes.getColumnIndex(choicecol.c_str()));
    } else {
        map.setDisplayedAttribute(attributes.getColumnIndex(meandepthcol.c_str()));
    }

    return retvar;
}

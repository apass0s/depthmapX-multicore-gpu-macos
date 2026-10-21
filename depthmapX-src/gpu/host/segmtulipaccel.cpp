// depthmapX GPU module - segment tulip (angular) analysis on all CPU cores
//
// Upstream (salalib/segmmodules/segmtulip.cpp) runs one bucketed search per segment.
// Its per-search state (audit trail, uncovered masks, bins) is reset for every source,
// so here each thread keeps its own copy and runs the unchanged search. Only choice is
// shared between sources:
//  - unweighted choice adds integers (path counts) to doubles, exact in any order, so
//    each thread sums its own and the totals are added at the end;
//  - weighted choice adds fractional values, whose rounding depends on order, so each
//    source logs its additions and the logs are replayed in upstream's source order.
// Results are therefore identical to upstream.

#include "gpu/host/segmtulipaccel.h"
#include "gpu/host/shaperows.h"

#include "genlib/containerutils.h"
#include "genlib/stringutils.h"

#include <algorithm>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

struct Trail { // AnalysisInfo without the cumulative choice values
    int depth;
    SegmentRef previous;
    bool leaf;
    bool choicecovered;
};

struct SourceResult {
    std::vector<double> node_count, total_depth, total_weight, total_weighted_depth; // per radius
    std::vector<std::pair<size_t, double>> weighted_log; // (trail index, addition), in upstream order
};

struct TulipSetup {
    const std::vector<Connector> &connections;
    size_t n;
    int tulip_bins; // already halved + 1, as used by the search
    int radiussize;
    int radiusmask;
    std::vector<double> radius;
    std::vector<float> weights;
    std::vector<float> lengths;
    int radius_type;
    bool weighted;
    bool choice;
};

class TulipSearch {
  public:
    explicit TulipSearch(const TulipSetup &s)
        : m_s(s), m_bins(size_t(s.tulip_bins)), m_trail(s.n * size_t(s.radiussize) * 2),
          m_uncovered(s.n * 2), m_choice(s.choice ? s.n * size_t(s.radiussize) * 2 : 0, 0.0) {}

    std::vector<double> &choice() { return m_choice; }

    void run(size_t cursor, double rootseglength, SourceResult &out) {
        const TulipSetup &s = m_s;
        const int tulip_bins = s.tulip_bins, radiussize = s.radiussize, radiusmask = s.radiusmask;
        const std::vector<double> &radius = s.radius;
        const std::vector<float> &weights = s.weights;
        const std::vector<float> &lengths = s.lengths;

        for (int k = 0; k < tulip_bins; k++) {
            m_bins[size_t(k)].clear();
        }
        for (size_t j = 0; j < s.n; j++) {
            for (int dir = 0; dir < 2; dir++) {
                for (int k = 0; k < radiussize; k++) {
                    Trail &t = trail(j, k, dir);
                    t.choicecovered = false;
                    t.leaf = true;
                    t.previous = SegmentRef();
                    t.depth = 0;
                }
                uncovered(j, dir) = radiusmask;
            }
        }

        double rootweight = s.weighted ? weights[cursor] : 0.0;

        // setup: direction 0 (both ways), segment i, previous -1, segdepth (step depth) 0, metricdepth 0.5 *
        // rootseglength, bin 0
        SegmentData segmentData(0, int(cursor), SegmentRef(), 0, 0.5 * rootseglength, radiusmask);
        auto it = std::lower_bound(m_bins[0].begin(), m_bins[0].end(), segmentData);
        if (it == m_bins[0].end() || segmentData != *it) {
            m_bins[0].insert(it, segmentData);
        }
        int depthlevel = 0;
        int opencount = 1;
        size_t currentbin = 0;
        while (opencount) {
            while (!m_bins[currentbin].size()) {
                depthlevel++;
                currentbin++;
                if (currentbin == static_cast<size_t>(tulip_bins)) {
                    currentbin = 0;
                }
            }
            SegmentData lineindex = m_bins[currentbin].back();
            m_bins[currentbin].pop_back();
            //
            opencount--;

            int ref = lineindex.ref;
            int dir = (lineindex.dir == 1) ? 0 : 1;
            int coverage = lineindex.coverage & uncovered(size_t(ref), dir);
            if (coverage != 0) {
                int rbin = 0;
                int rbinbase;
                if (lineindex.previous.ref != -1) {
                    uncovered(size_t(ref), dir) &= ~coverage;
                    while (((coverage >> rbin) & 0x1) == 0)
                        rbin++;
                    rbinbase = rbin;
                    while (rbin < radiussize) {
                        if (((coverage >> rbin) & 0x1) == 1) {
                            trail(size_t(ref), rbin, dir).depth = depthlevel;
                            trail(size_t(ref), rbin, dir).previous = lineindex.previous;
                            trail(size_t(lineindex.previous.ref), rbin, (lineindex.previous.dir == 1) ? 0 : 1).leaf =
                                false;
                        }
                        rbin++;
                    }
                } else {
                    rbinbase = 0;
                    uncovered(size_t(ref), 0) &= ~coverage;
                    uncovered(size_t(ref), 1) &= ~coverage;
                }
                const Connector &line = s.connections[size_t(ref)];
                if (lineindex.dir != -1) {
                    expand(line.m_forward_segconns, 1, lineindex, coverage, rbinbase, depthlevel, currentbin,
                           opencount, lengths);
                }
                if (lineindex.dir != 1) {
                    expand(line.m_back_segconns, -1, lineindex, coverage, rbinbase, depthlevel, currentbin,
                           opencount, lengths);
                }
            }
        }

        // totals and choice, as upstream after the search
        out.node_count.assign(size_t(radiussize), 0.0);
        out.total_depth.assign(size_t(radiussize), 0.0);
        out.total_weight.assign(size_t(radiussize), 0.0);
        out.total_weighted_depth.assign(size_t(radiussize), 0.0);
        out.weighted_log.clear();
        for (int k = 0; k < radiussize; k++) {
            // note, curs_total_depth must use double as mantissa can get too long for int in large systems
            double curs_node_count = 0.0, curs_total_depth = 0.0;
            double curs_total_weight = 0.0, curs_total_weighted_depth = 0.0;
            for (size_t j = 0; j < s.n; j++) {
                // find dir according
                bool m0 = ((uncovered(j, 0) >> k) & 0x1) == 0;
                bool m1 = ((uncovered(j, 1) >> k) & 0x1) == 0;
                if ((m0 | m1) != 0) {
                    int dir;
                    if (m0 & m1) {
                        // dir is the one with the lowest depth:
                        if (trail(j, k, 0).depth < trail(j, k, 1).depth)
                            dir = 0;
                        else
                            dir = 1;
                    } else {
                        // dir is simply the one that's filled in:
                        dir = m0 ? 0 : 1;
                    }
                    curs_node_count++;
                    curs_total_depth += trail(j, k, dir).depth;
                    curs_total_weight += weights[j];
                    curs_total_weighted_depth += trail(j, k, dir).depth * weights[j];
                    //
                    if (s.choice && trail(j, k, dir).leaf) {
                        // note, graph may be directed (e.g., for one way streets), so both ways must be included from
                        // now on:
                        SegmentRef here = SegmentRef(dir == 0 ? 1 : -1, int(j));
                        if (here.ref != static_cast<int>(cursor)) {
                            int choicecount = 0;
                            double choiceweight = 0.0;
                            while (here.ref != static_cast<int>(cursor)) { // not rowid means not the current root
                                int heredir = (here.dir == 1) ? 0 : 1;
                                const size_t hi = index(size_t(here.ref), k, heredir);
                                // each node has the existing choicecount and choiceweight from previously
                                // encountered nodes added to it
                                m_choice[hi] += choicecount;
                                // nb, weighted values calculated anyway to save time on 'if'
                                if (s.weighted) {
                                    out.weighted_log.emplace_back(hi, choiceweight);
                                }
                                // if the node hasn't been encountered before, the choicecount and choiceweight is
                                // incremented for all remaining nodes to be encountered on the backwards route from it
                                if (!m_trail[hi].choicecovered) {
                                    choicecount++;
                                    choiceweight += weights[size_t(here.ref)] * rootweight;
                                    m_trail[hi].choicecovered = true;
                                    // note, for weighted choice, the start and end points have choice added to them:
                                    if (s.weighted) {
                                        out.weighted_log.emplace_back(hi,
                                                                      (weights[size_t(here.ref)] * rootweight) / 2.0);
                                    }
                                }
                                here = m_trail[hi].previous;
                            }
                            // note, for weighted choice, the start and end points have choice added to them:
                            // (this is the summed weight for all starting nodes encountered in this path)
                            if (s.weighted) {
                                out.weighted_log.emplace_back(index(size_t(here.ref), k, (here.dir == 1) ? 0 : 1),
                                                              choiceweight / 2.0);
                            }
                        }
                    }
                }
            }
            out.node_count[size_t(k)] = curs_node_count;
            out.total_depth[size_t(k)] = curs_total_depth;
            out.total_weight[size_t(k)] = curs_total_weight;
            out.total_weighted_depth[size_t(k)] = curs_total_weighted_depth;
        }
    }

    size_t index(size_t j, int k, int dir) const { return (j * size_t(m_s.radiussize) + size_t(k)) * 2 + size_t(dir); }

  private:
    Trail &trail(size_t j, int k, int dir) { return m_trail[index(j, k, dir)]; }
    unsigned int &uncovered(size_t j, int dir) { return m_uncovered[j * 2 + size_t(dir)]; }

    // upstream's loop over line.m_forward_segconns (dirSign 1) / m_back_segconns (dirSign -1)
    void expand(const std::map<SegmentRef, float> &segconns, char dirSign, const SegmentData &lineindex, int coverage,
                int rbinbase, int depthlevel, size_t currentbin, int &opencount, const std::vector<float> &lengths) {
        const TulipSetup &s = m_s;
        const int tulip_bins = s.tulip_bins, radiussize = s.radiussize;
        const std::vector<double> &radius = s.radius;
        float seglength;
        int extradepth;
        for (auto &segconn : segconns) {
            int rbin = rbinbase;
            SegmentRef conn = segconn.first;
            if ((uncovered(size_t(conn.ref), (conn.dir == 1 ? 0 : 1)) & coverage) != 0) {
                extradepth = (int)floor(segconn.second * tulip_bins * 0.5);
                seglength = lengths[size_t(conn.ref)];
                switch (s.radius_type) {
                case Options::RADIUS_ANGULAR:
                    while (rbin != radiussize && radius[size_t(rbin)] != -1 &&
                           depthlevel + extradepth > (int)radius[size_t(rbin)]) {
                        rbin++;
                    }
                    break;
                case Options::RADIUS_METRIC:
                    while (rbin != radiussize && radius[size_t(rbin)] != -1 &&
                           lineindex.metricdepth + seglength * 0.5 > radius[size_t(rbin)]) {
                        rbin++;
                    }
                    break;
                case Options::RADIUS_STEPS:
                    if (rbin != radiussize && radius[size_t(rbin)] != -1 &&
                        lineindex.segdepth >= (int)radius[size_t(rbin)]) {
                        rbin++;
                    }
                    break;
                }
                if ((coverage >> rbin) != 0) {
                    SegmentData sd(conn, SegmentRef(dirSign, lineindex.ref), lineindex.segdepth + 1,
                                   lineindex.metricdepth + seglength, (coverage >> rbin) << rbin);
                    size_t bin = (currentbin + tulip_bins + extradepth) % tulip_bins;
                    depthmapX::insert_sorted(m_bins[bin], sd);
                    opencount++;
                }
            }
        }
    }

    const TulipSetup &m_s;
    std::vector<std::vector<SegmentData>> m_bins;
    std::vector<Trail> m_trail;
    std::vector<unsigned int> m_uncovered;
    std::vector<double> m_choice; // this thread's choice sums (integer-valued)
};

} // namespace

bool SegmentTulipAccel::run(Communicator *comm, ShapeGraph &map, bool) {

    if (map.getMapType() != ShapeMap::SEGMENTMAP) {
        return false;
    }

    AttributeTable &attributes = map.getAttributeTable();

    int processed_rows = 0;

    time_t atime = 0;

    if (comm) {
        qtimer(atime, 0);
        comm->CommPostMessage(Communicator::NUM_RECORDS,
                              (m_sel_only ? map.getSelSet().size() : map.getConnections().size()));
    }

    // Same radius ordering as SegmentTulip::run (including its int conversion).
    bool radius_n = false;
    std::vector<double> radius_unconverted;
    for (int radius : m_radius_set) {
        if (radius == -1.0) {
            radius_n = true;
        } else {
            radius_unconverted.push_back(radius);
        }
    }
    if (radius_n) {
        radius_unconverted.push_back(-1.0);
    }

    // retrieve weighted col data, as this may well be overwritten in the new analysis:
    std::vector<float> weights;
    std::string weighting_col_text;

    int tulip_bins = m_tulip_bins;
    const std::vector<AttributeRow *> shapeRows = dmxaccel::rowsByShapeIndex(map);

    if (m_weighted_measure_col != -1) {
        weighting_col_text = attributes.getColumnName(m_weighted_measure_col);
        for (size_t i = 0; i < map.getConnections().size(); i++) {
            weights.push_back(shapeRows[i]->getValue(m_weighted_measure_col));
        }
    } else { // Normal run // TV
        for (size_t i = 0; i < map.getConnections().size(); i++) {
            weights.push_back(1.0f);
        }
    }

    std::string tulip_text = std::string("T") + dXstring::formatString(tulip_bins, "%d");

    // first enter the required attribute columns (SegmentTulip::run, "Normal run" branches):
    size_t r;
    for (r = 0; r < radius_unconverted.size(); r++) {
        std::string radius_text = makeRadiusText(m_radius_type, radius_unconverted[r]);
        if (m_choice) {
            std::string choice_col_text = tulip_text + " Choice" + radius_text;
            attributes.insertOrResetColumn(choice_col_text.c_str());
            if (m_weighted_measure_col != -1) {
                std::string w_choice_col_text = tulip_text + " Choice [" + weighting_col_text + " Wgt]" + radius_text;
                attributes.insertOrResetColumn(w_choice_col_text.c_str());
            }
        }
        std::string integ_col_text =
            tulip_text + " Integration" + radius_text; // <- note, the fact this is a tulip is unnecessary
        std::string w_integ_col_text = tulip_text + " Integration [" + weighting_col_text + " Wgt]" + radius_text;

        std::string count_col_text =
            tulip_text + " Node Count" + radius_text; // <- note, the fact this is a tulip is unnecessary
        std::string td_col_text =
            tulip_text + " Total Depth" + radius_text; // <- note, the fact this is a tulip is unnecessary
        // '[' comes after 'R' in ASCII, so this column will come after Mean Depth R...
        std::string w_td_text = tulip_text + " Total Depth [" + weighting_col_text + " Wgt]" + radius_text;
        std::string total_weight_text = tulip_text + " Total " + weighting_col_text + radius_text;

        attributes.insertOrResetColumn(integ_col_text.c_str());
        attributes.insertOrResetColumn(count_col_text.c_str());
        attributes.insertOrResetColumn(td_col_text.c_str());
        if (m_weighted_measure_col != -1) {
            attributes.insertOrResetColumn(w_integ_col_text.c_str());
            attributes.insertOrResetColumn(w_td_text.c_str());
            attributes.insertOrResetColumn(total_weight_text.c_str());
        }
    }
    std::vector<int> choice_col, w_choice_col, count_col, integ_col, w_integ_col, td_col, w_td_col, total_weight_col;
    // then look them up! eek....
    for (r = 0; r < radius_unconverted.size(); r++) {
        std::string radius_text = makeRadiusText(m_radius_type, radius_unconverted[r]);
        if (m_choice) {
            std::string choice_col_text = tulip_text + " Choice" + radius_text;
            choice_col.push_back(attributes.getColumnIndex(choice_col_text.c_str()));
            if (m_weighted_measure_col != -1) {
                std::string w_choice_col_text = tulip_text + " Choice [" + weighting_col_text + " Wgt]" + radius_text;
                w_choice_col.push_back(attributes.getColumnIndex(w_choice_col_text.c_str()));
            }
        }
        std::string integ_col_text =
            tulip_text + " Integration" + radius_text; // <- note, the fact this is a tulip is unnecessary
        std::string w_integ_col_text = tulip_text + " Integration [" + weighting_col_text + " Wgt]" + radius_text;

        std::string count_col_text =
            tulip_text + " Node Count" + radius_text; // <- note, the fact this is a tulip is unnecessary
        std::string td_col_text =
            tulip_text + " Total Depth" + radius_text; // <- note, the fact this is a tulip is unnecessary
        std::string w_td_text = tulip_text + " Total Depth [" + weighting_col_text + " Wgt]" + radius_text;
        std::string total_weight_col_text = tulip_text + " Total " + weighting_col_text + radius_text;

        integ_col.push_back(attributes.getColumnIndex(integ_col_text.c_str()));
        count_col.push_back(attributes.getColumnIndex(count_col_text.c_str()));
        td_col.push_back(attributes.getColumnIndex(td_col_text.c_str()));
        if (m_weighted_measure_col != -1) {
            // '[' comes after 'R' in ASCII, so this column will come after Mean Depth R...
            w_integ_col.push_back(attributes.getColumnIndex(w_integ_col_text.c_str()));
            w_td_col.push_back(attributes.getColumnIndex(w_td_text.c_str()));
            total_weight_col.push_back(attributes.getColumnIndex(total_weight_col_text.c_str()));
        }
    }

    tulip_bins /= 2; // <- actually use semicircle of tulip bins
    tulip_bins += 1;

    std::vector<double> radius;
    for (r = 0; r < radius_unconverted.size(); r++) {
        if (m_radius_type == Options::RADIUS_ANGULAR && radius_unconverted[r] != -1) {
            radius.push_back(floor(radius_unconverted[r] * tulip_bins * 0.5));
        } else {
            radius.push_back(radius_unconverted[r]);
        }
    }
    // entered once for each segment
    int length_col = attributes.getColumnIndex("Segment Length");
    std::vector<float> lengths;
    if (length_col != -1) {
        for (size_t i = 0; i < map.getConnections().size(); i++) {
            AttributeRow &row = *shapeRows[i];
            lengths.push_back(row.getValue(length_col));
        }
    }

    int radiussize = int(radius.size());
    int radiusmask = 0;
    for (int i = 0; i < radiussize; i++) {
        radiusmask |= (1 << i);
    }

    const size_t n = map.getConnections().size();
    TulipSetup setup{map.getConnections(), n, tulip_bins, radiussize, radiusmask, radius, weights,
                     lengths, m_radius_type, m_weighted_measure_col != -1, m_choice};

    // sources in upstream order
    std::vector<size_t> sources;
    std::vector<double> rootlengths;
    for (size_t cursor = 0; cursor < n; cursor++) {
        AttributeRow &row = *shapeRows[cursor];
        if (m_sel_only && !row.isSelected()) {
            continue;
        }
        sources.push_back(cursor);
        rootlengths.push_back(row.getValue(length_col));
    }

    const int threads = dmxaccel::cpuThreadCount();
    const size_t trailSize = n * size_t(radiussize) * 2;
    std::vector<double> weighted_choice(m_choice && setup.weighted ? trailSize : 0, 0.0);
    std::vector<std::vector<double>> threadChoice(static_cast<size_t>(threads));
    // weighted runs keep each source's choice log until it is replayed, so use smaller batches
    const int batchSize = setup.weighted && m_choice ? 4 * threads : 32 * threads;
    std::vector<SourceResult> results(static_cast<size_t>(batchSize));

    bool cancelled = false;
    for (size_t start = 0; start < sources.size() && !cancelled; start += size_t(batchSize)) {
        const int count = int(std::min(sources.size() - start, size_t(batchSize)));
#pragma omp parallel
        {
#ifdef _OPENMP
            const int tid = omp_get_thread_num();
#else
            const int tid = 0;
#endif
            TulipSearch search(setup);
            if (!threadChoice[size_t(tid)].empty()) {
                search.choice().swap(threadChoice[size_t(tid)]); // continue this thread's sums
            }
#pragma omp for schedule(dynamic, 1)
            for (int i = 0; i < count; i++) {
                search.run(sources[start + size_t(i)], rootlengths[start + size_t(i)], results[size_t(i)]);
            }
            threadChoice[size_t(tid)].swap(search.choice());
        }
        for (int i = 0; i < count; i++) {
            const SourceResult &res = results[size_t(i)];
            for (const auto &add : res.weighted_log) { // upstream's order of additions
                weighted_choice[add.first] += add.second;
            }
            AttributeRow &row = *shapeRows[sources[start + size_t(i)]];
            for (int k = 0; k < radiussize; k++) {
                const double curs_node_count = res.node_count[size_t(k)];
                const double curs_total_weight = res.total_weight[size_t(k)];
                double total_depth_conv = res.total_depth[size_t(k)] / ((tulip_bins - 1.0f) * 0.5f);
                double total_weighted_depth_conv = res.total_weighted_depth[size_t(k)] / ((tulip_bins - 1.0f) * 0.5f);
                //
                row.setValue(count_col[size_t(k)], float(curs_node_count));
                if (curs_node_count > 1) {
                    // for dmap 8 and above, mean depth simply isn't calculated as for radius measures it is
                    // meaningless
                    row.setValue(td_col[size_t(k)], total_depth_conv);
                    if (m_weighted_measure_col != -1) {
                        row.setValue(total_weight_col[size_t(k)], float(curs_total_weight));
                        row.setValue(w_td_col[size_t(k)], float(total_weighted_depth_conv));
                    }
                } else {
                    row.setValue(td_col[size_t(k)], -1);
                    if (m_weighted_measure_col != -1) {
                        row.setValue(total_weight_col[size_t(k)], -1.0f);
                        row.setValue(w_td_col[size_t(k)], -1.0f);
                    }
                }
                // for dmap 10 an above, integration is included!
                if (total_depth_conv > 1e-9) {
                    row.setValue(integ_col[size_t(k)], (float)(curs_node_count * curs_node_count / total_depth_conv));
                    if (m_weighted_measure_col != -1) {
                        row.setValue(w_integ_col[size_t(k)],
                                     (float)(curs_total_weight * curs_total_weight / total_weighted_depth_conv));
                    }
                } else {
                    row.setValue(integ_col[size_t(k)], -1);
                    if (m_weighted_measure_col != -1) {
                        row.setValue(w_integ_col[size_t(k)], -1.0f);
                    }
                }
            }
            processed_rows++;
        }
        if (comm && qtimer(atime, 500)) {
            if (comm->IsCancelled()) {
                // upstream (non-interactive) keeps what has been processed already
                cancelled = true;
            } else {
                comm->CommPostMessage(Communicator::CURRENT_RECORD, int(start) + count);
            }
        }
    }
    dmxaccel::addReport("Segment tulip: CPU, " + std::to_string(threads) + " threads, " + std::to_string(n) +
                        " segments");

    if (m_choice) {
        // integer-valued sums: exact in any order
        std::vector<double> choice(trailSize, 0.0);
        for (const auto &c : threadChoice) {
            for (size_t i = 0; i < c.size(); i++) {
                choice[i] += c[i];
            }
        }
        auto at = [&](size_t j, size_t k, int dir) { return (j * size_t(radiussize) + k) * 2 + size_t(dir); };
        for (size_t cursor = 0; cursor < n; cursor++) {
            AttributeRow &row = *shapeRows[cursor];
            for (size_t r = 0; r < radius.size(); r++) {
                double total_choice = choice[at(cursor, r, 0)] + choice[at(cursor, r, 1)];
                row.setValue(choice_col[r], float(total_choice));
                if (m_weighted_measure_col != -1) {
                    double total_weighted_choice = weighted_choice[at(cursor, r, 0)] + weighted_choice[at(cursor, r, 1)];
                    row.setValue(w_choice_col[r], float(total_weighted_choice));
                }
            }
        }
    }

    map.setDisplayedAttribute(-2); // <- override if it's already showing
    if (m_choice) {
        map.setDisplayedAttribute(choice_col.back());
    } else {
        map.setDisplayedAttribute(td_col.back());
    }
    return processed_rows > 0;
}

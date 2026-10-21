// depthmapX GPU module - axial integration (and choice) on all CPU cores
//
// Upstream (salalib/axialmodules/axialintegration.cpp) runs one breadth-first search per
// line. Up to the column set-up this file is upstream's code; the main loop below is
// upstream's search with per-thread state:
//  - row values are buffered and written in upstream's order (setValue also updates
//    column statistics);
//  - with choice, upstream takes the next line at random with pafrand(), a 64-bit LCG
//    with global state. The number of draws per source does not depend on the draws,
//    so they are counted first and each source starts from the generator state upstream
//    would have reached (LCG jump-ahead); the global state is left as upstream leaves it;
//  - choice adds integers (exact in any order) and is summed per thread; weighted
//    choice additions are logged and replayed in upstream's order.
// Results are therefore identical to upstream.

#include "gpu/host/axialintegrationaccel.h"
#include "gpu/host/shaperows.h"

#include "salalib/axialmodules/axialintegration.h"

#include "genlib/containerutils.h"
#include "genlib/pafmath.h"
#include "genlib/stringutils.h"

#include <algorithm>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

struct AxialSetup {
    const std::vector<Connector> &connections;
    size_t nlines;
    const std::vector<int> &radii;
    const std::vector<double> &weights;
    bool weighted, choice, local, fulloutput, simple_version;
};

struct AxialColumns {
    const std::vector<int> &choice_col, &n_choice_col, &w_choice_col, &nw_choice_col, &entropy_col, &integ_dv_col,
        &integ_pv_col, &integ_tk_col, &intensity_col, &depth_col, &count_col, &rel_entropy_col, &penn_norm_col,
        &w_depth_col, &total_weight_col, &ra_col, &rra_col, &td_col, &harmonic_col;
    int control_col, controllability_col;
};

struct AxialResult {
    std::vector<std::pair<int, float>> writes;            // upstream's row.setValue calls, in order
    std::vector<std::pair<size_t, double>> weighted_log; // weighted choice additions, in order
};

// AxialIntegration::run's per-line search, with its own state and random generator.
class AxialSearch {
  public:
    AxialSearch(const AxialSetup &s, const AxialColumns &c)
        : m_s(s), m_c(c), m_covered(s.nlines, 0), m_prev(s.choice ? s.nlines : 0, -1),
          m_choice(s.choice ? s.nlines * s.radii.size() : 0, 0.0) {}

    std::vector<double> &choice() { return m_choice; }

    // Number of pafrand() draws the search from line i makes (one per line taken from
    // the front); the same whichever lines the draws pick.
    int64_t countDraws(int iInt) {
        const size_t i = size_t(iInt);
        std::fill(m_covered.begin(), m_covered.end(), 0);
        m_a.clear();
        m_b.clear();
        m_a.push_back(std::pair<int, int>(int(i), -1));
        m_covered[i] = 1;
        int depth = 1;
        int64_t draws = 0;
        for (int radius : m_s.radii) {
            while (m_a.size()) {
                const int index = m_a.back().first;
                draws++;
                for (int c : m_s.connections[size_t(index)].m_connections) {
                    if (!m_covered[size_t(c)]) {
                        m_covered[size_t(c)] = 1;
                        m_b.push_back(std::pair<int, int>(c, index));
                    }
                }
                m_a.pop_back();
                if (!m_a.size()) {
                    std::swap(m_a, m_b);
                    depth++;
                    if (radius != -1 && depth > radius) {
                        break;
                    }
                }
            }
        }
        return draws;
    }

    void run(int iInt, uint64_t rng, AxialResult &out) {
        const AxialSetup &s = m_s;
        const AxialColumns &c = m_c;
        const bool simple_version = s.simple_version;
        const size_t nradii = s.radii.size();
        out.writes.clear();
        out.weighted_log.clear();
        auto set = [&](int col, float value) { out.writes.emplace_back(col, value); };
        const size_t i = size_t(iInt);

        std::fill(m_covered.begin(), m_covered.end(), 0);
        if (s.choice) {
            std::fill(m_prev.begin(), m_prev.end(), -1); // note, choice values are cumulative
        }
        if (s.local) {
            double control = 0.0;
            const std::vector<int> &connections = s.connections[i].m_connections;
            std::vector<int> totalneighbourhood;
            for (int connection : connections) {
                // n.b., as of Depthmap 10.0, connections[j] and i cannot coexist
                depthmapX::addIfNotExists(totalneighbourhood, connection);
                int retro_size = 0;
                auto &retconnectors = s.connections[size_t(connection)].m_connections;
                for (auto retconnector : retconnectors) {
                    retro_size++;
                    depthmapX::addIfNotExists(totalneighbourhood, retconnector);
                }
                control += 1.0 / double(retro_size);
            }
            if (!simple_version) {
                if (connections.size() > 0) {
                    set(c.control_col, float(control));
                    set(c.controllability_col,
                        float(double(connections.size()) / double(totalneighbourhood.size() - 1)));
                } else {
                    set(c.control_col, -1);
                    set(c.controllability_col, -1);
                }
            }
        }

        std::vector<int> depthcounts;
        depthcounts.push_back(0);
        m_a.clear();
        m_b.clear();
        m_a.push_back(std::pair<int, int>(int(i), -1));
        m_covered[i] = 1;
        int total_depth = 0, depth = 1, max_depth = 0, node_count = 1, pos = -1,
            previous = -1; // node_count includes this 1
        double weight = 0.0, rootweight = 0.0, total_weight = 0.0, w_total_depth = 0.0;
        if (s.weighted) {
            rootweight = s.weights[i];
            // include this line in total weights (as per nodecount)
            total_weight += rootweight;
        }
        int index = -1;
        int r = 0;
        for (int radius : s.radii) {
            while (m_a.size()) {
                if (!s.choice) {
                    index = m_a.back().first;
                } else {
                    rng = dmxaccel::pafrandStep(rng);
                    pos = dmxaccel::pafrandValue(rng) % m_a.size();
                    index = m_a.at(size_t(pos)).first;
                    previous = m_a.at(size_t(pos)).second;
                    m_prev[size_t(index)] = previous;
                }
                const Connector &line = s.connections[size_t(index)];
                for (size_t k = 0; k < line.m_connections.size(); k++) {
                    const int conn = line.m_connections[k];
                    if (!m_covered[size_t(conn)]) {
                        m_covered[size_t(conn)] = 1;
                        m_b.push_back(std::pair<int, int>(conn, index));
                        if (s.weighted) {
                            // the weight is taken from the discovered node:
                            weight = s.weights[size_t(conn)];
                            total_weight += weight;
                            w_total_depth += depth * weight;
                        }
                        if (s.choice && previous != -1) {
                            // both directional paths are now recorded for choice
                            size_t here = size_t(index); // note: start counting from index as actually looking ahead
                            while (here != i) {          // not i means not the current root for the path
                                m_choice[here * nradii + size_t(r)] += 1;
                                if (s.weighted) {
                                    out.weighted_log.emplace_back(here * nradii + size_t(r), weight * rootweight);
                                }
                                here = size_t(m_prev[here]);
                            }
                            if (s.weighted) {
                                // in weighted choice, root node and current node receive values:
                                out.weighted_log.emplace_back(i * nradii + size_t(r), (weight * rootweight) * 0.5);
                                out.weighted_log.emplace_back(size_t(conn) * nradii + size_t(r),
                                                              (weight * rootweight) * 0.5);
                            }
                        }
                        total_depth += depth;
                        max_depth = depth;
                        node_count++;
                        depthcounts.back() += 1;
                    }
                }
                if (!s.choice)
                    m_a.pop_back();
                else
                    m_a.erase(m_a.begin() + pos);
                if (!m_a.size()) {
                    std::swap(m_a, m_b);
                    depth++;
                    depthcounts.push_back(0);
                    if (radius != -1 && depth > radius) {
                        break;
                    }
                }
            }
            // set the attributes for this node:
            set(c.count_col[size_t(r)], float(node_count));
            if (s.weighted) {
                set(c.total_weight_col[size_t(r)], float(total_weight));
            }
            // node count > 1 to avoid divide by zero (was > 2)
            if (node_count > 1) {
                // note -- node_count includes this one -- mean depth as per p.108 Social Logic of Space
                double mean_depth = double(total_depth) / double(node_count - 1);
                set(c.depth_col[size_t(r)], float(mean_depth));
                if (s.weighted) {
                    // weighted mean depth:
                    set(c.w_depth_col[size_t(r)], float(w_total_depth / total_weight));
                }
                // total nodes > 2 to avoid divide by 0 (was > 3)
                if (node_count > 2 && mean_depth > 1.0) {
                    double ra = 2.0 * (mean_depth - 1.0) / double(node_count - 2);
                    // d-value / p-value from Depthmap 4 manual, note: node_count includes this one
                    double rra_d = ra / dvalue(node_count);
                    double rra_p = ra / pvalue(node_count);
                    double integ_tk = teklinteg(node_count, total_depth);
                    set(c.integ_dv_col[size_t(r)], float(1.0 / rra_d));
                    if (!simple_version) {
                        set(c.integ_pv_col[size_t(r)], float(1.0 / rra_p));
                        if (total_depth - node_count + 1 > 1) {
                            set(c.integ_tk_col[size_t(r)], float(integ_tk));
                        } else {
                            set(c.integ_tk_col[size_t(r)], -1.0f);
                        }
                    }
                    if (s.fulloutput) {
                        set(c.ra_col[size_t(r)], float(ra));
                        if (!simple_version) {
                            set(c.rra_col[size_t(r)], float(rra_d));
                        }
                        set(c.td_col[size_t(r)], float(total_depth));
                        if (!simple_version) {
                            // alan's palm-tree normalisation: palmtree
                            double dmin = node_count - 1;
                            double dmax = palmtree(node_count, max_depth);
                            if (dmax != dmin) {
                                set(c.penn_norm_col[size_t(r)], float((dmax - total_depth) / (dmax - dmin)));
                            }
                        }
                    }
                } else {
                    set(c.integ_dv_col[size_t(r)], -1.0f);
                    if (!simple_version) {
                        set(c.integ_pv_col[size_t(r)], -1.0f);
                        set(c.integ_tk_col[size_t(r)], -1.0f);
                    }
                    if (s.fulloutput) {
                        set(c.ra_col[size_t(r)], -1.0f);
                        if (!simple_version) {
                            set(c.rra_col[size_t(r)], -1.0f);
                        }
                        set(c.td_col[size_t(r)], -1.0f);
                        if (!simple_version) {
                            set(c.penn_norm_col[size_t(r)], -1.0f);
                        }
                    }
                }
                if (!simple_version) {
                    double entropy = 0.0, intensity = 0.0, rel_entropy = 0.0, factorial = 1.0, harmonic = 0.0;
                    for (size_t k = 0; k < depthcounts.size(); k++) {
                        if (depthcounts[k] != 0) {
                            // some debate over whether or not this should be node count - 1
                            // (i.e., including or not including the node itself)
                            double prob = double(depthcounts[k]) / double(node_count);
                            entropy -= prob * pafLog2(prob);
                            // Formula from Turner 2001, "Depthmap"
                            factorial *= double(k + 1);
                            double q = (pow(mean_depth, double(k)) / double(factorial)) * exp(-mean_depth);
                            rel_entropy += (double)prob * pafLog2(prob / q);
                            //
                            harmonic += 1.0 / double(depthcounts[k]);
                        }
                    }
                    harmonic = double(depthcounts.size()) / harmonic;
                    if (total_depth > node_count) {
                        intensity = node_count * entropy / (total_depth - node_count);
                    } else {
                        intensity = -1;
                    }
                    set(c.entropy_col[size_t(r)], float(entropy));
                    set(c.rel_entropy_col[size_t(r)], float(rel_entropy));
                    set(c.intensity_col[size_t(r)], float(intensity));
                    set(c.harmonic_col[size_t(r)], float(harmonic));
                }
            } else {
                set(c.depth_col[size_t(r)], -1.0f);
                set(c.integ_dv_col[size_t(r)], -1.0f);
                if (!simple_version) {
                    set(c.integ_pv_col[size_t(r)], -1.0f);
                    set(c.integ_tk_col[size_t(r)], -1.0f);
                    set(c.entropy_col[size_t(r)], -1.0f);
                    set(c.rel_entropy_col[size_t(r)], -1.0f);
                    set(c.harmonic_col[size_t(r)], -1.0f);
                }
            }
            ++r;
        }
    }

  private:
    const AxialSetup &m_s;
    const AxialColumns &m_c;
    std::vector<char> m_covered;
    std::vector<int> m_prev;     // upstream's audittrail[..][0].previous.ref
    std::vector<double> m_choice; // this thread's choice sums (integer-valued)
    std::vector<std::pair<int, int>> m_a, m_b; // upstream's foundlist (a = current front)
};

} // namespace

bool AxialIntegrationAccel::run(Communicator *comm, ShapeGraph &map, bool simple_version) {
    if (m_choice && !dmxaccel::pafrandModelOk()) {
        // cannot reproduce upstream's random draws: run the original code
        dmxaccel::addReport("Axial integration: original serial code (random generator check failed)");
        return AxialIntegration(m_radius_set, m_weighted_measure_col, m_choice, m_fulloutput, m_local)
            .run(comm, map, simple_version);
    }

    // note, from 10.0, Depthmap no longer includes *self* connections on axial lines
    // self connections are stripped out on loading graph files, as well as no longer made

    time_t atime = 0;
    if (comm) {
        qtimer(atime, 0);
        comm->CommPostMessage(Communicator::NUM_RECORDS, map.getShapeCount());
    }

    AttributeTable &attributes = map.getAttributeTable();

    // note: radius must be sorted lowest to highest, but if -1 occurs ("radius n") it needs to be last...
    // ...to ensure no mess ups, we'll re-sort here:
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

    // retrieve weighted col data, as this may well be overwritten in the new analysis:
    std::vector<double> weights;
    std::string weighting_col_text;
    if (m_weighted_measure_col != -1) {
        weighting_col_text = attributes.getColumnName(m_weighted_measure_col);
        const std::vector<AttributeRow *> shapeRows = dmxaccel::rowsByShapeIndex(map);
        for (size_t i = 0; i < map.getShapeCount(); i++) {
            weights.push_back(shapeRows[i]->getValue(m_weighted_measure_col));
        }
    }

    // first enter the required attribute columns:
    for (int radius : radii) {
        std::string radius_text;
        if (radius != -1) {
            radius_text = dXstring::formatString(radius, " R%d");
        }
        if (m_choice) {
            std::string choice_col_text = std::string("Choice") + radius_text;
            attributes.insertOrResetColumn(choice_col_text.c_str());
            std::string n_choice_col_text = std::string("Choice [Norm]") + radius_text;
            attributes.insertOrResetColumn(n_choice_col_text.c_str());
            if (m_weighted_measure_col != -1) {
                std::string w_choice_col_text = std::string("Choice [") + weighting_col_text + " Wgt]" + radius_text;
                attributes.insertOrResetColumn(w_choice_col_text.c_str());
                std::string nw_choice_col_text =
                    std::string("Choice [") + weighting_col_text + " Wgt][Norm]" + radius_text;
                attributes.insertOrResetColumn(nw_choice_col_text.c_str());
            }
        }

        if (!simple_version) {
            std::string entropy_col_text = std::string("Entropy") + radius_text;
            attributes.insertOrResetColumn(entropy_col_text.c_str());
        }

        std::string integ_dv_col_text = std::string("Integration [HH]") + radius_text;
        attributes.insertOrResetColumn(integ_dv_col_text.c_str());

        if (!simple_version) {
            std::string integ_pv_col_text = std::string("Integration [P-value]") + radius_text;
            attributes.insertOrResetColumn(integ_pv_col_text.c_str());
            std::string integ_tk_col_text = std::string("Integration [Tekl]") + radius_text;
            attributes.insertOrResetColumn(integ_tk_col_text.c_str());
            std::string intensity_col_text = std::string("Intensity") + radius_text;
            attributes.insertOrResetColumn(intensity_col_text.c_str());
            std::string harmonic_col_text = std::string("Harmonic Mean Depth") + radius_text;
            attributes.insertOrResetColumn(harmonic_col_text.c_str());
        }

        std::string depth_col_text = std::string("Mean Depth") + radius_text;
        attributes.insertOrResetColumn(depth_col_text.c_str());
        std::string count_col_text = std::string("Node Count") + radius_text;
        attributes.insertOrResetColumn(count_col_text.c_str());

        if (!simple_version) {
            std::string rel_entropy_col_text = std::string("Relativised Entropy") + radius_text;
            attributes.insertOrResetColumn(rel_entropy_col_text);
        }

        if (m_weighted_measure_col != -1) {
            std::string w_md_col_text = std::string("Mean Depth [") + weighting_col_text + " Wgt]" + radius_text;
            attributes.insertOrResetColumn(w_md_col_text.c_str());
            std::string total_weight_text = std::string("Total ") + weighting_col_text + radius_text;
            attributes.insertOrResetColumn(total_weight_text.c_str());
        }
        if (m_fulloutput) {
            if (!simple_version) {
                std::string penn_norm_text = std::string("RA [Penn]") + radius_text;
                attributes.insertOrResetColumn(penn_norm_text);
            }
            std::string ra_col_text = std::string("RA") + radius_text;
            attributes.insertOrResetColumn(ra_col_text.c_str());

            if (!simple_version) {
                std::string rra_col_text = std::string("RRA") + radius_text;
                attributes.insertOrResetColumn(rra_col_text.c_str());
            }

            std::string td_col_text = std::string("Total Depth") + radius_text;
            attributes.insertOrResetColumn(td_col_text.c_str());
        }
        //
    }
    if (m_local) {
        if (!simple_version) {
            attributes.insertOrResetColumn("Control");
            attributes.insertOrResetColumn("Controllability");
        }
    }
    // then look up all the columns... eek:
    std::vector<int> choice_col, n_choice_col, w_choice_col, nw_choice_col, entropy_col, integ_dv_col, integ_pv_col,
        integ_tk_col, intensity_col, depth_col, count_col, rel_entropy_col, penn_norm_col, w_depth_col,
        total_weight_col, ra_col, rra_col, td_col, harmonic_col;
    for (int radius : radii) {
        std::string radius_text;
        if (radius != -1) {
            radius_text = std::string(" R") + dXstring::formatString(int(radius), "%d");
        }
        if (m_choice) {
            std::string choice_col_text = std::string("Choice") + radius_text;
            choice_col.push_back(attributes.getColumnIndex(choice_col_text.c_str()));
            std::string n_choice_col_text = std::string("Choice [Norm]") + radius_text;
            n_choice_col.push_back(attributes.getColumnIndex(n_choice_col_text.c_str()));
            if (m_weighted_measure_col != -1) {
                std::string w_choice_col_text = std::string("Choice [") + weighting_col_text + " Wgt]" + radius_text;
                w_choice_col.push_back(attributes.getColumnIndex(w_choice_col_text.c_str()));
                std::string nw_choice_col_text =
                    std::string("Choice [") + weighting_col_text + " Wgt][Norm]" + radius_text;
                nw_choice_col.push_back(attributes.getColumnIndex(nw_choice_col_text.c_str()));
            }
        }
        if (!simple_version) {
            std::string entropy_col_text = std::string("Entropy") + radius_text;
            entropy_col.push_back(attributes.getColumnIndex(entropy_col_text.c_str()));
        }

        std::string integ_dv_col_text = std::string("Integration [HH]") + radius_text;
        integ_dv_col.push_back(attributes.getColumnIndex(integ_dv_col_text.c_str()));

        if (!simple_version) {
            std::string integ_pv_col_text = std::string("Integration [P-value]") + radius_text;
            integ_pv_col.push_back(attributes.getColumnIndex(integ_pv_col_text.c_str()));
            std::string integ_tk_col_text = std::string("Integration [Tekl]") + radius_text;
            integ_tk_col.push_back(attributes.getColumnIndex(integ_tk_col_text.c_str()));
            std::string intensity_col_text = std::string("Intensity") + radius_text;
            intensity_col.push_back(attributes.getColumnIndex(intensity_col_text.c_str()));
            std::string harmonic_col_text = std::string("Harmonic Mean Depth") + radius_text;
            harmonic_col.push_back(attributes.getColumnIndex(harmonic_col_text.c_str()));
        }

        std::string depth_col_text = std::string("Mean Depth") + radius_text;
        depth_col.push_back(attributes.getColumnIndex(depth_col_text.c_str()));
        std::string count_col_text = std::string("Node Count") + radius_text;
        count_col.push_back(attributes.getColumnIndex(count_col_text.c_str()));

        if (!simple_version) {
            std::string rel_entropy_col_text = std::string("Relativised Entropy") + radius_text;
            rel_entropy_col.push_back(attributes.getColumnIndex(rel_entropy_col_text.c_str()));
        }

        if (m_weighted_measure_col != -1) {
            std::string w_md_col_text = std::string("Mean Depth [") + weighting_col_text + " Wgt]" + radius_text;
            w_depth_col.push_back(attributes.getColumnIndex(w_md_col_text.c_str()));
            std::string total_weight_col_text = std::string("Total ") + weighting_col_text + radius_text;
            total_weight_col.push_back(attributes.getColumnIndex(total_weight_col_text.c_str()));
        }
        if (m_fulloutput) {
            std::string ra_col_text = std::string("RA") + radius_text;
            ra_col.push_back(attributes.getColumnIndex(ra_col_text.c_str()));

            if (!simple_version) {
                std::string penn_norm_text = std::string("RA [Penn]") + radius_text;
                penn_norm_col.push_back(attributes.getColumnIndex(penn_norm_text));
                std::string rra_col_text = std::string("RRA") + radius_text;
                rra_col.push_back(attributes.getColumnIndex(rra_col_text.c_str()));
            }

            std::string td_col_text = std::string("Total Depth") + radius_text;
            td_col.push_back(attributes.getColumnIndex(td_col_text.c_str()));
        }
    }
    int control_col = -1, controllability_col = -1;
    if (m_local) {
        if (!simple_version) {
            control_col = attributes.getColumnIndex("Control");
            controllability_col = attributes.getColumnIndex("Controllability");
        }
    }

    // ---- from here on: upstream's main loop and choice output, run in parallel ----

    // Upstream searches from line i for the i-th row of the attribute table.
    std::vector<AttributeRow *> rows;
    for (auto &iter : attributes) {
        rows.push_back(&iter.getRow());
    }
    const int nrows = int(rows.size());
    const size_t nlines = map.getShapeCount();
    const size_t nradii = radii.size();
    const bool weighted = m_weighted_measure_col != -1;

    AxialSetup setup{map.getConnections(), nlines, radii, weights, weighted, m_choice, m_local, m_fulloutput,
                     simple_version};
    AxialColumns cols{choice_col,     n_choice_col,  w_choice_col, nw_choice_col, entropy_col, integ_dv_col,
                      integ_pv_col,   integ_tk_col,  intensity_col, depth_col,    count_col,   rel_entropy_col,
                      penn_norm_col,  w_depth_col,   total_weight_col, ra_col,    rra_col,     td_col,
                      harmonic_col,   control_col,   controllability_col};

    // With choice, upstream draws pafrand() once per line taken from the search front.
    // The number of draws per source does not depend on the draws themselves, so count
    // them first, then give every source the generator state it would start from.
    std::vector<uint64_t> rngStart(size_t(nrows), 0);
    uint64_t rngEnd = dmxaccel::pafrandState();
    if (m_choice) {
        std::vector<int64_t> draws(size_t(nrows), 0);
#pragma omp parallel
        {
            AxialSearch counter(setup, cols);
#pragma omp for schedule(dynamic, 16)
            for (int i = 0; i < nrows; i++) {
                draws[size_t(i)] = counter.countDraws(i);
            }
        }
        for (int i = 0; i < nrows; i++) {
            rngStart[size_t(i)] = rngEnd;
            rngEnd = dmxaccel::pafrandAdvance(rngEnd, uint64_t(draws[size_t(i)]));
        }
    }

    const int threads = dmxaccel::cpuThreadCount();
    std::vector<std::vector<double>> threadChoice(static_cast<size_t>(threads));
    std::vector<double> weighted_choice(m_choice && weighted ? nlines * nradii : 0, 0.0);
    const int batchSize = 8 * threads;
    std::vector<AxialResult> results(static_cast<size_t>(batchSize));
    for (int start = 0; start < nrows; start += batchSize) {
        const int count = std::min(nrows, start + batchSize) - start;
#pragma omp parallel
        {
#ifdef _OPENMP
            const int tid = omp_get_thread_num();
#else
            const int tid = 0;
#endif
            AxialSearch search(setup, cols);
            if (!threadChoice[size_t(tid)].empty()) {
                search.choice().swap(threadChoice[size_t(tid)]); // continue this thread's sums
            }
#pragma omp for schedule(dynamic, 1)
            for (int k = 0; k < count; k++) {
                search.run(start + k, rngStart[size_t(start + k)], results[size_t(k)]);
            }
            threadChoice[size_t(tid)].swap(search.choice());
        }
        // Apply in upstream's order: row writes (they also update column statistics) and
        // weighted choice additions.
        for (int k = 0; k < count; k++) {
            const AxialResult &res = results[size_t(k)];
            AttributeRow &row = *rows[size_t(start + k)];
            for (const auto &w : res.writes) {
                row.setValue(size_t(w.first), w.second);
            }
            for (const auto &add : res.weighted_log) {
                weighted_choice[add.first] += add.second;
            }
        }
        if (comm) {
            if (qtimer(atime, 500)) {
                if (comm->IsCancelled()) {
                    throw Communicator::CancelledException();
                }
                comm->CommPostMessage(Communicator::CURRENT_RECORD, start + count);
            }
        }
    }
    if (m_choice) {
        dmxaccel::setPafrandState(rngEnd); // as upstream leaves the generator
    }
    dmxaccel::addReport("Axial integration: CPU, " + std::to_string(threads) + " threads, " +
                        std::to_string(nlines) + " lines");

    if (m_choice) {
        std::vector<double> choice(nlines * nradii, 0.0); // integer-valued sums: exact in any order
        for (const auto &c : threadChoice) {
            for (size_t j = 0; j < c.size(); j++) {
                choice[j] += c[j];
            }
        }
        size_t i = size_t(-1);
        for (auto &iter : attributes) {
            i++;
            AttributeRow &row = iter.getRow();
            double total_choice = 0.0, w_total_choice = 0.0;
            for (size_t r = 0; r < radii.size(); r++) {
                total_choice += choice[i * nradii + r];
                if (weighted) {
                    w_total_choice += weighted_choice[i * nradii + r];
                }
                // n.b., normalise choice according to (n-1)(n-2)/2 (maximum possible through routes)
                double node_count = row.getValue(count_col[r]);
                double total_weight = 0;
                if (m_weighted_measure_col != -1) {
                    total_weight = row.getValue(total_weight_col[r]);
                }
                if (node_count > 2) {
                    row.setValue(choice_col[r], float(total_choice));
                    row.setValue(n_choice_col[r], float(2.0 * total_choice / ((node_count - 1) * (node_count - 2))));
                    if (m_weighted_measure_col != -1) {
                        row.setValue(w_choice_col[r], float(w_total_choice));
                        row.setValue(nw_choice_col[r], float(2.0 * w_total_choice / (total_weight * total_weight)));
                    }
                } else {
                    row.setValue(choice_col[r], -1);
                    row.setValue(n_choice_col[r], -1);
                    if (m_weighted_measure_col != -1) {
                        row.setValue(w_choice_col[r], -1);
                        row.setValue(nw_choice_col[r], -1);
                    }
                }
            }
        }
    }

    map.setDisplayedAttribute(-1); // <- override if it's already showing
    map.setDisplayedAttribute(integ_dv_col.back());

    return true;
}

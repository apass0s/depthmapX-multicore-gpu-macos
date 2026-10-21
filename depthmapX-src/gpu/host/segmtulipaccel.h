// depthmapX GPU module - segment tulip (angular) analysis on all CPU cores

#pragma once

#include "gpu/host/accelcommon.h"

#include "salalib/isegment.h"

#include <set>

// Produces exactly the attributes of SegmentTulip (salalib/segmmodules) as called by
// MetaGraph::analyseSegmentsTulip (no route weights, no second weight column): the same
// per-segment searches, run in parallel.
class SegmentTulipAccel : ISegment {
  private:
    std::set<double> m_radius_set;
    bool m_sel_only;
    int m_tulip_bins;
    int m_weighted_measure_col;
    int m_radius_type;
    bool m_choice;

  public:
    std::string getAnalysisName() const override { return "Tulip Analysis"; }
    bool run(Communicator *comm, ShapeGraph &map, bool) override;
    SegmentTulipAccel(std::set<double> radius_set, bool sel_only, int tulip_bins, int weighted_measure_col,
                      int radius_type, bool choice)
        : m_radius_set(radius_set), m_sel_only(sel_only), m_tulip_bins(tulip_bins),
          m_weighted_measure_col(weighted_measure_col), m_radius_type(radius_type), m_choice(choice) {}
};

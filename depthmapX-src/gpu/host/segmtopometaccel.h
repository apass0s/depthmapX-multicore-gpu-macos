// depthmapX GPU module - segment topological / metric analysis on all CPU cores

#pragma once

#include "gpu/host/accelcommon.h"

#include "salalib/isegment.h"

// Produces exactly the attributes of SegmentTopological (metric = false) or SegmentMetric
// (metric = true) from salalib/segmmodules: the same per-segment searches, in parallel.
class SegmentTopoMetAccel : ISegment {
  private:
    bool m_metric;
    double m_radius;
    bool m_sel_only;

  public:
    std::string getAnalysisName() const override {
        return m_metric ? "Metric Analysis" : "Topological Analysis";
    }
    bool run(Communicator *comm, ShapeGraph &map, bool) override;
    SegmentTopoMetAccel(bool metric, double radius, bool sel_only)
        : m_metric(metric), m_radius(radius), m_sel_only(sel_only) {}
};

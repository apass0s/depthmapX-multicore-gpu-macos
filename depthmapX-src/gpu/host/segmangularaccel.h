// depthmapX GPU module - segment angular analysis on all CPU cores

#pragma once

#include "gpu/host/accelcommon.h"

#include "salalib/isegment.h"

#include <set>

// Produces exactly the attributes of SegmentAngular (salalib/segmmodules): the same
// per-segment searches, run in parallel.
class SegmentAngularAccel : ISegment {
  private:
    std::set<double> m_radius_set;

  public:
    std::string getAnalysisName() const override { return "Angular Analysis"; }
    bool run(Communicator *comm, ShapeGraph &map, bool) override;
    SegmentAngularAccel(std::set<double> radius_set) : m_radius_set(radius_set) {}
};

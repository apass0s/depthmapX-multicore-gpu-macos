// depthmapX GPU module - exact axial choice (Brandes betweenness) on all CPU cores

#pragma once

#include "gpu/host/accelcommon.h"

#include "salalib/iaxial.h"

#include <set>

// "Choice [Brandes]" (plus " R<n>" per finite radius), as in the rapido build's -xacb:
// Brandes betweenness of each axial line over all ordered pairs of other lines, counting
// every shortest path (unlike upstream's Choice, which follows one random shortest path
// per pair). With a radius, searches stop at that depth. Per-source contributions are
// added in source order, so results do not depend on the number of threads.
class AxialBrandesChoice : IAxial {
  private:
    std::set<double> m_radius_set;

  public:
    std::string getAnalysisName() const override { return "Brandes Choice"; }
    bool run(Communicator *comm, ShapeGraph &map, bool) override;
    AxialBrandesChoice(std::set<double> radius_set) : m_radius_set(radius_set) {}
};

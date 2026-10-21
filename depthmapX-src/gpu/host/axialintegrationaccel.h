// depthmapX GPU module - axial integration (and choice) on all CPU cores

#pragma once

#include "gpu/host/accelcommon.h"

#include "salalib/iaxial.h"

#include <set>

// Produces exactly the attributes of AxialIntegration (salalib/axialmodules): the same
// per-line searches, run in parallel (including the random choice of the next line).
class AxialIntegrationAccel : IAxial {
  private:
    std::set<double> m_radius_set;
    int m_weighted_measure_col;
    bool m_choice;
    bool m_fulloutput;
    bool m_local;

  public:
    std::string getAnalysisName() const override { return "Angular Analysis"; }
    bool run(Communicator *, ShapeGraph &map, bool) override;
    AxialIntegrationAccel(std::set<double> radius_set, int weighted_measure_col, bool choice, bool fulloutput,
                          bool local)
        : m_radius_set(radius_set), m_weighted_measure_col(weighted_measure_col), m_choice(choice),
          m_fulloutput(fulloutput), m_local(local) {}
};

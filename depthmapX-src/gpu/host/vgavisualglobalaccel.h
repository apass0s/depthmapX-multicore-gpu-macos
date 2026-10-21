// depthmapX GPU module - VGA visual global measures on the GPU or on all CPU cores

#pragma once

#include "gpu/host/accelcommon.h"

#include "salalib/ivga.h"
#include "salalib/pointdata.h"

#include <utility>
#include <vector>

// Produces exactly the attributes (and the left-over point state) of VGAVisualGlobal
// (salalib/vgamodules), computed with the CUDA module when requested and possible,
// otherwise with OpenMP. With several radii the result equals separate runs, one per
// radius in the given order, but the map is searched once.
class VGAVisualGlobalAccel : IVGA {
  private:
    std::vector<double> m_radii;
    bool m_gates_only;
    dmxaccel::AccelSettings m_settings;

  public:
    std::string getAnalysisName() const override { return "Global Visibility Analysis"; }
    bool run(Communicator *comm, PointMap &map, bool simple_version) override;
    VGAVisualGlobalAccel(std::vector<double> radii, bool gates_only, const dmxaccel::AccelSettings &settings)
        : m_radii(std::move(radii)), m_gates_only(gates_only), m_settings(settings) {}
};

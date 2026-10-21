// depthmapX GPU module - VGA visual local measures on the GPU or on all CPU cores

#pragma once

#include "gpu/host/accelcommon.h"

#include "salalib/ivga.h"
#include "salalib/pointdata.h"

// Produces exactly the attributes of VGAVisualLocal (salalib/vgamodules), computed
// with the CUDA module when requested and possible, otherwise with OpenMP.
class VGAVisualLocalAccel : IVGA {
  private:
    bool m_gates_only;
    dmxaccel::AccelSettings m_settings;

  public:
    std::string getAnalysisName() const override { return "Local Visibility Analysis"; }
    bool run(Communicator *comm, PointMap &map, bool simple_version) override;
    VGAVisualLocalAccel(bool gates_only, const dmxaccel::AccelSettings &settings)
        : m_gates_only(gates_only), m_settings(settings) {}
};

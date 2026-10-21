// depthmapX GPU module - VGA metric and angular measures on all CPU cores (and the GPU)

#pragma once

#include "gpu/host/accelcommon.h"

#include "salalib/ivga.h"
#include "salalib/pointdata.h"

// Produce exactly the attributes (and left-over point state) of VGAMetric / VGAAngular
// (salalib/vgamodules): the same per-source searches, run in parallel.
class VGAMetricAccel : IVGA {
  private:
    double m_radius;
    bool m_gates_only;
    dmxaccel::AccelSettings m_settings;

  public:
    std::string getAnalysisName() const override { return "Metric Analysis"; }
    bool run(Communicator *comm, PointMap &map, bool simple_version) override;
    VGAMetricAccel(double radius, bool gates_only, const dmxaccel::AccelSettings &settings)
        : m_radius(radius), m_gates_only(gates_only), m_settings(settings) {}
};

class VGAAngularAccel : IVGA {
  private:
    double m_radius;
    bool m_gates_only;
    dmxaccel::AccelSettings m_settings;

  public:
    std::string getAnalysisName() const override { return "Angular Analysis"; }
    bool run(Communicator *comm, PointMap &map, bool simple_version) override;
    VGAAngularAccel(double radius, bool gates_only, const dmxaccel::AccelSettings &settings)
        : m_radius(radius), m_gates_only(gates_only), m_settings(settings) {}
};

// depthmapX GPU module - run-time loader for depthmapX_gpu.dll

#pragma once

#include "gpu/include/dmx_gpu_api.h"

#include <string>

namespace dmxaccel {

// Function table of the GPU DLL. Loaded once, on first use, from the directory of the
// running executable (never from the current directory or PATH).
class GpuModule {
  public:
    static GpuModule &instance();

    // True when the DLL loaded, its API version matches and the device probe passed.
    bool available() const { return m_available; }
    // Why the GPU cannot be used (empty when available()).
    const std::string &unavailableReason() const { return m_reason; }
    const dmx_gpu_device_info &device() const { return m_device; }
    std::string deviceDescription() const;

    // Message of the last failed call on this thread.
    std::string lastError() const;

    decltype(&dmx_vis_local_create) visLocalCreate = nullptr;
    decltype(&dmx_vis_local_run) visLocalRun = nullptr;
    decltype(&dmx_vis_local_destroy) visLocalDestroy = nullptr;
    decltype(&dmx_vis_global_create) visGlobalCreate = nullptr;
    decltype(&dmx_vis_global_lanes) visGlobalLanes = nullptr;
    decltype(&dmx_vis_global_run) visGlobalRun = nullptr;
    decltype(&dmx_vis_global_destroy) visGlobalDestroy = nullptr;

  private:
    GpuModule();
    bool load();

    void *m_handle = nullptr;
    bool m_available = false;
    std::string m_reason;
    dmx_gpu_device_info m_device{};

    decltype(&dmx_gpu_api_version) m_apiVersion = nullptr;
    decltype(&dmx_gpu_last_error) m_lastError = nullptr;
    decltype(&dmx_gpu_probe) m_probe = nullptr;
};

} // namespace dmxaccel

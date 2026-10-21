// depthmapX GPU module (Metal) - the device layer under the C interface
//
// dmx_metal_api.cpp implements dmx_gpu_api.h (argument checks, memory planning, the
// dispatch sequence of each analysis) against this small interface, which mirrors how
// Metal encodes work: bind buffers / bytes / threadgroup memory by index, dispatch a
// kernel as a grid of threadgroups, commit the command buffer. Two implementations:
//   dmx_metal_device.mm     Metal (macOS)
//   emu/dmx_metal_emu.cpp   runs the same kernel source on CPU threads, to test the
//                           module on machines without Metal (never shipped)

#pragma once

#include "gpu/include/dmx_gpu_api.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace dmxmetal {

enum class Kernel { Probe, VisLocal, VisGlobalPull, VisGlobalDownsweep, VisGlobalCombine, VisGlobalCount };
constexpr int kKernelCount = 6;

struct Buffer; // device buffer in memory shared with the CPU

class Device {
  public:
    virtual ~Device() = default;

    // Name, GPU family (cc_major: newest Apple family, 0 for Mac-family GPUs) and memory.
    virtual void describe(dmx_gpu_device_info &info) const = 0;
    virtual uint64_t workingSetBytes() const = 0; // memory the GPU can use without paging
    virtual uint64_t allocatedBytes() const = 0;  // already allocated by this process
    virtual uint64_t maxBufferBytes() const = 0;
    virtual uint32_t maxThreadgroupMemory() const = 0;
    virtual uint32_t maxThreadsPerThreadgroup(Kernel kernel) const = 0;
    // Threadgroups per dispatch along x: larger 1D grids are split. Metal documents no
    // per-dimension limit we could confirm, so the default is the most conservative one.
    virtual uint32_t maxGroupsPerDispatch() const { return 65535; }

    // Buffer whose contents the CPU reads and writes directly (no copies). Returns null
    // and sets error on failure. Contents are undefined until written.
    virtual Buffer *allocate(size_t bytes, std::string &error) = 0;
    virtual void release(Buffer *buffer) = 0;
    virtual void *contents(Buffer *buffer) = 0;

    // Recording: begin() opens a command buffer with one serial compute encoder (each
    // dispatch sees the writes of the previous ones); bindings stay set until replaced.
    virtual bool begin(std::string &error) = 0;
    virtual void setBuffer(Buffer *buffer, size_t offset, int index) = 0;
    virtual void setBytes(const void *bytes, size_t length, int index) = 0;
    virtual void setThreadgroupMemory(size_t length, int index) = 0;
    virtual void dispatch(Kernel kernel, uint32_t groupsX, uint32_t groupsY, uint32_t threadsPerGroup) = 0;
    // Ends the encoder and commits; with wait, blocks until the GPU finished and reports
    // its errors. Command buffers run in commit order.
    virtual bool commit(bool wait, std::string &error) = 0;
    // Blocks until every committed command buffer finished.
    virtual bool waitAll(std::string &error) = 0;
};

// Defined by the backend: creates the device (compiling the kernels). Null on failure,
// with *code a DMX_GPU_ERR_* value and error set.
std::unique_ptr<Device> createDevice(int *code, std::string &error);

} // namespace dmxmetal

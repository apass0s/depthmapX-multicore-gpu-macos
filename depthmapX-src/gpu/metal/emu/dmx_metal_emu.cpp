// depthmapX GPU module (Metal) - CPU emulator of the device layer (tests only)
//
// Runs dmx_metal_kernels.metal, compiled as C++, on CPU threads behind the same device
// interface as Metal, so the whole Metal module (dmx_metal_api.cpp + kernels) can be
// tested where there is no Mac: built as depthmapX_gpu.dll / libdepthmapX_gpu.so it is
// loaded by depthmapX like the real GPU module and its results compared with the CPU.
//
// Faithful where it matters for correctness:
//  - the threads of a threadgroup run concurrently and threadgroup_barrier is a real
//    barrier, so a missing barrier can show up as wrong or varying results;
//  - threadgroups run in parallel on all cores and atomics are real atomics;
//  - dispatches that exceed the threads-per-threadgroup or threadgroup-memory limits,
//    or miss a binding, fail as Metal validation would;
//  - new buffers are filled with junk, as Metal does not clear them.
// Not emulated: the Metal compiler (Metal-only syntax errors), GPU memory ordering
// beyond the barriers and atomics, timing.
//
// Limits (environment, for tests): DMX_METAL_EMU_THREADS (max threads per threadgroup,
// default 256), DMX_METAL_EMU_TGMEM (threadgroup memory bytes, default 32768),
// DMX_METAL_EMU_WORKING_SET_MB (default 8192), DMX_METAL_EMU_MAX_BUFFER_MB (default 4096);
// DMX_METAL_EMU_MAX_GROUPS (threadgroups per dispatch, default 65535);
// DMX_METAL_EMU_TRACE=1 prints every buffer allocation to stderr.

#include "../dmx_metal_backend.h"

#include <atomic>
#include <barrier>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

// The kernels, with the Metal names of emu/metal_shim.h; nothing below may use the
// identifiers kernel, device, threadgroup or constant.
#include "../dmx_metal_kernels.metal"

namespace dmxemu {
thread_local void (*t_barrier)() = nullptr;
}

namespace dmxmetal {

struct Buffer {
    std::vector<unsigned char> data;
};

namespace {

uint64_t envNumber(const char *name, uint64_t fallback) {
    const char *v = std::getenv(name);
    if (!v || !*v) {
        return fallback;
    }
    char *end = nullptr;
    unsigned long long n = std::strtoull(v, &end, 10);
    return (end && *end == '\0' && n > 0) ? uint64_t(n) : fallback;
}

thread_local std::barrier<> *t_groupBarrier = nullptr;
void groupBarrier() { t_groupBarrier->arrive_and_wait(); }
void noBarrier() { std::terminate(); } // a kernel without threadgroup memory must not call it

class EmuDevice : public Device {
  public:
    EmuDevice()
        : m_maxThreads(uint32_t(envNumber("DMX_METAL_EMU_THREADS", 256))),
          m_tgMemory(uint32_t(envNumber("DMX_METAL_EMU_TGMEM", 32768))),
          m_workingSet(envNumber("DMX_METAL_EMU_WORKING_SET_MB", 8192) << 20),
          m_maxBuffer(envNumber("DMX_METAL_EMU_MAX_BUFFER_MB", 4096) << 20),
          m_maxGroups(uint32_t(envNumber("DMX_METAL_EMU_MAX_GROUPS", 65535))),
          m_trace(envNumber("DMX_METAL_EMU_TRACE", 0) != 0) {}

    void describe(dmx_gpu_device_info &info) const override {
        std::snprintf(info.name, sizeof(info.name), "Metal kernels on CPU threads (emulator, %u threads/group)",
                      m_maxThreads);
        info.cc_major = 0;
        info.cc_minor = 0;
        info.total_mem = m_workingSet;
        info.free_mem = m_workingSet > m_allocated ? m_workingSet - m_allocated : 0;
    }
    uint64_t workingSetBytes() const override { return m_workingSet; }
    uint64_t allocatedBytes() const override { return m_allocated; }
    uint64_t maxBufferBytes() const override { return m_maxBuffer; }
    uint32_t maxThreadgroupMemory() const override { return m_tgMemory; }
    uint32_t maxThreadsPerThreadgroup(Kernel) const override { return m_maxThreads; }
    uint32_t maxGroupsPerDispatch() const override { return m_maxGroups; }

    Buffer *allocate(size_t bytes, std::string &error) override {
        if (bytes > m_maxBuffer) {
            error = "buffer larger than the maximum buffer length";
            return nullptr;
        }
        auto *b = new Buffer();
        b->data.assign(bytes, 0xA5); // Metal does not clear new buffers
        if (m_trace) {
            std::fprintf(stderr, "metal-emu: buffer of %zu bytes\n", bytes);
        }
        m_allocated += bytes;
        return b;
    }
    void release(Buffer *b) override {
        m_allocated -= b->data.size();
        delete b;
    }
    void *contents(Buffer *b) override { return b->data.data(); }

    bool begin(std::string &error) override {
        if (m_open) {
            error = "begin() while a command buffer is open";
            return false;
        }
        m_open = true;
        m_failure.clear();
        for (auto &s : m_slots) {
            s = Slot();
        }
        for (auto &t : m_tgLength) {
            t = 0;
        }
        return true;
    }
    void setBuffer(Buffer *b, size_t offset, int index) override {
        Slot &s = slot(index);
        s.buffer = b;
        s.offset = offset;
        s.bytes.clear();
    }
    void setBytes(const void *bytes, size_t length, int index) override {
        Slot &s = slot(index);
        s.buffer = nullptr;
        s.offset = 0;
        s.bytes.assign(static_cast<const unsigned char *>(bytes), static_cast<const unsigned char *>(bytes) + length);
        if (length > 4096) {
            fail("setBytes over 4 KiB");
        }
    }
    void setThreadgroupMemory(size_t length, int index) override {
        if (index < 0 || index >= kTgSlots || length % 16 != 0) {
            fail("threadgroup memory index out of range or length not a multiple of 16");
            return;
        }
        m_tgLength[index] = length;
    }
    void dispatch(Kernel k, uint32_t groupsX, uint32_t groupsY, uint32_t threads) override {
        if (!m_open) {
            fail("dispatch outside a command buffer");
        }
        if (!m_failure.empty() || groupsX == 0 || groupsY == 0) {
            return;
        }
        if (threads == 0 || threads > m_maxThreads) {
            fail("threads per threadgroup above the pipeline limit");
            return;
        }
        if (groupsX > m_maxGroups) {
            fail("threadgroups per dispatch above the device limit");
            return;
        }
        try {
            run(k, groupsX, groupsY, threads);
        } catch (const std::exception &e) {
            fail(e.what());
        }
    }
    bool commit(bool, std::string &error) override {
        m_open = false;
        if (!m_failure.empty()) {
            error = m_failure;
            m_pendingFailure = m_failure;
            return false;
        }
        return true;
    }
    bool waitAll(std::string &error) override {
        if (!m_pendingFailure.empty()) {
            error = m_pendingFailure;
            m_pendingFailure.clear();
            return false;
        }
        return true;
    }

  private:
    static constexpr int kSlots = 8, kTgSlots = 4;
    struct Slot {
        Buffer *buffer = nullptr;
        size_t offset = 0;
        std::vector<unsigned char> bytes;
    };

    Slot &slot(int index) {
        if (index < 0 || index >= kSlots) {
            fail("buffer index out of range");
            index = 0;
        }
        return m_slots[index];
    }
    void fail(const std::string &why) {
        if (m_failure.empty()) {
            m_failure = why;
        }
    }
    template <typename T> T *arg(int index) {
        Slot &s = m_slots[index];
        if (s.buffer) {
            if (s.offset > s.buffer->data.size() || s.offset % alignof(T) != 0) {
                throw std::runtime_error("buffer offset out of range or misaligned");
            }
            return reinterpret_cast<T *>(s.buffer->data.data() + s.offset);
        }
        if (s.bytes.size() >= sizeof(T)) {
            return reinterpret_cast<T *>(s.bytes.data());
        }
        throw std::runtime_error("kernel argument " + std::to_string(index) + " not bound");
    }

    // Kernels without threadgroup memory: every thread independently, groups in parallel.
    template <typename Fn> void forEachThread(uint32_t groupsX, uint32_t groupsY, uint32_t threads, Fn fn) {
        const uint64_t groups = uint64_t(groupsX) * groupsY;
        std::atomic<uint64_t> next(0);
        auto worker = [&]() {
            dmxemu::t_barrier = noBarrier;
            for (uint64_t g; (g = next.fetch_add(1)) < groups;) {
                const uint2 group{uint(g % groupsX), uint(g / groupsX)};
                for (uint32_t tid = 0; tid < threads; ++tid) {
                    fn(group, tid);
                }
            }
        };
        std::vector<std::thread> pool;
        const unsigned n = std::max(1u, std::thread::hardware_concurrency());
        for (unsigned i = 1; i < n; ++i) {
            pool.emplace_back(worker);
        }
        worker();
        for (auto &t : pool) {
            t.join();
        }
    }

    void run(Kernel k, uint32_t groupsX, uint32_t groupsY, uint32_t threads) {
        size_t tgTotal = 0;
        for (size_t len : m_tgLength) {
            tgTotal += len;
        }
        if (tgTotal > m_tgMemory) {
            throw std::runtime_error("threadgroup memory above the device limit");
        }
        switch (k) {
        case Kernel::Probe: {
            int *out = arg<int>(0);
            forEachThread(groupsX, groupsY, threads,
                          [&](uint2 g, uint32_t tid) { dmx_probe(out, g.x * threads + tid); });
            break;
        }
        case Kernel::VisLocal:
            runVisLocal(groupsX * groupsY, threads);
            break;
        case Kernel::VisGlobalPull: {
            const i64 *keyOffsets = arg<const i64>(0);
            const int *keyNodes = arg<const int>(1);
            const u64 *frontier = arg<const u64>(2);
            u64 *blocks = arg<u64>(3);
            const DmxVisGlobalParams &p = *arg<const DmxVisGlobalParams>(4);
            forEachThread(groupsX, groupsY, threads, [&](uint2 g, uint32_t tid) {
                dmx_vis_global_pull(keyOffsets, keyNodes, frontier, blocks, p, g.x * threads + tid);
            });
            break;
        }
        case Kernel::VisGlobalDownsweep: {
            u64 *blocks = arg<u64>(0);
            const DmxVisGlobalParams &p = *arg<const DmxVisGlobalParams>(1);
            forEachThread(groupsX, groupsY, threads, [&](uint2 g, uint32_t tid) {
                dmx_vis_global_downsweep(blocks, p, g.x * threads + tid);
            });
            break;
        }
        case Kernel::VisGlobalCombine: {
            const u64 *blocks = arg<const u64>(0);
            const int *nodePositions = arg<const int>(1);
            u64 *frontier = arg<u64>(2);
            u64 *visited = arg<u64>(3);
            atomic_int *anyReached = arg<atomic_int>(4);
            const DmxVisGlobalParams &p = *arg<const DmxVisGlobalParams>(5);
            forEachThread(groupsX, groupsY, threads, [&](uint2 g, uint32_t tid) {
                dmx_vis_global_combine(blocks, nodePositions, frontier, visited, anyReached, p, g.x * threads + tid);
            });
            break;
        }
        case Kernel::VisGlobalCount: {
            const u64 *frontier = arg<const u64>(0);
            atomic_int *levelCounts = arg<atomic_int>(1);
            const DmxVisGlobalParams &p = *arg<const DmxVisGlobalParams>(2);
            forEachThread(groupsX, groupsY, threads,
                          [&](uint2 g, uint32_t tid) { dmx_vis_global_count(frontier, levelCounts, p, g, uint2{tid, 0}, uint2{threads, 1}); });
            break;
        }
        }
    }

    // Threadgroups one after another; the `threads` threads of each run concurrently and
    // meet at every threadgroup_barrier, and again before the next group reuses the memory.
    void runVisLocal(uint32_t groups, uint32_t threads) {
        const u64 *rows = arg<const u64>(0);
        const int *pixelToRow = arg<const int>(1);
        const i64 *dupOffsets = arg<const i64>(2);
        const int *dupPixels = arg<const int>(3);
        i64 *clusterAcc = arg<i64>(4);
        int *unionAcc = arg<int>(5);
        const DmxVisLocalParams &p = *arg<const DmxVisLocalParams>(6);
        if (size_t(p.sliceWords) * 8 > m_tgLength[0] || size_t(p.sliceWords) * 8 > m_tgLength[1] ||
            size_t(threads) * 8 > m_tgLength[2] || size_t(threads) * 4 > m_tgLength[3]) {
            throw std::runtime_error("vis_local: threadgroup memory smaller than the kernel uses");
        }
        std::vector<u64> sliceU(m_tgLength[0] / 8, 0xA5A5A5A5A5A5A5A5ull);
        std::vector<u64> sliceUnion(m_tgLength[1] / 8, 0xA5A5A5A5A5A5A5A5ull);
        std::vector<i64> redIntersect(m_tgLength[2] / 8);
        std::vector<int> redUnion(m_tgLength[3] / 4);
        std::barrier<> barrier(threads);
        auto member = [&](uint32_t tid) {
            t_groupBarrier = &barrier;
            dmxemu::t_barrier = groupBarrier;
            for (uint32_t g = 0; g < groups; ++g) {
                dmx_vis_local_rows(rows, pixelToRow, dupOffsets, dupPixels, clusterAcc, unionAcc, p, sliceU.data(),
                              sliceUnion.data(), redIntersect.data(), redUnion.data(), g, tid, threads);
                barrier.arrive_and_wait();
            }
        };
        std::vector<std::thread> team;
        for (uint32_t tid = 1; tid < threads; ++tid) {
            team.emplace_back(member, tid);
        }
        member(0);
        for (auto &t : team) {
            t.join();
        }
    }

    const uint32_t m_maxThreads, m_tgMemory;
    const uint64_t m_workingSet, m_maxBuffer;
    const uint32_t m_maxGroups;
    const bool m_trace;
    uint64_t m_allocated = 0;
    bool m_open = false;
    std::string m_failure, m_pendingFailure;
    Slot m_slots[kSlots];
    size_t m_tgLength[kTgSlots] = {};
};

} // namespace

std::unique_ptr<Device> createDevice(int *code, std::string &) {
    *code = DMX_GPU_OK;
    return std::unique_ptr<Device>(new EmuDevice());
}

} // namespace dmxmetal

// depthmapX GPU module (Metal) - the Metal names used by dmx_metal_kernels.metal, for
// compiling that file as C++ in the CPU emulator (tests only, never shipped).
//
// Address-space qualifiers become nothing (constant: const), attributes are dropped
// (DMX_ATTR), integer built-ins and atomics map to their C++ equivalents, and
// threadgroup_barrier waits on the barrier of the emulated threadgroup the calling
// thread belongs to (set by emu/dmx_metal_emu.cpp).

#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace dmxemu {
// Called by threadgroup_barrier; set per emulated thread.
extern thread_local void (*t_barrier)();
} // namespace dmxemu

typedef uint64_t u64;
typedef int64_t i64;
typedef unsigned int uint;
struct uint2 {
    uint x, y;
};

inline u64 popcount(u64 x) {
#if defined(_MSC_VER)
    return u64(__popcnt64(x));
#else
    return u64(__builtin_popcountll(x));
#endif
}

inline u64 ctz(u64 x) { // x != 0 at every call in the kernels
#if defined(_MSC_VER)
    unsigned long index;
    _BitScanForward64(&index, x);
    return u64(index);
#else
    return u64(__builtin_ctzll(x));
#endif
}

using std::min;

typedef std::atomic<int> atomic_int;
using std::atomic_fetch_add_explicit;
using std::atomic_store_explicit;
using std::memory_order_relaxed;

namespace mem_flags {
constexpr int mem_none = 0, mem_device = 1, mem_threadgroup = 2;
}
inline void threadgroup_barrier(int) { dmxemu::t_barrier(); }

#define DMX_ATTR(...)
#define kernel
#define device
#define threadgroup
#define constant const

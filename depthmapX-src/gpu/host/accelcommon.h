// depthmapX GPU module - settings, run reports and bit helpers shared by the
// accelerated (multicore CPU / CUDA) analysis paths

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace dmxaccel {

enum class ComputeMode {
    CpuParallel, // multicore CPU (OpenMP), the default
    Gpu,         // CUDA when available and the map is large enough, otherwise multicore CPU
    Serial       // the original single-threaded upstream code
};

// Maps smaller than this run on the CPU even when the GPU is requested: below it the
// CUDA set-up and transfers cost more than they save.
constexpr int kDefaultGpuMinNodes = 10000;

struct AccelSettings {
    ComputeMode mode = ComputeMode::CpuParallel;
    int gpuMinNodes = kDefaultGpuMinNodes;
};

// One line per accelerated analysis run, saying which path ran and why. The CLI prints
// these after the analysis; the GUI shows them in the status bar.
void addReport(const std::string &line);
std::vector<std::string> takeReports();

int cpuThreadCount();
uint64_t physicalMemoryBytes();

// genlib's pafrand() (set 0): a 64-bit linear congruential generator with global state.
// These reproduce it so that parallel searches can draw exactly the numbers the serial
// code would have drawn.
constexpr uint64_t kPafrandMult = 0x71A7FA85; // genlib/pafmath.cpp g_mult
constexpr uint64_t kPafrandConst = 0xF5E958B9; // genlib/pafmath.cpp g_const
inline uint64_t pafrandStep(uint64_t state) { return kPafrandMult * state + kPafrandConst; }
inline unsigned int pafrandValue(uint64_t state) { return (unsigned int)((state >> 32) & 0x0FFFFFFF); }
uint64_t pafrandAdvance(uint64_t state, uint64_t steps); // state after `steps` draws, O(log steps)
uint64_t pafrandState();
void setPafrandState(uint64_t state);
// Checks against pafrand() itself that the model above draws the same numbers (the
// generator state is restored afterwards).
bool pafrandModelOk();

inline int popcount64(uint64_t x) {
#if defined(_MSC_VER)
    return int(__popcnt64(x));
#else
    return __builtin_popcountll(x);
#endif
}

inline int lowestBit64(uint64_t x) { // x must be non-zero
#if defined(_MSC_VER)
    unsigned long index;
    _BitScanForward64(&index, x);
    return int(index);
#else
    return __builtin_ctzll(x);
#endif
}

} // namespace dmxaccel

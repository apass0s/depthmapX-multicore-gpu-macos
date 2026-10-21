// depthmapX GPU module - settings, run reports and bit helpers

#include "gpu/host/accelcommon.h"

#include <mutex>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

#include "genlib/pafmath.h"

extern uint64_t g_rand[11]; // genlib/pafmath.cpp: pafrand() state per set

namespace dmxaccel {

uint64_t pafrandAdvance(uint64_t state, uint64_t steps) {
    // x -> a x + c, applied `steps` times, by squaring the affine map (mod 2^64)
    uint64_t mult = 1, add = 0;
    uint64_t curMult = kPafrandMult, curAdd = kPafrandConst;
    while (steps) {
        if (steps & 1) {
            mult = curMult * mult;
            add = curMult * add + curAdd;
        }
        curAdd = (curMult + 1) * curAdd;
        curMult = curMult * curMult;
        steps >>= 1;
    }
    return mult * state + add;
}

uint64_t pafrandState() { return g_rand[0]; }

void setPafrandState(uint64_t state) { g_rand[0] = state; }

bool pafrandModelOk() {
    const uint64_t saved = g_rand[0];
    uint64_t model = saved;
    bool ok = true;
    for (int i = 0; i < 8 && ok; i++) {
        const unsigned int drawn = pafrand(0);
        model = pafrandStep(model);
        ok = drawn == pafrandValue(model) && g_rand[0] == model;
    }
    ok = ok && pafrandAdvance(saved, 8) == model;
    g_rand[0] = saved;
    return ok;
}

namespace {
std::mutex g_reportMutex;
std::vector<std::string> g_reports;
} // namespace

void addReport(const std::string &line) {
    std::lock_guard<std::mutex> lock(g_reportMutex);
    g_reports.push_back(line);
}

std::vector<std::string> takeReports() {
    std::lock_guard<std::mutex> lock(g_reportMutex);
    std::vector<std::string> out;
    out.swap(g_reports);
    return out;
}

int cpuThreadCount() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

uint64_t physicalMemoryBytes() {
#ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) {
        return status.ullTotalPhys;
    }
    return 0;
#else
    long pages = sysconf(_SC_PHYS_PAGES);
    long pageSize = sysconf(_SC_PAGE_SIZE);
    return (pages > 0 && pageSize > 0) ? uint64_t(pages) * uint64_t(pageSize) : 0;
#endif
}

} // namespace dmxaccel

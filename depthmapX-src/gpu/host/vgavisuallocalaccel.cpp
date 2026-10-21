// depthmapX GPU module - VGA visual local measures on the GPU or on all CPU cores
//
// Upstream (salalib/vgamodules/vgavisuallocal.cpp), for every source point u with
// neighbourhood N(u) (the de-duplicated pixels of u's bins) computes, over the
// neighbours v that are filled points with a node:
//   cluster  = sum_v (pixels visited by v's iterator that are in N(u))
//   control  = sum_v 1.0f / float(pixels visited by v's iterator)   (float, sorted v order)
//   total    = | union_v pixels of v |
// The integer sums are computed here with bitsets (GPU or CPU); control is summed on
// the CPU in the same order and precision as upstream. Values are therefore identical.

#include "gpu/host/vgavisuallocalaccel.h"

#include "gpu/host/gpuloader.h"
#include "gpu/host/visibilitybits.h"

#include "salalib/vgamodules/vgavisuallocal.h"

#include "genlib/stringutils.h"

#include <algorithm>
#include <atomic>
#include <memory>

namespace {

using dmxaccel::VisibilityBits;

constexpr int32_t kRowsPerBatch = 1024;

struct LocalCounts {
    std::vector<int64_t> cluster;  // per row
    std::vector<int32_t> total;    // per row
    std::vector<float> control;    // per row
};

// Posts progress and honours cancellation between batches (never inside OpenMP regions).
class Progress {
  public:
    Progress(Communicator *comm, int records) : m_comm(comm) {
        if (m_comm) {
            qtimer(m_time, 0);
            m_comm->CommPostMessage(Communicator::NUM_RECORDS, records);
        }
    }
    void update(int done) {
        if (m_comm && qtimer(m_time, 500)) {
            if (m_comm->IsCancelled()) {
                throw Communicator::CancelledException();
            }
            m_comm->CommPostMessage(Communicator::CURRENT_RECORD, done);
        }
    }

  private:
    Communicator *m_comm;
    time_t m_time = 0;
};

// control: float sum over neighbours in PixelRef order, exactly as upstream.
void computeControl(const VisibilityBits &g, const std::vector<int32_t> &sourceRows, LocalCounts &out) {
    const int32_t nSources = int32_t(sourceRows.size());
#pragma omp parallel for schedule(dynamic, 64)
    for (int32_t s = 0; s < nSources; s++) {
        const int32_t u = sourceRows[size_t(s)];
        const uint64_t *rowU = g.row(u);
        float control = 0.0f;
        for (int32_t w = g.firstWord[size_t(u)]; w <= g.lastWord[size_t(u)]; w++) {
            uint64_t bits = rowU[w];
            while (bits) {
                const int32_t v = g.pixelToRow[size_t(w) * 64 + size_t(dmxaccel::lowestBit64(bits))];
                bits &= bits - 1;
                if (v >= 0) {
                    control += 1.0f / float(g.iterationSize[size_t(v)]);
                }
            }
        }
        out.control[size_t(u)] = control;
    }
}

void computeCountsCpu(const VisibilityBits &g, const std::vector<int32_t> &sourceRows, LocalCounts &out,
                      Progress &progress) {
    const int32_t nSources = int32_t(sourceRows.size());
    for (int32_t start = 0; start < nSources; start += kRowsPerBatch) {
        const int32_t end = std::min(nSources, start + kRowsPerBatch);
#pragma omp parallel
        {
            std::vector<uint64_t> unionBits(size_t(g.wordsPerRow), 0);
#pragma omp for schedule(dynamic, 8)
            for (int32_t s = start; s < end; s++) {
                const int32_t u = sourceRows[size_t(s)];
                const uint64_t *rowU = g.row(u);
                const int32_t loU = g.firstWord[size_t(u)], hiU = g.lastWord[size_t(u)];
                int32_t loUnion = g.wordsPerRow, hiUnion = -1;
                int64_t cluster = 0;
                for (int32_t w = loU; w <= hiU; w++) {
                    uint64_t bits = rowU[w];
                    while (bits) {
                        const int32_t v = g.pixelToRow[size_t(w) * 64 + size_t(dmxaccel::lowestBit64(bits))];
                        bits &= bits - 1;
                        if (v < 0) {
                            continue;
                        }
                        const uint64_t *rowV = g.row(v);
                        const int32_t loV = g.firstWord[size_t(v)], hiV = g.lastWord[size_t(v)];
                        for (int32_t k = std::max(loU, loV), kEnd = std::min(hiU, hiV); k <= kEnd; k++) {
                            cluster += dmxaccel::popcount64(rowV[k] & rowU[k]);
                        }
                        for (int32_t k = loV; k <= hiV; k++) {
                            unionBits[size_t(k)] |= rowV[k];
                        }
                        loUnion = std::min(loUnion, loV);
                        hiUnion = std::max(hiUnion, hiV);
                        for (int64_t d = g.dupOffsets[size_t(v)]; d < g.dupOffsets[size_t(v) + 1]; d++) {
                            cluster += g.test(u, g.dupPixels[size_t(d)]) ? 1 : 0;
                        }
                    }
                }
                int32_t total = 0;
                for (int32_t k = loUnion; k <= hiUnion; k++) {
                    total += dmxaccel::popcount64(unionBits[size_t(k)]);
                    unionBits[size_t(k)] = 0;
                }
                out.cluster[size_t(u)] = cluster;
                out.total[size_t(u)] = total;
            }
        }
        progress.update(end);
    }
}

// Returns false (with reason) if the GPU could not complete; out is then incomplete.
bool computeCountsGpu(const VisibilityBits &g, const std::vector<int32_t> &sourceRows, LocalCounts &out,
                      Progress &progress, std::string &reason) {
    auto &gpu = dmxaccel::GpuModule::instance();
    dmx_vis_local *ctx = nullptr;
    if (gpu.visLocalCreate(&ctx, g.rows.data(), g.nRows(), g.wordsPerRow, g.pixelToRow.data(), g.nPixels,
                           g.dupOffsets.data(), g.dupPixels.empty() ? nullptr : g.dupPixels.data()) != DMX_GPU_OK) {
        reason = gpu.lastError();
        return false;
    }
    std::unique_ptr<dmx_vis_local, decltype(gpu.visLocalDestroy)> guard(ctx, gpu.visLocalDestroy);

    // Rows are processed in contiguous ranges; non-source rows in a range are computed and ignored.
    std::vector<int64_t> cluster(kRowsPerBatch);
    std::vector<int32_t> total(kRowsPerBatch);
    for (int32_t first = 0; first < g.nRows(); first += kRowsPerBatch) {
        const int32_t count = std::min(kRowsPerBatch, g.nRows() - first);
        if (gpu.visLocalRun(ctx, first, count, cluster.data(), total.data()) != DMX_GPU_OK) {
            reason = gpu.lastError();
            return false;
        }
        std::copy(cluster.begin(), cluster.begin() + count, out.cluster.begin() + first);
        std::copy(total.begin(), total.begin() + count, out.total.begin() + first);
        progress.update(int(std::min<size_t>(sourceRows.size(), size_t(first + count))));
    }
    return true;
}

} // namespace

bool VGAVisualLocalAccel::run(Communicator *comm, PointMap &map, bool simple_version) {
    using dmxaccel::ComputeMode;

    // Bitset rows need memory proportional to points x pixels; past the budget, fall
    // back to the original algorithm rather than exhausting RAM.
    const uint64_t budget = dmxaccel::physicalMemoryBytes() / 5 * 2;
    if (budget != 0 && VisibilityBits::estimateRowBytes(map) > budget) {
        dmxaccel::addReport("VGA visual local: original serial code (graph too large for the bitset representation)");
        return VGAVisualLocal(m_gates_only).run(comm, map, simple_version);
    }

    VisibilityBits g;
    std::string reason;
    if (!g.build(map, reason)) {
        dmxaccel::addReport("VGA visual local: original serial code (" + reason + ")");
        return VGAVisualLocal(m_gates_only).run(comm, map, simple_version);
    }

    // Same column set-up as VGAVisualLocal::run.
    int cluster_col = -1, control_col = -1, controllability_col = -1;
    if (!simple_version) {
        cluster_col = map.getAttributeTable().insertOrResetColumn("Visual Clustering Coefficient");
        control_col = map.getAttributeTable().insertOrResetColumn("Visual Control");
        controllability_col = map.getAttributeTable().insertOrResetColumn("Visual Controllability");
    }

    // Sources in upstream visiting order; skipped points keep their reset values.
    std::vector<int32_t> sourceRows;
    for (int32_t r = 0; r < g.nRows(); r++) {
        const PixelRef curs = g.rowPixel[size_t(r)];
        if ((map.getPoint(curs).contextfilled() && !curs.iseven()) || m_gates_only) {
            continue;
        }
        sourceRows.push_back(r);
    }

    LocalCounts counts;
    counts.cluster.assign(size_t(g.nRows()), 0);
    counts.total.assign(size_t(g.nRows()), 0);
    counts.control.assign(size_t(g.nRows()), 0.0f);

    Progress progress(comm, map.getFilledPointCount());
    std::string path;
    bool done = false;
    if (m_settings.mode == ComputeMode::Gpu) {
        auto &gpu = dmxaccel::GpuModule::instance();
        if (!gpu.available()) {
            path = "GPU unavailable: " + gpu.unavailableReason();
        } else if (g.nRows() < m_settings.gpuMinNodes) {
            path = "map below the GPU threshold of " + std::to_string(m_settings.gpuMinNodes) + " points";
        } else if (computeCountsGpu(g, sourceRows, counts, progress, reason)) {
            done = true;
            dmxaccel::addReport("VGA visual local: GPU (" + gpu.deviceDescription() + "), " +
                                std::to_string(g.nRows()) + " points");
        } else {
            path = "GPU failed: " + reason;
        }
    }
    if (!done) {
        computeCountsCpu(g, sourceRows, counts, progress);
        dmxaccel::addReport("VGA visual local: CPU, " + std::to_string(dmxaccel::cpuThreadCount()) + " threads, " +
                            std::to_string(g.nRows()) + " points" + (path.empty() ? "" : " (" + path + ")"));
    }
    computeControl(g, sourceRows, counts);

    // Write results exactly as VGAVisualLocal::run does.
    if (!simple_version) {
        for (int32_t u : sourceRows) {
            const PixelRef curs = g.rowPixel[size_t(u)];
            AttributeRow &row = map.getAttributeTable().getRow(AttributeKey(curs));
            const size_t hoodSize = [&] {
                size_t c = 0;
                for (int32_t w = g.firstWord[size_t(u)]; w <= g.lastWord[size_t(u)]; w++) {
                    c += size_t(dmxaccel::popcount64(g.row(u)[w]));
                }
                return c;
            }();
            if (hoodSize > 1) {
                const int cluster = int(counts.cluster[size_t(u)]);
                row.setValue(cluster_col, float(cluster / double(hoodSize * (hoodSize - 1.0))));
                row.setValue(control_col, float(counts.control[size_t(u)]));
                row.setValue(controllability_col,
                             float(double(hoodSize) / double(size_t(counts.total[size_t(u)]))));
            } else {
                row.setValue(cluster_col, -1);
                row.setValue(control_col, -1);
                row.setValue(controllability_col, -1);
            }
        }
    }

#ifndef _COMPILE_dX_SIMPLE_VERSION
    if (!simple_version)
        map.setDisplayedAttribute(cluster_col);
#endif

    return true;
}

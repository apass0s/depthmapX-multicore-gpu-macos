# GPU (CUDA) acceleration plan for depthmapX

Status: VGA visibility: GPU + multicore CPU. VGA metric/angular, axial and segment analyses: multicore CPU. Rapido CLI extras (-xacb, -vr list) added. All bit-identical to upstream (see "Results").

Target: NVIDIA RTX 4060 Ti 16 GB (Ada, compute capability 8.9 / `sm_89`), Windows 11,
driver 617.42.

## 1. Starting point and an important caveat

The multicore build in `depthmapX-rapido-2026/` was made from source we do not have.
This repository is upstream [SpaceGroupUCL/depthmapX](https://github.com/SpaceGroupUCL/depthmapX)
at tag `v0.9.1` (commit `6ea9755`, "Bump version to 0.9.1", 2026-08-09; identical to
`master` at clone time). Work happens on branch `gpu`.

Differences between upstream 0.9.1 and the rapido build (from binary inspection and timing):

| | upstream v0.9.1 | rapido build |
|---|---|---|
| CLI compiler | (any; we build with MSVC 19.44) | MinGW-w64 GCC 16.2.0 (WinLibs, UCRT) |
| Parallelism | none (no OpenMP, no threads) | OpenMP in CLI only (`GOMP_parallel`, dynamic schedule) |
| GUI parallelism | single worker `QThread` | same; GUI imports no OpenMP runtime, so it is **not** multicore |
| Extra CLI options | — | `-xacb` (Brandes exact choice), comma-separated `-vr` radius list |
| Algorithms | original | **also changed**, see timings below (not just threaded) |
| Results | reference | **bit-identical** to upstream on all 8 analyses tested (§3) |

Consequence: our CPU fallback is upstream's **serial** code, which is 10× to 750×
slower than the rapido build (§3). See Decision A in §8.

## 2. Architecture

```
genlib/        generic utilities
salalib/       analysis library (static lib)
  pointdata.*    PointMap: grid cells; graph construction (sparkGraph2 etc.)
  ngraph.*       Node/Bin: per-cell visibility stored as 32 directional bins of
                 run-length pixel spans (PixelVec), not as an edge list
  mgraph.cpp     MetaGraph::analyseGraph(): dispatches every VGA analysis  <-- hook point
  vgamodules/    VGAVisualGlobal, VGAVisualLocal, VGAMetric, VGAAngular, VGAIsovist,
                 VGAThroughVision (+ *Depth step-depth variants)
  axialmodules/  AxialIntegration, AxialStepDepth
  segmmodules/   SegmentTulip, SegmentAngular, SegmentMetric, SegmentTopological (+ PD/depth)
depthmapXcli/  CLI: parsers -> runmethods.cpp -> MetaGraph (links salalib statically)
depthmapX/     Qt5 GUI -> MetaGraph (links salalib statically; never calls depthmapXcli)
```

The GUI and CLI each link `salalib` statically, so adding GPU dispatch to `salalib` means
**two rebuilds (CLI + GUI)** but only **one GPU module** shared by both (§5).

Algorithm notes that drive the GPU design:

- **Visual global** (`vgavisualglobal.cpp`): one BFS per source cell over the visibility
  graph. Neighbours are enumerated from run-length spans, and an `extents` matrix lets
  later scans along the same direction stop early. Integer depths, so results can
  match exactly.
- **Visual local** (`vgavisuallocal.cpp`): for each cell u and each neighbour v,
  computes |N(v) ∩ N(u)|, |N(v)| and the union of all N(v), using linear `std::find`
  (≈ O(N·deg³) in upstream). Integer counts plus a float sum of `1/|N(v)|` in sorted
  neighbour order.
- **Metric / angular** (`vgametric.cpp`, `vgaangular.cpp`): per-source lazy Dijkstra on a
  `std::set` ordered by (distance, pixel id). Only the source and cells that are blocked
  or wall-adjacent expand further. The accumulated angle depends on the predecessor, so
  **results depend on processing order and tie-breaking**. Upstream also keeps per-search
  state (`m_misc`, `m_dist`, `m_cumangle`) on the shared `Point` objects, which forces
  serial execution.

## 3. Measurements

All times are the analysis step only (`-t` CSV), CLI, Release builds. Rapido 16T = 16
OpenMP threads; 1T = `OMP_NUM_THREADS=1`.

### Example files (VGA grid 4,332 cells / 0.89 M directed edges; 7,987 axial lines; 3,434 segments)

| Analysis | upstream serial (MSVC) | rapido 1T | rapido 16T |
|---|---:|---:|---:|
| VGA visibility, global + local (`-vg -vl -vr n`) | 92.45 s | 0.81 s | 0.12 s |
| VGA metric (`-vr n`) | 34.33 s | 17.12 s | 1.56 s |
| VGA angular | 55.19 s | 24.84 s | 2.41 s |
| VGA isovist | 0.07 s | – | 0.07 s |
| VGA through-vision | 0.24 s | – | 0.16 s |
| Graph construction (`VISPREP -pm`) | – | 0.11 s | 0.02 s |
| Axial n + choice + local + RA | 7.27 s | 4.56 s | 0.41 s |
| Axial exact choice (`-xacb`, rapido only) | n/a | – | 0.20 s |
| Segment tulip 1024 bins + choice, metric radius n | 2.32 s | 1.95 s | 0.20 s |
| Segment angular full | 10.27 s | – | 0.14 s |
| Segment topological / metric | – | – | 0.09 / 0.12 s |

Correctness check: every attribute column of all 8 analyses above (VGA visibility,
metric, angular, isovist, thruvision; axial; segment tulip and angular) is
**bit-identical** between upstream and rapido (`tools/gpu/compare_outputs.py`, zero
tolerance).

### Finer grids generated from `galeria_vazia.graph` (rapido 16T)

| Grid | Cells | Directed edges | Make graph | Vis. global | Vis. local | Metric | Angular |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.04 (example) | 4,332 | 0.89 M | 0.02 s | – | – | 1.6 s | 2.4 s |
| 0.02 | 17,244 | 14.2 M | 0.19 s | 1.6 s | 3.7 s | 38 s | 76 s |
| 0.01 | 68,633 | 224 M | 2.6 s | **112 s** | **213 s** | not run (est. 15–40+ min) | not run |

In this open gallery the edge count grows with N² (mean degree 205 → 824 → 3,260).

## 4. What to port, in priority order

PCIe transfer and CUDA context creation (~0.1–0.3 s) make the GPU pointless below a few
thousand cells. On the example grid the rapido CPU path already finishes visibility in
0.12 s. **All GPU paths get a size threshold below which the CPU runs** (§7).

| # | Analysis | Why | GPU method | Exactness | Expected speedup vs rapido 16T |
|---|---|---|---|---|---|
| 1 | **VGA visibility: local** | 213 s at 68k cells, the largest single cost | Bitset adjacency (N² bits). cluster = Σ popcount(A[u] & A[v]); union = OR of A[v] then popcount; control summed per node in sorted order, as on the CPU | **Exact** (integer counts; float sum in identical order) | 10–30× at ≥ 50k cells |
| 1 | **VGA visibility: global** | 112 s at 68k cells | Multi-source BFS (MS-BFS): batches of up to 4,096 sources run as bit lanes; per-level popcounts give depth distribution, mean depth, integration, entropy | **Exact** (integer depths; final formulas on host, as now) | 5–20× at ≥ 50k cells |
| 2 | VGA metric / angular | Slowest in absolute terms (est. tens of minutes at 68k cells) | One serial search per GPU thread, thousands of sources in flight, keeping CPU ordering and tie-break by pixel id. **Gated by a prototype benchmark** before full port | Within tolerance (double `acos`/`atan2` may differ by 1 ulp between CUDA and MSVC; build with `-fmad=false`) | Uncertain, 2–8×; branchy, irregular-memory code |
| 3 | Graph construction (`VISPREP -pm`) | 2.6 s at 68k cells; grows ~N² | Line-of-sight tests per cell pair | Exact | Only worthwhile at ≥ 200k cells |
| 4 | Axial / segment | < 0.5 s on the examples with rapido | — | — | None at example scale; revisit only for city-scale maps (≥ 100k segments) |
| – | Isovist, through-vision | < 0.25 s | — | — | Not worth porting |

Note: a naive GPU BFS over a full edge list would do N × E ≈ 1.5·10¹³ edge visits at
68k cells, which is **slower** than the current CPU. That is why visibility uses bitsets
and MS-BFS rather than one frontier BFS per source.

## 5. Build strategy

**Recommendation: build CLI and GUI with MSVC, plus a separate CUDA DLL with a plain C
API, loaded at runtime.**

- The MinGW constraint came from the rapido CLI. Since we now start from upstream, both
  executables can use one toolchain (MSVC 19.44 + CMake 4.4 + Ninja); nvcc on Windows
  requires MSVC anyway.
- `depthmapX_gpu.dll` (nvcc 13.4 + MSVC) exposes a plain C API (`extern "C"`, POD
  structs, no STL across the boundary). `salalib` loads it with `LoadLibrary`, so CLI
  and GUI start normally and use the CPU on machines without the DLL, the driver or a
  suitable GPU. The plain C API also stays usable from a MinGW build if the rapido
  source ever turns up.
- The CUDA runtime is linked **statically** into the DLL (`-cudart static`), so only the
  NVIDIA driver is needed at run time; no `cudart64_*.dll` to ship.
- Architectures: `sm_89` SASS + `compute_89` PTX (JIT on newer GPUs). See Decision C.

Layout (all GPU code in one directory):

```
gpu/
  CMakeLists.txt        builds depthmapX_gpu.dll (only if CUDA is found) + gpuloader (static)
  include/dmx_gpu.h     C API: dmx_gpu_probe(), dmx_gpu_vga_visual_local(), ...
  src/*.cu              kernels
  loader/               LoadLibrary wrapper + PointMap -> flat arrays, used by salalib
tools/gpu/
  compare_outputs.py    CPU vs GPU attribute comparison
  validate.ps1          runs CPU and GPU on exemplos/ + generated grids, then compares
```

Hook: in `MetaGraph::analyseGraph()` (`salalib/mgraph.cpp:353-364`), if GPU is requested
and available, call the GPU path; on any failure, run the existing CPU code. CPU module
code stays untouched.

Build trees go to `C:\dmx\` because the long OneDrive path breaks MSVC (MAX_PATH; the
first configure attempt failed with `C1083`), and to keep build output out of OneDrive
sync. Release output goes to `depthmapX-multicore-gpu/` (`interface/` + `motor/`, with
Qt via `windeployqt`, MSVC runtime DLLs and `depthmapX_gpu.dll`). `depthmapX-rapido-2026/`
is never written to.

## 6. GPU memory (16 GB card, plan to use ≤ 12 GB)

| Cells N | Bitset adjacency N²/8 | Edge list CSR, int32 (this gallery) | MS-BFS state (3 × N × 4,096 bits) |
|---:|---:|---:|---:|
| 17k | 37 MB | 54 MB | 26 MB |
| 68k | 589 MB | 854 MB | 105 MB |
| 137k | 2.3 GB | ~3.5 GB | 210 MB |
| 200k | 5.0 GB | ~7 GB | 310 MB |
| 275k | 9.4 GB | ~14 GB | 420 MB |
| 340k | 14.4 GB (too big) | – | 520 MB |

Batching:
- Global: sources in batches sized from `cudaMemGetInfo` (up to 4,096 lanes); the
  adjacency stays resident.
- Local: nodes processed in blocks; the adjacency must be resident (random access to
  N(v)). Practical limit ≈ 300k cells. Above it: CPU fallback with a clear message, or
  later a span-based (run-length) representation, which is O(N·spans) instead of O(N²).
- Metric/angular: per in-flight source ≈ 13 bytes × N (dist, cumangle, flags, heap index);
  concurrency = free memory / that (≈ 8k sources at 68k cells in 7 GB).

## 7. CPU/GPU selection and fallback

- CLI: new global option `-gpu` (try GPU, fall back to CPU) and `-gpu force` (skip the
  size threshold, for benchmarking). Without `-gpu`, behaviour is exactly upstream.
  Prints `GPU: <device> (<analysis>)` or `GPU unavailable: <reason>; using CPU`, and
  `-t` timings show which path ran.
- GUI: checkbox "Use NVIDIA GPU (CUDA) when available" in the analysis options dialog,
  saved in settings; the status bar reports the path used.
- Automatic fallback when: the DLL is missing; driver/runtime init fails; no device with
  compute capability ≥ 8.9 (or the chosen arch list); not enough free memory for the
  minimum batch; any CUDA call returns an error mid-run (partial GPU results are
  discarded and the CPU path re-runs); the map is below the size threshold (initially
  10k cells, tuned in Phase 3).
- Every CUDA call is checked; errors carry the call site and `cudaGetErrorString`.

## 8. Decisions needed before Phase 2

**A. CPU fallback speed.** Upstream serial is 750× slower than rapido for visibility on the
example grid. Options:
1. Accept upstream serial as the CPU path (simplest; the fallback is painfully slow on
   large grids).
2. **(Recommended)** For each analysis I port to GPU, also add an OpenMP CPU version
   using the same data layout (bitsets / per-source independence). The fallback is then
   multicore again, is validated against upstream like the GPU path, and gives a fair
   CPU-vs-GPU comparison. This is my own implementation, not the rapido code.
3. Keep using the rapido binaries for CPU work and use our build only for GPU runs.

**B. First target:** VGA visibility (local + global together, since `-vm visibility`
runs both). OK?

**C. GPU architectures:** `sm_89` only, or also `sm_75` / `sm_86` (Turing / Ampere) so the
build runs on lab machines with older RTX cards? Adds build time and DLL size only.
Recommended: include them.

**D.** If the rapido source becomes available later, porting this work onto it is
straightforward: the GPU module is self-contained and only `mgraph.cpp` is hooked.

## Results: VGA visibility (Phase 2/3)

Validated with `tools/gpu/validate.py` on the release binaries in
`depthmapX-multicore-gpu/`. Every attribute column was compared with **zero
tolerance** (bit-identical): against upstream serial on the example grid, and against
rapido on the larger grids (serial would take days there; rapido is itself
bit-identical to upstream on the example). Times are the analysis step only (`-t`).
"new CPU" is the multicore fallback in this build (16 threads); GPU times include CUDA
start-up (~0.1 s).

| Grid (points) | Analysis | upstream serial | rapido 16T | new CPU 16T | GPU | match |
|---|---|---:|---:|---:|---:|---|
| 0.04 example (4,332) | local | 84.66 s | 0.070 s | 0.005 s | 0.125 s | exact |
| | global, radius n | 9.73 s | 0.054 s | 0.046 s | 0.140 s | exact |
| | global, radius 3 | 2.86 s | 0.022 s | 0.028 s | 0.128 s | exact |
| 0.02 (17,244) | local | – | 3.62 s | 0.144 s | 0.199 s | exact |
| | global, radius n | – | 1.51 s | 1.32 s | 0.390 s | exact |
| | global, radius 3 | – | 0.64 s | 0.51 s | 0.226 s | exact |
| 0.01 (68,633) | local | – | 198.4 s | 7.76 s | 2.81 s | exact |
| | global, radius n | – | 96.3 s | 32.5 s | 4.72 s | exact |
| | global, radius 3 | – | 40.7 s | 12.5 s | 2.04 s | exact |

Also validated (exact): a copy of the example grid with two merge links, where the
global measures use the per-source CPU search (the GPU path does not handle links).
Upstream's unit tests all pass (3,785 assertions). Fallback without
`depthmapX_gpu.dll` was tested; out-of-memory and mid-run CUDA failures were not forced.

## Results: VGA metric and angular

Multicore CPU only (the GPU prototype was not adopted, see CHANGELOG-GPU.md). Validated
with zero tolerance: against upstream serial on the example grid (with and without merge
links), against rapido on the 17k grid.

| Grid (points) | Analysis | upstream serial | rapido 16T | new CPU 16T | match |
|---|---|---:|---:|---:|---|
| 0.04 example (4,332) | metric, radius n | 36.6 s | 1.60 s | 0.99 s | exact |
| | angular | 54.3 s | 2.34 s | 1.45 s | exact |
| 0.02 (17,244) | metric, radius n | – | 37.7 s | 25.8 s | exact |
| | angular | – | 76.1 s | 47.3 s | exact |

GPU prototype on the 17k grid (exact): metric 24.2 s, angular 89.9 s.

## Results: axial and segment analyses, rapido's extra options

Multicore CPU (small graphs: the GPU would not help). Release binaries, zero tolerance:
against upstream serial (Brandes and the -vr list against rapido, as upstream has
neither). Barnsbury: 7,987 axial lines, 3,434 segments.

| Analysis | upstream serial | rapido 16T | new CPU 16T |
|---|---:|---:|---:|
| Segment angular, radius n | 10.14 s | 0.146 s | 0.114 s |
| Segment tulip 1024, metric radius n, choice | 2.31 s | 0.191 s | 0.265 s |
| Segment tulip, metric radii 250,n | 5.00 s | 0.449 s | 0.511 s |
| Segment tulip, weighted (Segment Length) | 2.34 s | 0.209 s | 0.440 s |
| Segment tulip, steps radii 3,n | 3.08 s | 0.260 s | 0.316 s |
| Segment tulip, angular radii 1,n | 1.95 s | 0.175 s | 0.229 s |
| Segment topological, radius n | 0.99 s | 0.091 s | 0.428 s |
| Segment metric, radius n | 1.38 s | 0.124 s | 0.462 s |
| Axial n + choice + local + RA | 7.17 s | 0.403 s | 0.872 s |
| Axial 3,n + choice | 7.36 s | 0.434 s | 0.879 s |
| Axial n + local + RA (no choice) | 0.66 s | 0.074 s | 0.091 s |
| Axial 3,n + weighted choice | 7.65 s | 0.545 s | 1.657 s |
| Axial 3,n + choice + Brandes (`-xacb`) | n/a | 0.558 s | 1.331 s |
| VGA visibility -vr 3,5,n (+ local) | n/a | 0.123 s | 0.127 s (GPU 0.192 s) |

Where this build is slower than rapido it is because exactness forces some work into
upstream's order: weighted/segment choice additions are replayed serially, and axial
choice keeps upstream's O(front) random removal so the same random lines are picked.

Deviations from the plan:
- Default is now the multicore CPU path (decision A); `-serial` gives the original code.
- The CLI flag is `-gpu` plus `-gpumin <points>` (instead of `-gpu force`).
- Global measures: the GPU path covers maps without merge links or context-filled points;
  others run on the CPU per-source search (still multicore, still exact).

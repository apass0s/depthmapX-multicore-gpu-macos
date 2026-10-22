# GPU changelog

Branch `gpu`, based on upstream depthmapX `v0.9.1` (commit `6ea9755`). The multicore
"rapido" build this project started from has no available source, so all CPU
parallelism here is new code, validated against upstream and against the rapido
binaries.

## Unreleased

### macOS port (experimental; `macos-testing` workspace, not run on a Mac yet)

Full description, test results and the steps for Apple hardware: `../MACOS-PORT.md`.
- Multicore CPU on macOS (arm64 and x86_64): the same OpenMP code, with LLVM libomp
  (Apple's clang has no OpenMP runtime); `-ffp-contract=off` on clang and GCC.
- GPU on macOS: `gpu/metal/`, a Metal module behind the same C interface as the CUDA
  one (VGA visual local and global measures), with a CPU emulator (`gpu/metal/emu/`)
  that runs the same kernels for tests on machines without Metal.
- The Metal module's device layer opens Metal at run time (`dmx_metal_device_rt.cpp`,
  no SDK needed), so it is cross-compiled on Windows; the Objective-C++ version stays as
  `-DDEPTHMAPX_METAL_SDK=ON`. Tested on Windows on a fake Objective-C runtime and Metal
  (`gpu/metal/emu/objcmock/`, `-DDEPTHMAPX_METAL_OBJC_MOCK=ON`) over the emulator. The
  device is never destroyed (no Metal calls while the process exits).
- Test package for a Mac (`tools/mac/package_cross.py`): our command line program and
  GPU module (universal), upstream's official 0.9.1 apps as the interface, and a
  double-click self-test (`tools/mac/selftest.command`) that compares serial, multicore,
  GPU and upstream's official program and writes a report.
- `-gpu` means Metal on macOS (help text, GUI checkbox, run reports); the GPU threshold
  is per platform (Metal value provisional, `tools/mac/bench_gpu_threshold.py`).
- Build: `DEPTHMAPX_GUI` option, macOS 11 minimum, Metal module in the app bundle,
  `DEPTHMAPX_LIBOMP_ROOT` / `DEPTHMAPX_LIBOMP_LIBRARY`, `DEPTHMAPX_METAL_EMULATOR`.
- `tools/mac/`: cross-compiling with Zig, upstream reference builds, Mach-O tools
  (inspect, universal binaries, libomp stub), packaging and test scripts for a Mac.
- Portability fixes: `mgraph440/comm.h` (`<sys/timeb.h>` only where it exists),
  `salalib/layermanager.h` (`<cstdint>`).
- `tools/gpu/validate.py`: `upstream` mode (compare with a separately built upstream).
- Windows: same results as the current release in every validated case (the CUDA path
  included); the credits say "Multicore + GPU build".

### Authorship credits

The About box and the start-up licence screen now credit this build:
"Multicore + NVIDIA GPU build: (C) 2026 Leonardo Luna, Aderson Passos, LInA-UFC -
Laboratório de Modelagem da Informação em Arte, Arquitetura e Ambiente da Universidade
Federal do Ceará" (`depthmapX/UI/AboutDlg.ui`, `depthmapX/UI/licenseagreement.ui`; the
accented letters are HTML entities, so the files stay ASCII).

### Improvements taken from the rapido build's source (received 2026-10-19)

The rapido source (`depthmapx-perf`, base commit `2884093` = upstream v0.9.1, same tree)
was compared with this branch; what helps here was ported, each change re-validated:
- Validation now also requires **byte-identical `.graph` files** (rapido's criterion),
  not only identical exported attributes. This found one difference: VGA visual local
  did not set the displayed attribute as upstream does (values were identical). Fixed.
- **O(1) attribute-row lookup** by shape index in the axial/segment code (upstream's
  accessor walks a std::map: quadratic loops). Barnsbury tulip 0.27 -> 0.19 s.
- **Parallel visibility graph construction** (VISPREP): rows created serially, each
  point's visibility in parallel, statistics written serially. Byte-identical; gallery
  0.01 grid 21.0 -> 2.9 s.
- **Several `-vr` radii from one search** (GPU and multi-source CPU): each radius uses a
  prefix of the level counts and is written as a separate run would write it.
- Tested and not adopted: rapido's AoS layout for axial choice (slower with MSVC:
  0.89 vs 0.70 s); its SCC counting pre-pass (our counting pass takes 0.04 s); its
  per-thread weighted-choice sums (they reorder double additions; this branch keeps the
  exact replay).

### Axial and segment analyses (multicore CPU) and rapido's extra CLI options

- Multicore versions, bit-identical to upstream 0.9.1, of: segment angular, segment
  tulip (all radius types, with choice and weights), segment topological and metric,
  axial integration (with choice, local measures, RA/RRA, weights). `-serial` gives the
  original code; the CLI now passes `-serial` to axial and segment analyses too.
- How exactness is kept where upstream shares state between sources:
  - queues are replaced by heaps that pop in upstream's order (segment angular: angle,
    then insertion order);
  - choice made of integer counts is summed per thread (exact in any order); choice with
    fractional weights is logged per source and replayed in upstream's source order;
  - axial choice picks lines with pafrand(): draws per source are counted first and each
    source starts from the generator state upstream would have reached (LCG jump-ahead,
    model verified against pafrand at run time); the global state ends as upstream's;
  - row values are written in upstream's order (setValue also updates column stats).
- New CLI options from the rapido build (it has no source, so they were rebuilt from its
  help text and checked against its output):
  - `-xacb` (axial): "Choice [Brandes]" (+ " R<n>"), Brandes betweenness over ordered
    pairs, end points excluded, searches limited by each radius. Identical to rapido.
  - `-vr 3,5,n` (VGA visibility): global measures once per listed radius. Identical to
    rapido, which matches separate single-radius runs.

### VGA metric and angular (multicore CPU)

- `-m VGA -vm metric` and `-vm angular` now run on all CPU cores by default
  (`-serial` gives the original code). Results are **bit-identical** to upstream 0.9.1.
- Same per-source searches as upstream, with per-thread state and an indexed heap that
  keeps std::set's (key, pixel) pop order and insert semantics (including the equal-key
  case: metric tests improvements in double but stores keys as float).
- Pixels that provably cannot pass upstream's strict improvement test are skipped
  before sqrt/acos (done bitset; key bound; acos(x) >= sqrt(2(1-x)) for angular;
  squared distance for metric), with margins far above rounding.
- A GPU version (one warp per source) was prototyped: exact, but only 1.5x faster than
  rapido for metric and slower for angular, with kernel launches near the Windows display
  watchdog limit. Not merged; kept on branch `gpu-path-prototype`. `-gpu` runs metric and
  angular on the CPU.

### Phase 2/3: VGA visibility (local and global measures)

New behaviour
- `-m VGA -vm visibility` (local `-vl` and global `-vg`) now runs on all CPU cores by
  default, and on an NVIDIA GPU with `-gpu`. Results are **bit-identical** to upstream
  0.9.1 in every mode (see Validation).
- New CLI options (any mode; only visibility uses them for now):
  - `-gpu`: use the GPU when available, otherwise all CPU cores.
  - `-gpumin <points>`: smallest map sent to the GPU (default 10000; 0 = always).
  - `-serial`: run the original single-threaded upstream code.
  - The CLI prints one line per analysis saying which path ran and, on fallback, why.
- GUI: "Use NVIDIA GPU (CUDA) when available" checkbox in Tools > Visibility > Run
  Visibility Graph Analysis; saved in the settings. The status bar shows the path used.
- Automatic CPU fallback when `depthmapX_gpu.dll` is missing or has the wrong API
  version, no CUDA device/driver, the device has no kernel image for its architecture,
  not enough GPU memory, any CUDA call fails mid-run, or the map is below `-gpumin`.

Algorithms (all produce upstream's exact numbers)
- Local measures: each point's visibility bins become a bitset row; clustering and
  controllability counts are popcounts of row intersections/unions (CPU: OpenMP,
  GPU: one thread block per point). Control is summed on the CPU in upstream's order
  and float precision. Pixels repeated across bins are counted like upstream.
- Global measures: multi-source BFS, up to 1024 searches as bit lanes; visibility runs
  become range-OR updates (two power-of-two blocks + downsweep). Same algorithm on GPU
  and CPU. Maps with merge links or context-filled points use upstream's per-source
  search on all cores, with per-orientation extents and stamp-reset scratch.
- Upstream's left-over per-point scratch state (`m_misc`, `m_extent`) is reproduced.

Code layout
- `gpu/include/dmx_gpu_api.h`: plain C API of the GPU DLL (version 2).
- `gpu/cuda/`: `depthmapX_gpu.dll` (nvcc, static CUDA runtime, sm_75/86/89 + PTX 8.9).
- `gpu/host/`: compiled into salalib: run-time loader, bitset builder, and the
  accelerated `VGAVisualLocalAccel` / `VGAVisualGlobalAccel`.
- Upstream files touched: `CMakeLists.txt`, `salalib/options.h`, `salalib/mgraph.cpp`
  (dispatch), `depthmapXcli/commandlineparser.*`, `depthmapXcli/runmethods.cpp`,
  `depthmapX/` (options dialog, settings, status bar). Upstream analysis modules are
  unchanged.
- `tools/gpu/`: `build.bat`, `deploy.py`, `validate.py`, `compare_outputs.py`.

### Phase 1 (investigation)
- `GPU_PLAN.md`: architecture map, timings, priorities, build strategy, memory.
- `tools/gpu/compare_outputs.py`: column-by-column comparison of attribute CSVs.

## Building

Requirements: Visual Studio 2022 Build Tools (C++ workload), CMake >= 3.13, Ninja,
CUDA Toolkit 12+ (tested with 13.4; optional: without it the build skips the DLL), Qt
5.15.2 `msvc2019_64` for the GUI.

```bat
tools\gpu\build.bat [build dir] [Qt prefix]
rem defaults: C:\dmx\build and C:\Qt\5.15.2\msvc2019_64
```

Use a short build directory outside OneDrive: MSVC fails with `C1083` on very long
paths. CMake option `-DDEPTHMAPX_GPU=OFF` builds without CUDA.

Release folder (CLI + GUI + every DLL they need, mirroring depthmapX-rapido-2026):

```bat
python tools\gpu\deploy.py --build C:\dmx\build --out ..\depthmapX-multicore-gpu --examples ..\depthmapX-rapido-2026\exemplos
```

Target machines need Windows 10/11 x64 and, for the GPU path, an NVIDIA driver that
supports CUDA 13 (R580 or newer; tested with 617.42) on a Turing (RTX 20) or newer GPU.

## Validation

```bat
python tools\gpu\validate.py --cli <depthmapXcli.exe> --rapido <rapido depthmapXcli.exe> ^
    --graph <grid.graph> --modes serial,rapido,cpu,gpu --reference serial ^
    --cases vis_local,vis_global,vis_global_r3 --out C:\dmx\validate
```

Each run's attributes are exported and compared column by column with the reference
(tolerance 0). The GPU mode fails validation if the GPU was not actually used.
Results: see `GPU_PLAN.md`, section "Results".

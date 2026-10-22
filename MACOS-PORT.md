# depthmapX multicore + GPU: macOS port (experimental)

This folder is an isolated workspace for porting our Windows build (multicore CPU + NVIDIA
CUDA, `depthmapX-src` commit `374f69b`) to macOS on Apple Silicon (arm64) and Intel
(x86_64). Nothing outside `macos-testing/` was changed; the Windows build and its release
are untouched.

**Nothing here has been run on a Mac.** Everything below marked *verified* was verified on
Windows, by cross-compiling, by running the same code with Apple's compiler family
(clang + libc++), or by running the Metal kernels on a CPU emulator. The steps to finish
the validation on Apple hardware are at the end.

## Workspace

| Path | What |
|---|---|
| `depthmapX-src/` | the fork: a copy of `depthmapX-src` at `374f69b` plus the macOS changes. Its own git repository at `macos-testing/` (branch `macos`), one commit per step |
| `upstream/depthmapX-0.9.1/` | unmodified upstream v0.9.1 from GitHub (`v0.9.1.tar.gz`, sha256 `ce912354…d0b`); identical, file for file, to the base of our Windows build |
| `tools/zig-x86_64-windows-0.14.1/` | Zig 0.14.1 (official zip, checksum verified): clang 19 + libc++ + macOS system stubs, for cross-compiling |
| `tools/libomp/` | LLVM OpenMP 19.1.5 runtime for macOS 11+, universal arm64 + x86_64 (mac.r-project.org/openmp) |
| `upstream/release-0.9.1-macos/` | upstream's official v0.9.1 macOS release files (GitHub release assets, unmodified): `depthmapX-0.9.1_macos_arm64.zip` (sha256 `01678a0c…3535`), `depthmapX-0.9.1_macos_intel.zip` (`5042f09d…cd41`), `depthmapXcli_macos_arm64` (`070ffa04…4704`), `depthmapXcli_macos_intel` (`1a196b9a…93aa`) |
| `out/` | build products: `depthmapX-multicore-gpu-macos.zip`, the test package for a Mac |

`tools/`, `upstream/` and `out/` are downloads and products, not tracked by git.

## What changed (in `depthmapX-src/`)

Design: the analyses, the multicore code and the CPU fallback are shared with Windows
unchanged; macOS differs only in the build, in the GPU module behind the existing C
interface (`gpu/include/dmx_gpu_api.h`), and in a few messages.

**Multicore CPU (all accelerated analyses: VGA visibility, metric, angular, axial
integration/choice, segment analyses, visibility graph construction).** The code is the
Windows OpenMP code. Apple's clang compiles OpenMP but ships no runtime, so:
- `gpu/CMakeLists.txt` finds LLVM libomp (Homebrew's `libomp`, or `DEPTHMAPX_LIBOMP_ROOT`)
  and passes `-Xpreprocessor -fopenmp`; `DEPTHMAPX_LIBOMP_LIBRARY` lets cross-builds link a
  stub instead (see packaging).
- `-ffp-contract=off` for clang and GCC: clang fuses `a*b+c` by default and arm64 always
  has FMA, which would change rounding relative to upstream.
- `-serial` still runs the original upstream code (reference and fallback).

**GPU (VGA visual local and global measures): Metal instead of CUDA** (`gpu/metal/`).
CUDA does not exist on macOS. The Metal module implements the same C interface as the
CUDA one, so the host code, the dispatch decision and the CPU fallback are unchanged:
- `dmx_metal_kernels.metal`: the same algorithms as the CUDA kernels (bitset rows for the
  local measures; multi-source breadth-first search with range-OR blocks for the global
  ones), checked against the CUDA code and reworked where Apple GPUs differ:
  threadgroup memory is at most 32 KiB (CUDA used 48 KiB), so row slices are sized from
  the device limit at run time; no kernel assumes a threadgroup size (a pipeline can
  allow fewer than 256 threads); reductions go through threadgroup memory instead of
  32-lane shuffles; 64-bit popcount / count-trailing-zeros are built from the 32-bit
  built-ins. All arithmetic is integer, so results cannot depend on fast-math or on
  thread scheduling. Data races were checked per kernel: every written word has one
  writer per dispatch, counters use atomics.
- `dmx_metal_api.cpp`: argument checks, memory planning and the dispatch sequence.
  Memory is unified: no copies (the CPU seeds and clears the search buffers), the budget
  is the GPU's recommended working set minus a 512 MiB reserve, no buffer may exceed the
  maximum buffer length, and lanes are reduced until every grid has fewer than 2^32
  threads. The local measures are committed in command buffers of 256 rows, well below
  the macOS GPU watchdog.
- The Metal calls (device, kernels compiled at run time from source embedded by CMake,
  pipelines, buffers in shared storage, serial compute encoders) exist in two versions
  of the same device layer (`dmx_metal_backend.h`):
  - `dmx_metal_device_rt.cpp` (default): plain C++ that opens the Objective-C runtime,
    Foundation and Metal with `dlopen` and sends every message through `objc_msgSend`
    cast to the method's exact C signature. It needs no Apple header or SDK, so the
    module is cross-compiled on Windows with the command line program, and it links
    only `libSystem`. Ownership is explicit (no ARC); every selector is checked with
    `respondsToSelector:` when the device is created, so a missing method means the CPU,
    not a crash; no message returns a struct or a float (no `_stret`/`_fpret` variants).
  - `dmx_metal_device.mm` (`-DDEPTHMAPX_METAL_SDK=ON`, a Mac with the SDK): the same in
    Objective-C++ with ARC, kept as the reference.
  Compiling the kernels at run time means no offline `metal` compiler (full Xcode) is
  needed. The device is created once and never destroyed (no Metal calls at exit).
- `emu/objcmock/`: a fake Objective-C runtime and Metal for Windows x64: an assembly
  `objc_msgSend` trampoline (looks the method up, jumps with the caller's arguments),
  Metal objects with reference counts over the CPU emulator, autorelease pools,
  zombies (a message to a released object is fatal), checks on encoder/command-buffer
  rules, a report of the objects alive at exit, and failure injection
  (`DMX_OBJCMOCK_FAIL`). It tests `dmx_metal_device_rt.cpp` itself; it cannot test that
  the selector names and signatures are Apple's (those were taken from Apple's API).
- `emu/`: the same module on a CPU emulator (for tests on machines without Metal; never
  shipped): the kernels compiled as C++, threadgroups with real barriers, threadgroups
  in parallel, real atomics, Metal's limit checks, junk-filled new buffers.
- The loader (`gpu/host/gpuloader.cpp`) finds `libdepthmapX_gpu.dylib` next to
  `depthmapXcli` or in `depthmapX.app/Contents/Frameworks`.
- Intel Macs: Metal runs on their Intel/AMD GPUs too; buffers use shared storage, which
  is correct but slower on a discrete GPU.

**Command line and GUI.** `-gpu`, `-gpumin`, `-serial`, `-vr` lists and `-xacb` work as on
Windows. On macOS `-gpu` means Metal: the help text, the GUI checkbox ("Use the GPU
(Metal) when available") and the run reports ("GPU (Apple M2, Metal, GPU family Apple8,
… MiB)") say so. Unsupported or unavailable GPU → the CPU, with the reason in the report.
The credits say "Multicore + GPU build" (it named NVIDIA).

**GPU threshold.** `kDefaultGpuMinNodes` (`gpu/host/accelcommon.h`) is now per platform.
The Metal value is still 10000 and marked **provisional**: it has not been measured.
`tools/mac/bench_gpu_threshold.py` measures the break-even on a Mac.

**Build.** `DEPTHMAPX_GUI` (build without Qt), macOS 11 minimum (first release on Apple
Silicon; libomp's minimum), AppleClang treated as Clang, Metal module built on macOS,
copied next to `depthmapXcli` and into the app bundle. `mgraph440/comm.h`: unused
`<sys/timeb.h>` only where it exists (fix from the rapido build); `salalib/layermanager.h`:
missing `<cstdint>` (found with GCC 16; newer libc++ versions need it too).

**Tools** (`tools/mac/`): `zig-macos.cmake` (cross-compile for macOS), `zig-windows.cmake`
(Apple's compiler family targeting Windows, for tests), `upstream-cli/` (builds unmodified
upstream's CLI as the reference, no Qt), `machoinfo.py` (inspect Mach-O files),
`make_tbd.py` (libomp stub with install name `@rpath/libomp.dylib`), `lipo.py`
(universal binaries), `package_cross.py` (the test package, from Windows),
`selftest.command` (the package's double-click self-test),
`build_macos.sh`, `test_macos.sh`, `package_macos.sh` (on a Mac),
`bench_gpu_threshold.py`. `tools/gpu/validate.py` gained an `upstream` mode.

## Tests actually run (all on this Windows PC)

"Identical" means: every exported attribute equal with tolerance 0 **and** the `.graph`
files byte for byte equal (`tools/gpu/validate.py`). Maps: the gallery grid (4,332
points) and a generated 17k-point grid, `turns_connected` and `gallery_connected` (merge
links), Barnsbury axial (and `barnsbury_extended1_axial`), Barnsbury segments.

| # | What was built and run | Compared with | Result |
|---|---|---|---|
| 1 | **macOS cross-builds** (Zig 0.14.1) of `depthmapXcli` and the 5 non-GUI test suites, **arm64 and x86_64**, with libomp | build + link only (cannot run) | build and link; Mach-O inspection: macOS 11 minimum, links only `libSystem` and `@rpath/libomp.dylib`, rpath `@executable_path`, arm64 ad-hoc signed; universal binary assembled (`lipo.py`, same header layout as Apple's `lipo`) |
| 2 | Metal module in the arm64 cross-build (`-DDEPTHMAPX_GPU=ON`) | configure + compile | CMake rules and embedded kernel source OK; `dmx_metal_api.cpp` compiles for macOS; `dmx_metal_device.mm` stops at `#import <Foundation/Foundation.h>` (no SDK), as expected |
| 3 | Workspace tree with **MSVC + CUDA** (Windows), all 7 test suites incl. GUI | — | all pass (genlib 295, sala 2887, cli 446, module 75, mgraph440 41, depthmapX 3, GuiUnit 7 assertions) |
| 4 | same build, every analysis case, multicore and **CUDA** | the current Windows release (`depthmapX-multicore-gpu`) | **identical** in all cases (Windows behaviour unchanged) |
| 5 | Workspace tree with **clang + libc++** (Zig, Windows target, no OpenMP = one thread): 5 test suites | — | all pass, same counts |
| 6 | same, all analyses: VGA visibility (+ radius lists, merge links), metric, angular, axial (+ Brandes), segment (11 cases), VISPREP 0.04 / 0.02; modes `-serial`, accelerated, Metal emulator | **unmodified upstream 0.9.1 built with the same clang + libc++** | **identical**: 117 comparisons, 0 differences (+ VISPREP `.graph` identical) |
| 7 | Workspace tree with **GCC 16 + libgomp** (multithreaded OpenMP, non-Microsoft): 5 test suites; all analyses, multicore + Metal emulator | upstream 0.9.1 built with MSVC | tests pass; **identical** in every case |
| 8 | **Metal module on the CPU emulator** (MSVC), VGA visibility, local + global, radius lists | upstream 0.9.1 (MSVC) / `-serial` | **identical** (gallery grid); 17k grid identical to the multicore CPU |
| 9 | Metal emulator under stress: threadgroups of 8, 32, 48, 64, 96, 256 threads; threadgroup memory 512 B – 32 KiB (rows split into many slices); max buffer 2 MiB (lanes 1024 → 64); dispatches capped at 7 threadgroups; working set 530/560 MiB | multicore CPU | **identical** in every configuration; limits it cannot meet fall back to the CPU, still identical |
| 10 | **SDK-free Metal module** (`dmx_metal_device_rt.cpp`) cross-built for **arm64 and x86_64** | build + link (cannot run) | builds without warnings; links only `libSystem`; install name `@rpath/libdepthmapX_gpu.dylib`; arm64 ad-hoc signed; a macOS test program linked against it resolves all 10 C entry points (both architectures); universal module assembled |
| 11 | the same device layer on the **fake Objective-C runtime + Metal** (`emu/objcmock`, clang/libc++ Windows build), VGA visibility local, global, radius 3, radius list | upstream 0.9.1 (same clang) / `-serial` | **identical** in all 4 cases; 0 Objective-C rule violations; at exit only the device, its queue and 6 pipelines alive (no buffer, command buffer, encoder or string leaked) |
| 12 | failure injection through the fake runtime: no device, kernel compile error, pipeline error, buffer allocation failure, GPU fault, missing `dispatchThreadgroups:…`, missing `supportsFamily:`, missing `setFastMathEnabled:` | — | each falls back to the CPU with the reason in the report (the last one still runs on the GPU, only fast math is left at Metal's default; the kernels are integer-only) |
| 13 | the package's self-test (`selftest.command`) run with Git Bash on Windows, the Windows release (CUDA) standing in for the Mac programs, in a folder with a space in its path | — | runs to the end, every comparison made; all identical, `PASSED` |

Bugs found and fixed on the way: the local kernel's name clashed with the API type under
clang (MSVC accepted it); `salalib/layermanager.h` missed `<cstdint>` (GCC 16); the count
kernel needed 64 threads per threadgroup (now any number); large grids are split into
dispatches of at most 65,535 threadgroups; 64-bit popcount / ctz built from 32-bit
built-ins (the 64-bit forms are not guaranteed in every Metal version); the GPU module
released its Metal objects from a static destructor while the process exited, which
crashed under the fake runtime after the results were written (now the device is never
destroyed).

Timings measured here are not meaningful for Metal (the emulator runs on the CPU and is
slow by design).

## Not verified (needs a Mac)

- That anything runs on macOS: no binary built here has been executed on a Mac.
- The Metal module on real hardware: the SDK-free device layer was only run on a fake
  runtime, which checks its logic but not that the selector names and signatures match
  Apple's (a mismatch would be reported and fall back to the CPU, or crash if a signature
  is wrong); `dmx_metal_device.mm` was never compiled (it needs the macOS SDK, whose
  licence ties it to Apple hardware); the kernels were never compiled by Apple's Metal
  compiler (the emulator compiles them as C++, which does not check Metal-only rules);
  GPU memory ordering, the watchdog and performance are untested.
- The GUI on macOS (needs Qt for macOS; Homebrew's `qt@5` is assumed to still be
  installable; if it is not, the GUI needs Qt 5.15 from elsewhere or a port to Qt 6).
- Packaging with `macdeployqt`, ad-hoc signing and Gatekeeper behaviour.
- Bit-identical results on arm64: expected (IEEE doubles, no FMA contraction, same
  source), but only a run against upstream built on the same Mac proves it. Results
  are compared with upstream built by the same compiler on the same machine, not with
  the Windows files, since the system math libraries may differ in the last bit.
- The Metal dispatch threshold.

## On real Apple hardware: build and test both architectures

Prerequisites (once per Mac):
```sh
xcode-select --install                       # Command Line Tools (clang, Metal framework)
# Homebrew: https://brew.sh
brew install cmake ninja qt@5 libomp python  # arm64 packages in /opt/homebrew
```
For an Intel (x86_64) build on an Apple Silicon Mac, also install Rosetta and the Intel
Homebrew in `/usr/local`:
```sh
softwareupdate --install-rosetta --agree-to-license
arch -x86_64 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
arch -x86_64 /usr/local/bin/brew install qt@5 libomp
```
(On an Intel Mac, the normal Homebrew is already in `/usr/local`.)

Copy `macos-testing/` and the Windows release's `exemplos/` folder to the Mac, then, in
Terminal (`$EX` = that `exemplos` folder):
```sh
cd macos-testing/depthmapX-src
tools/mac/build_macos.sh arm64        # -> ../build-macos-arm64 (GUI, CLI, Metal, tests)
tools/mac/test_macos.sh ../build-macos-arm64 ../upstream/depthmapX-0.9.1 "$EX" arm64
tools/mac/build_macos.sh x86_64       # -> ../build-macos-x86_64
tools/mac/test_macos.sh ../build-macos-x86_64 ../upstream/depthmapX-0.9.1 "$EX" x86_64
```
`test_macos.sh` runs the unit tests, builds unmodified upstream 0.9.1 with the same
compiler and flags, compares every analysis with it bit for bit (exported attributes and
`.graph` bytes; `-serial`, multicore and Metal runs), compares Metal with the CPU on
17k- and 68k-point grids, and ends with `PASSED` or `FAILED`. On an x86_64 build the
"gpu" runs use the Mac's GPU too (Metal is not architecture-specific).

GPU threshold (on the Apple Silicon Mac, ideally also on an Intel Mac):
```sh
python3 tools/mac/bench_gpu_threshold.py --cli ../build-macos-arm64/depthmapXcli/depthmapXcli \
    --drawing "$EX/galeria_vazia.graph" --fill 0.68,7.04 --out /tmp/dmx-bench
```
Put the reported point count in `kDefaultGpuMinNodes` (macOS branch of
`gpu/host/accelcommon.h`), rebuild, and rerun `test_macos.sh`.

Packages (one per architecture; folder + zip next to `macos-testing/`):
```sh
tools/mac/package_macos.sh ../build-macos-arm64 "$EX" arm64
tools/mac/package_macos.sh ../build-macos-x86_64 "$EX" x86_64
```
Universal 2: the command line part can be joined with
`lipo -create -output depthmapXcli motor-arm64/depthmapXcli motor-x86_64/depthmapXcli`
(or `tools/mac/lipo.py`, as the test package does); a universal GUI needs a universal Qt
(Homebrew's is single-architecture), so ship the two apps separately until then.

## On GitHub's Macs (GitHub Actions)

`.github/workflows/macos.yml` (this folder is the repository root) runs on every push to
the `macos` branch and by hand, on an Apple Silicon (`macos-latest`) and an Intel
(`macos-15-intel`) machine. Each one:
1. installs `qt@5`, `libomp` and `ninja` with Homebrew;
2. downloads upstream 0.9.1's sources and official command line programs (checksums
   checked);
3. compiles the Metal kernels with Apple's offline compiler (`check_metal_kernels.sh`;
   also as Metal 2.3 for macOS 11, reported);
4. builds everything natively (`build_macos.sh`: our interface, command line, Metal
   module, tests), and the Objective-C++ Metal module separately;
5. runs `test_macos.sh` (unit tests; every analysis against upstream built on that
   machine; `-serial`, multicore, Metal). If the machine has no usable Metal device, the
   GPU runs fall back to the CPU, must still be identical, and the result says
   "GPU NOT tested";
6. compares the two Metal modules' `.graph` output;
7. packages the app (`package_macos.sh`, with the self-test and upstream's official
   programs) and runs the package's self-test;
8. uploads the package (`depthmapX-multicore-gpu-macos-<arch>`) and the logs.

Examples: `exemplos/` (the Windows release's example maps).

## Testing without building: the test package

`out/depthmapX-multicore-gpu-macos.zip` (built on Windows with `tools/mac/package_cross.py`)
needs nothing installed on the Mac:
- `motor/`: our `depthmapXcli`, `libdepthmapX_gpu.dylib` (Metal) and `libomp.dylib`, all
  universal (arm64 + x86_64);
- `interface/Apple-Silicon/`, `interface/Intel/`: upstream's official depthmapX 0.9.1
  apps, unchanged (copied entry by entry from upstream's zips: symlinks, permissions and
  signatures intact). Analyses run from their menus are upstream's single-core code;
- `teste/upstream-0.9.1/`: upstream's official `depthmapXcli` for both architectures;
- `exemplos/`, `LEIA-ME README.txt`, `Terminal depthmapX.command`, `Teste depthmapX.command`.

On the Mac: unzip, run `xattr -dr com.apple.quarantine <folder>` once (the files are not
notarised), double-click `Teste depthmapX.command`. It runs 11 analyses with `-serial`,
multicore, Metal (VGA visibility) and upstream's official program, compares `.graph` files
and exported tables byte for byte (multicore and Metal must equal `-serial`; upstream's
official program is reported for information, since its compiler settings may differ),
times all cores against the GPU on 17k- and 22k-point grids, repeats part of it as Intel
code under Rosetta 2 when installed, and writes `test-report.txt` (10–20 minutes).

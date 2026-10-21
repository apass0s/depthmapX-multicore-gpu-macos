depthmapX 0.9.1 - multicore + NVIDIA GPU build
==============================================

Based on upstream depthmapX v0.9.1 (github.com/SpaceGroupUCL/depthmapX), branch "gpu".
Results are identical to upstream 0.9.1; only the speed differs.

Folders
  interface\   depthmapX.exe     graphical interface
  motor\       depthmapXcli.exe  command line
  exemplos\    example files

Requirements
  Windows 10 or 11, 64-bit. Nothing to install: all needed DLLs are in each folder.
  GPU (optional): NVIDIA GeForce RTX 20 series or newer with a recent driver (R580 or
  newer, supporting CUDA 13). Without one, everything runs on the CPU.

What runs where
  VGA visibility (local, global)       all CPU cores, or the GPU on request
  VGA metric, VGA angular              all CPU cores
  Axial integration, segment analyses  all CPU cores
  Visibility graph construction        all CPU cores
  Other analyses                       as in upstream (single core)

Command line (motor\depthmapXcli.exe), in addition to the upstream options
  -gpu              use the NVIDIA GPU where supported (falls back to the CPU)
  -gpumin <points>  smallest map sent to the GPU (default 10000; 0 = always)
  -serial           use the original single-core code
  -vr 3,5,n         VGA visibility: global measures for several radii in one run
  -xacb             axial: exact choice (Brandes betweenness), column "Choice [Brandes]"
  After each analysis a line says whether it ran on the GPU or CPU, and why.

  Example:
    depthmapXcli -m VGA -f exemplos\galeria_grade_0.04.graph -o out.graph -vm visibility -vg -vl -vr n -gpu

Graphical interface (interface\depthmapX.exe)
  Tools > Visibility > Run Visibility Graph Analysis: tick "Use NVIDIA GPU (CUDA) when
  available". The status bar shows where each analysis ran.

Note: on small maps (under about 10000 points) the GPU is slower than the CPU because of
its start-up time, so it is only used for larger maps unless -gpumin says otherwise.

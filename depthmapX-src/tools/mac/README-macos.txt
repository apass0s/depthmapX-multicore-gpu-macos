depthmapX 0.9.1 - multicore + GPU build for macOS
=================================================

Based on upstream depthmapX v0.9.1 (github.com/SpaceGroupUCL/depthmapX).
Multicore + GPU build: (C) 2026 Leonardo Luna, Aderson Passos - LInA-UFC, Laboratorio de
Modelagem da Informacao em Arte, Arquitetura e Ambiente da Universidade Federal do Ceara.
Results are meant to be identical to upstream 0.9.1; only the speed differs.

{STATUS}

Folders
{FOLDERS}
  exemplos/                  example files
  Terminal depthmapX.command double-click: a Terminal in exemplos/ with depthmapXcli ready

Requirements
  macOS 11 (Big Sur) or newer, on Apple Silicon (M1 and later) or Intel.
  Nothing to install: the OpenMP runtime (libomp.dylib) is included.
  GPU: every Mac that runs macOS 11 has Metal; if the GPU cannot be used, the CPU is.

First use: macOS blocks programs downloaded from the internet that are not notarised
by Apple. Once, open Terminal (Applications > Utilities), type
    xattr -dr com.apple.quarantine
followed by a space, drag this folder onto the Terminal window and press Enter.
(Copied from a USB stick there is usually no block.)
Alternatively: right-click (Control-click) a program > Open > Open.

Testing (please): double-click "Teste depthmapX.command". It runs every accelerated
analysis on one core, on all cores and on the GPU, compares the results byte for byte
with each other and with upstream's official program, and writes test-report.txt in this
folder. It takes 10 to 20 minutes. Send test-report.txt back to us.

What runs where
  VGA visibility (local, global)       all CPU cores, or the GPU (Metal) on request
  VGA metric, VGA angular              all CPU cores
  Axial integration, segment analyses  all CPU cores
  Visibility graph construction        all CPU cores
  Other analyses                       as in upstream (single core)
  Anything started from the interface  as in upstream (single core): it is upstream's app

Command line (motor/depthmapXcli), in addition to the upstream options
  -gpu              use the GPU (Metal) where supported (falls back to the CPU)
  -gpumin <points>  smallest map sent to the GPU (0 = always)
  -serial           use the original single-core code
  -vr 3,5,n         VGA visibility: global measures for several radii in one run
  -xacb             axial: exact choice (Brandes betweenness), column "Choice [Brandes]"
  After each analysis a line says whether it ran on the GPU or the CPU, and why.
  Fewer cores (e.g. to keep the computer responsive):  export OMP_NUM_THREADS=4

  Example (in the Terminal opened by "Terminal depthmapX.command"):
    depthmapXcli -m VGA -f galeria_grade_0.04.graph -o out.graph -vm visibility -vg -vl -vr n

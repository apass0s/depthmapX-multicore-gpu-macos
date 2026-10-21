"""Assemble a self-contained Windows release folder.

Layout (mirrors depthmapX-rapido-2026):
  <out>/motor/      depthmapXcli.exe + depthmapX_gpu.dll + MSVC/OpenMP runtime
  <out>/interface/  depthmapX.exe + depthmapX_gpu.dll + Qt (windeployqt) + MSVC/OpenMP runtime
  <out>/exemplos/   copied from --examples, if given

Usage:
  python tools/gpu/deploy.py --build C:/dmx/build --out ../depthmapX-multicore-gpu \
      --examples ../depthmapX-rapido-2026/exemplos

The CUDA runtime is linked statically into depthmapX_gpu.dll, so the only GPU
requirement on the target machine is an NVIDIA driver; without one, or without the
DLL, everything runs on the CPU.
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys

VS_REDIST_GLOB = r"C:\Program Files*\Microsoft Visual Studio\2022\*\VC\Redist\MSVC\*\x64"
RUNTIME_DLLS = {
    "Microsoft.VC143.CRT": ["msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll"],
    "Microsoft.VC143.OpenMP": ["vcomp140.dll"],
}


def find_redist(explicit):
    if explicit:
        return explicit
    candidates = sorted(glob.glob(VS_REDIST_GLOB))
    if not candidates:
        sys.exit("MSVC redistributable folder not found; pass --redist")
    return candidates[-1]


def copy_runtime(redist, dest):
    for sub, names in RUNTIME_DLLS.items():
        for name in names:
            shutil.copy2(os.path.join(redist, sub, name), dest)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True, help="CMake build directory")
    ap.add_argument("--out", required=True, help="release folder to create (must not exist or be empty)")
    ap.add_argument("--qt", default=r"C:\Qt\5.15.2\msvc2019_64", help="Qt prefix (for windeployqt)")
    ap.add_argument("--redist", help=r"...\VC\Redist\MSVC\<version>\x64 (found automatically)")
    ap.add_argument("--examples", help="folder with example files to copy into exemplos/")
    a = ap.parse_args()

    out = os.path.abspath(a.out)
    if os.path.exists(out) and os.listdir(out):
        sys.exit(f"{out} exists and is not empty; remove it first")
    redist = find_redist(a.redist)
    motor, interface = os.path.join(out, "motor"), os.path.join(out, "interface")
    os.makedirs(motor)
    os.makedirs(interface)

    gpu_dll = os.path.join(a.build, "depthmapXcli", "depthmapX_gpu.dll")
    has_gpu = os.path.exists(gpu_dll)

    shutil.copy2(os.path.join(a.build, "depthmapXcli", "depthmapXcli.exe"), motor)
    shutil.copy2(os.path.join(a.build, "depthmapX", "depthmapX.exe"), interface)
    for d in (motor, interface):
        if has_gpu:
            shutil.copy2(gpu_dll, d)
        copy_runtime(redist, d)

    windeployqt = os.path.join(a.qt, "bin", "windeployqt.exe")
    # Qt libraries, plugins, translations and the ANGLE/software OpenGL fallbacks, as in
    # the original release; the MSVC runtime is copied above.
    subprocess.run([windeployqt, "--release", "--no-compiler-runtime", os.path.join(interface, "depthmapX.exe")],
                   check=True, stdout=subprocess.DEVNULL)

    if a.examples:
        shutil.copytree(a.examples, os.path.join(out, "exemplos"))
    shutil.copy2(os.path.join(os.path.dirname(os.path.abspath(__file__)), "README-release.txt"),
                 os.path.join(out, "LEIA-ME README.txt"))

    print(f"Release folder: {out}" + ("" if has_gpu else "  (built without CUDA: no depthmapX_gpu.dll)"))


if __name__ == "__main__":
    main()

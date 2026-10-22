"""Packages the macOS build cross-compiled on Windows (Zig) as one zip.

  motor/depthmapXcli               universal (arm64 + x86_64), joined with lipo.py
  motor/libdepthmapX_gpu.dylib     the Metal module, universal (SDK-free device layer)
  motor/libomp.dylib               the universal LLVM OpenMP runtime, unchanged (Apple-signed)
  interface/Apple-Silicon|Intel/   upstream's official depthmapX 0.9.1 apps, unchanged
  teste/upstream-0.9.1/            upstream's official depthmapXcli 0.9.1, the self-test's reference
  exemplos/                        example maps
  LEIA-ME README.txt, Terminal depthmapX.command, Teste depthmapX.command (selftest.command)

The binaries must have been linked against the libomp stub (make_tbd.py), so they look
for @rpath/libomp.dylib next to themselves. The official apps are copied entry by entry
from upstream's release zips, never extracted here, so their symlinks (Qt frameworks),
permissions and signatures arrive intact. Unix permissions are stored in the zip, which
Finder's Archive Utility and unzip honour. Our own interface is not included: building
it needs Qt and the SDK on a Mac (build_macos.sh).

Usage: python package_cross.py --cli-arm64 <file> --cli-x86_64 <file>
           --gpu-arm64 <dylib> --gpu-x86_64 <dylib> --libomp <dylib>
           --app-arm64 <upstream depthmapX-0.9.1_macos_arm64.zip> --app-intel <..._intel.zip>
           --upstream-cli-arm64 <depthmapXcli_macos_arm64> --upstream-cli-intel <depthmapXcli_macos_intel>
           --examples <folder> --out <zip>
"""
import argparse
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import machoinfo  # noqa: E402

TOP = "depthmapX-multicore-gpu-macos"
STATUS = """STATUS: TEST VERSION, NOT YET RUN ON A MAC. The command line program and its GPU module
were cross-compiled on Windows (Zig 0.14.1: clang + libc++) from the same source as the
Windows build, and checked there as far as possible without a Mac (same results as
upstream 0.9.1 with the same compiler; the Metal module's code on a CPU emulator). They
have not been executed on macOS yet: please run "Teste depthmapX.command" and send back
test-report.txt.
The interface (interface/) is upstream's official depthmapX 0.9.1, unchanged: use it to
open and view the maps the command line program writes. Analyses started from its menus
run upstream's original single-core code; the speed-up is in motor/depthmapXcli."""
FOLDERS = """  motor/                     depthmapXcli (command line, Apple Silicon and Intel),
                             with the GPU module (Metal) and the OpenMP runtime
  interface/Apple-Silicon/   depthmapX.app, official upstream 0.9.1 (M1 and later)
  interface/Intel/           depthmapX.app, official upstream 0.9.1 (Intel Macs)
  teste/                     upstream's official depthmapXcli 0.9.1, used by the self-test
  Teste depthmapX.command    double-click: self-test, writes test-report.txt"""
TERMINAL = """#!/bin/sh
DIR="$(cd "$(dirname "$0")" && pwd)"
export PATH="$DIR/motor:$PATH"
cd "$DIR/exemplos"
echo
echo " depthmapX multicore + GPU - depthmapXcli ready. See LEIA-ME README.txt"
echo " Help: depthmapXcli -h"
echo
exec "${SHELL:-/bin/zsh}" -i
"""


def entry(name, mode):
    info = zipfile.ZipInfo(f"{TOP}/{name}")
    info.create_system = 3  # Unix, so external_attr carries the permissions
    info.external_attr = (mode & 0xFFFF) << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    return info


def arches(path):
    return {s["arch"] for s in machoinfo.slices(Path(path).read_bytes())}


def need_thin(path, arch):
    found = arches(path)
    if found != {arch}:
        sys.exit(f"{path}: expected a thin {arch} binary, found {sorted(found)}")


def copy_app(z, app_zip, folder):
    """Copies depthmapX.app from an upstream release zip into interface/<folder>/."""
    count = 0
    with zipfile.ZipFile(app_zip) as src:
        for info in src.infolist():
            if not info.filename.startswith("depthmapX.app/"):
                sys.exit(f"{app_zip}: unexpected entry {info.filename}")
            out = zipfile.ZipInfo(f"{TOP}/interface/{folder}/{info.filename}", date_time=info.date_time)
            out.create_system = info.create_system
            out.external_attr = info.external_attr  # Unix mode, symlink bit included
            out.compress_type = zipfile.ZIP_DEFLATED if not info.is_dir() else zipfile.ZIP_STORED
            z.writestr(out, src.read(info))
            count += 1
    return count


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("cli-arm64", "cli-x86_64", "gpu-arm64", "gpu-x86_64", "libomp", "app-arm64", "app-intel",
                 "upstream-cli-arm64", "upstream-cli-intel", "examples", "out"):
        ap.add_argument("--" + name, required=True)
    a = ap.parse_args()

    for cli, arch in ((a.cli_arm64, "arm64"), (a.cli_x86_64, "x86_64")):
        need_thin(cli, arch)
        s = machoinfo.slices(Path(cli).read_bytes())[0]
        if "@rpath/libomp.dylib" not in s["libs"] or "@executable_path" not in s["rpaths"]:
            sys.exit(f"{cli} does not look for @rpath/libomp.dylib next to itself (link with the stub)")
    for gpu, arch in ((a.gpu_arm64, "arm64"), (a.gpu_x86_64, "x86_64")):
        need_thin(gpu, arch)
        s = machoinfo.slices(Path(gpu).read_bytes())[0]
        if s["libs"] != ["/usr/lib/libSystem.B.dylib"]:
            sys.exit(f"{gpu} links {s['libs']}: the SDK-free module needs only libSystem")
    need_thin(a.upstream_cli_arm64, "arm64")
    need_thin(a.upstream_cli_intel, "x86_64")
    if arches(a.libomp) != {"arm64", "x86_64"}:
        sys.exit(f"{a.libomp} is not a universal arm64 + x86_64 library")

    readme = (HERE / "README-macos.txt").read_text(encoding="utf-8")
    readme = readme.replace("{STATUS}", STATUS).replace("{FOLDERS}", FOLDERS)
    selftest = (HERE / "selftest.command").read_bytes()
    with tempfile.TemporaryDirectory() as tmp:
        cli = Path(tmp) / "depthmapXcli"
        gpu = Path(tmp) / "libdepthmapX_gpu.dylib"
        subprocess.run([sys.executable, str(HERE / "lipo.py"), str(cli), a.cli_arm64, a.cli_x86_64], check=True)
        subprocess.run([sys.executable, str(HERE / "lipo.py"), str(gpu), a.gpu_arm64, a.gpu_x86_64], check=True)
        with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            z.writestr(entry("LEIA-ME README.txt", 0o100644), readme.encode("utf-8"))
            z.writestr(entry("Terminal depthmapX.command", 0o100755), TERMINAL.encode("utf-8"))
            z.writestr(entry("Teste depthmapX.command", 0o100755), selftest)
            z.writestr(entry("motor/depthmapXcli", 0o100755), cli.read_bytes())
            z.writestr(entry("motor/libdepthmapX_gpu.dylib", 0o100755), gpu.read_bytes())
            z.writestr(entry("motor/libomp.dylib", 0o100755), Path(a.libomp).read_bytes())
            z.writestr(entry("teste/upstream-0.9.1/depthmapXcli-arm64", 0o100755), Path(a.upstream_cli_arm64).read_bytes())
            z.writestr(entry("teste/upstream-0.9.1/depthmapXcli-intel", 0o100755), Path(a.upstream_cli_intel).read_bytes())
            apps = copy_app(z, a.app_arm64, "Apple-Silicon") + copy_app(z, a.app_intel, "Intel")
            for p in sorted(Path(a.examples).iterdir()):
                if p.is_file() and p.suffix.lower() in (".graph", ".dxf"):
                    z.writestr(entry(f"exemplos/{p.name}", 0o100644), p.read_bytes())
    print(f"{a.out}: {len(zipfile.ZipFile(a.out).namelist())} entries ({apps} from the two official apps)")


if __name__ == "__main__":
    main()

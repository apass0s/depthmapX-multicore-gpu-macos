"""Packages the macOS command line program cross-compiled on Windows (Zig) as one zip.

  motor/depthmapXcli    universal (arm64 + x86_64), joined with lipo.py
  motor/libomp.dylib    the universal LLVM OpenMP runtime, unchanged (Apple-signed)
  exemplos/             example maps
  LEIA-ME README.txt, Terminal depthmapX.command

The binaries must have been linked against the libomp stub (make_tbd.py), so they look
for @rpath/libomp.dylib next to themselves. Unix permissions (execute bits) are stored
in the zip, which Finder's Archive Utility and unzip honour. There is no GUI and no GPU
module in this package: both need the macOS SDK, i.e. a build on a Mac (build_macos.sh).

Usage: python package_cross.py --cli-arm64 <file> --cli-x86_64 <file> --libomp <dylib>
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
STATUS = """STATUS: PREVIEW, NOT YET RUN ON A MAC. This command line program was cross-compiled on
Windows (Zig 0.14.1: clang + libc++) from the same source as the Windows build. The same
toolchain building for Windows gives results identical to upstream 0.9.1, but this macOS
build itself has not been executed. It has no GPU module (Metal) and no graphical
interface: both need a build on a Mac (see the macos-testing documentation)."""
FOLDERS = "  motor/                     depthmapXcli (command line, Apple Silicon and Intel)"
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cli-arm64", required=True)
    ap.add_argument("--cli-x86_64", required=True)
    ap.add_argument("--libomp", required=True)
    ap.add_argument("--examples", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    for cli in (a.cli_arm64, a.cli_x86_64):
        s = machoinfo.slices(Path(cli).read_bytes())[0]
        if "@rpath/libomp.dylib" not in s["libs"] or "@executable_path" not in s["rpaths"]:
            sys.exit(f"{cli} does not look for @rpath/libomp.dylib next to itself (link with the stub)")
    if {s["arch"] for s in machoinfo.slices(Path(a.libomp).read_bytes())} != {"arm64", "x86_64"}:
        sys.exit(f"{a.libomp} is not a universal arm64 + x86_64 library")

    readme = (HERE / "README-macos.txt").read_text(encoding="utf-8")
    readme = readme.replace("{STATUS}", STATUS).replace("{FOLDERS}", FOLDERS)
    with tempfile.TemporaryDirectory() as tmp:
        universal = Path(tmp) / "depthmapXcli"
        subprocess.run([sys.executable, str(HERE / "lipo.py"), str(universal), a.cli_arm64, a.cli_x86_64], check=True)
        with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            z.writestr(entry("LEIA-ME README.txt", 0o100644), readme.encode("utf-8"))
            z.writestr(entry("Terminal depthmapX.command", 0o100755), TERMINAL.encode("utf-8"))
            z.writestr(entry("motor/depthmapXcli", 0o100755), universal.read_bytes())
            z.writestr(entry("motor/libomp.dylib", 0o100755), Path(a.libomp).read_bytes())
            for p in sorted(Path(a.examples).iterdir()):
                if p.is_file() and p.suffix.lower() in (".graph", ".dxf"):
                    z.writestr(entry(f"exemplos/{p.name}", 0o100644), p.read_bytes())
    print(f"{a.out}: {len(zipfile.ZipFile(a.out).namelist())} files")


if __name__ == "__main__":
    main()

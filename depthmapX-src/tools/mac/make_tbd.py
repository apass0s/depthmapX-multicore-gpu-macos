"""Writes a text stub (.tbd) of a macOS dynamic library with a new install name.

Linking against the stub instead of the library makes the program look for the library
under that name, e.g. @rpath/libomp.dylib, with no install_name_tool afterwards (which
would also invalidate an arm64 code signature). Used for the cross-builds from Windows,
where the downloaded libomp.dylib says /usr/local/lib/libomp.dylib.

Usage: python make_tbd.py <library.dylib> <out.tbd> [--install-name @rpath/libomp.dylib]
"""
import argparse
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import machoinfo  # noqa: E402

LC_SYMTAB, LC_DYSYMTAB = 0x2, 0xB
N_EXT, N_TYPE, N_SECT = 0x01, 0x0E, 0x0E
TARGETS = {"x86_64": "x86_64-macos", "arm64": "arm64-macos"}


def exported_symbols(data, off):
    _, _, _, ncmds, _, _, _ = struct.unpack_from("<iiIIIII", data, off + 4)
    p = off + 32
    symtab = dysymtab = None
    for _ in range(ncmds):
        cmd, size = struct.unpack_from("<II", data, p)
        if cmd == LC_SYMTAB:
            symtab = struct.unpack_from("<IIII", data, p + 8)
        elif cmd == LC_DYSYMTAB:
            dysymtab = struct.unpack_from("<IIIIII", data, p + 8)
        p += size
    symoff, nsyms, stroff, _ = symtab
    first, count = (dysymtab[2], dysymtab[3]) if dysymtab else (0, nsyms)
    names = []
    for i in range(first, first + count):
        strx, ntype, _, _, _ = struct.unpack_from("<IBBHQ", data, off + symoff + 16 * i)
        if ntype & N_EXT and (ntype & N_TYPE) == N_SECT:
            names.append(data[off + stroff + strx:].split(b"\0", 1)[0].decode())
    return sorted(set(names))


def slice_offsets(data):
    if struct.unpack_from(">I", data, 0)[0] == 0xCAFEBABE:
        n = struct.unpack_from(">I", data, 4)[0]
        return [struct.unpack_from(">iiIII", data, 8 + 20 * i)[2] for i in range(n)]
    return [0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("library")
    ap.add_argument("out")
    ap.add_argument("--install-name", default="@rpath/libomp.dylib")
    a = ap.parse_args()
    data = Path(a.library).read_bytes()
    exports = []
    for off in slice_offsets(data):
        arch = machoinfo.thin(data, off)["arch"]
        exports.append((TARGETS[arch], exported_symbols(data, off)))
    targets = ", ".join(t for t, _ in exports)
    lines = ["--- !tapi-tbd", "tbd-version:     4", f"targets:         [ {targets} ]",
             f"install-name:    '{a.install_name}'", "current-version: 5.0", "exports:"]
    for target, symbols in exports:
        lines.append(f"  - targets:         [ {target} ]")
        quoted = ["'" + s.replace("'", "''") + "'" for s in symbols]
        lines.append("    symbols:         [ " + (",\n" + " " * 23).join(quoted) + " ]")
    lines.append("...")
    Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{a.out}: {', '.join(f'{t} {len(s)} symbols' for t, s in exports)}, install name {a.install_name}")


if __name__ == "__main__":
    main()

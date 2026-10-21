"""Prints what a Mach-O file (macOS executable or library) contains, without macOS tools:
architectures (thin or universal), file type, minimum macOS version, linked libraries,
rpaths and whether it carries a code signature. Used to check cross-built binaries from
Windows; on a Mac, `file`, `lipo -info` and `otool -L` give the same information.

Usage: python machoinfo.py <file> [<file> ...]
"""
import struct
import sys

CPU = {0x01000007: "x86_64", 0x0100000C: "arm64"}
FILETYPE = {2: "executable", 6: "dylib", 8: "bundle"}
LC_LOAD_DYLIB, LC_ID_DYLIB, LC_RPATH, LC_CODE_SIGNATURE = 0xC, 0xD, 0x8000001C, 0x1D
LC_LOAD_WEAK_DYLIB, LC_REEXPORT_DYLIB, LC_BUILD_VERSION, LC_VERSION_MIN_MACOSX = 0x80000018, 0x8000001F, 0x32, 0x24


def version(v):
    return f"{v >> 16}.{(v >> 8) & 0xFF}.{v & 0xFF}"


def thin(data, off):
    magic = struct.unpack_from("<I", data, off)[0]
    if magic != 0xFEEDFACF:
        raise ValueError(f"not a 64-bit little-endian Mach-O slice (magic {magic:#x})")
    cputype, _, filetype, ncmds, _, _, _ = struct.unpack_from("<iiIIIII", data, off + 4)
    info = {"arch": CPU.get(cputype & 0xFFFFFFFF, hex(cputype)), "type": FILETYPE.get(filetype, filetype),
            "libs": [], "rpaths": [], "id": None, "minos": None, "signed": False}
    p = off + 32
    for _ in range(ncmds):
        cmd, size = struct.unpack_from("<II", data, p)
        if cmd in (LC_LOAD_DYLIB, LC_LOAD_WEAK_DYLIB, LC_REEXPORT_DYLIB, LC_ID_DYLIB, LC_RPATH):
            name_off = struct.unpack_from("<I", data, p + 8)[0]
            name = data[p + name_off:p + size].split(b"\0")[0].decode()
            if cmd == LC_ID_DYLIB:
                info["id"] = name
            elif cmd == LC_RPATH:
                info["rpaths"].append(name)
            else:
                info["libs"].append(name + (" (weak)" if cmd == LC_LOAD_WEAK_DYLIB else ""))
        elif cmd == LC_BUILD_VERSION:
            info["minos"] = version(struct.unpack_from("<I", data, p + 12)[0])
        elif cmd == LC_VERSION_MIN_MACOSX:
            info["minos"] = version(struct.unpack_from("<I", data, p + 8)[0])
        elif cmd == LC_CODE_SIGNATURE:
            info["signed"] = True
        p += size
    return info


def slices(data):
    magic = struct.unpack_from(">I", data, 0)[0]
    if magic == 0xCAFEBABE:  # universal ("fat") file, big-endian header
        n = struct.unpack_from(">I", data, 4)[0]
        return [thin(data, struct.unpack_from(">iiIII", data, 8 + 20 * i)[2]) for i in range(n)]
    return [thin(data, 0)]


def main():
    for path in sys.argv[1:]:
        with open(path, "rb") as f:
            data = f.read()
        found = slices(data)
        print(f"{path}: {'universal' if len(found) > 1 else 'thin'} {', '.join(s['arch'] for s in found)}")
        for s in found:
            print(f"  [{s['arch']}] {s['type']}, macOS >= {s['minos']}, "
                  f"{'code signature' if s['signed'] else 'NOT signed'}")
            if s["id"]:
                print(f"    id     {s['id']}")
            for lib in s["libs"]:
                print(f"    links  {lib}")
            for r in s["rpaths"]:
                print(f"    rpath  {r}")


if __name__ == "__main__":
    main()

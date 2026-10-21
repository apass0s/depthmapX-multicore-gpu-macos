"""Joins thin Mach-O files (one per architecture) into a universal binary, like
`lipo -create`, for building a Universal 2 depthmapXcli without a Mac.

The slices are copied unchanged (each keeps its own code signature) behind a fat header,
each aligned to 2^14 bytes (the arm64 page size; lipo uses the same for both).

Usage: python lipo.py <out> <thin file> <thin file> [...]
"""
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import machoinfo  # noqa: E402

ALIGN_LOG2 = 14


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    out, inputs = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
    slices = []
    for p in inputs:
        data = p.read_bytes()
        if struct.unpack_from(">I", data, 0)[0] == 0xCAFEBABE:
            sys.exit(f"{p} is already universal")
        arch = machoinfo.thin(data, 0)["arch"]  # validates the slice
        cputype, cpusubtype = struct.unpack_from("<iI", data, 4)
        slices.append((arch, cputype, cpusubtype & 0x00FFFFFF, data))
    if len({s[0] for s in slices}) != len(slices):
        sys.exit("two slices of the same architecture")
    slices.sort(key=lambda s: s[1])  # by CPU type, as lipo orders them (x86_64 before arm64)

    header = struct.pack(">II", 0xCAFEBABE, len(slices))
    offset = 1 << ALIGN_LOG2
    entries, body = b"", b""
    for _, cputype, cpusubtype, data in slices:
        entries += struct.pack(">iIIII", cputype, cpusubtype, offset, len(data), ALIGN_LOG2)
        padded = len(data) + (-len(data)) % (1 << ALIGN_LOG2)
        body += data + b"\0" * (padded - len(data))
        offset += padded
    head = header + entries
    out.write_bytes(head + b"\0" * ((1 << ALIGN_LOG2) - len(head)) + body)
    print(f"{out}: universal {', '.join(s[0] for s in slices)}")


if __name__ == "__main__":
    main()

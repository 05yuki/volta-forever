#!/usr/bin/env python3
"""Compare the sections of ELF64 files (cubins) by name.

usage: elfsec.py A B          per-section comparison (size, identical bytes, number of differing bytes)
       elfsec.py --dump A SEC hex dump of one section
"""
import struct
import sys


def sections(path):
    data = open(path, "rb").read()
    shoff, = struct.unpack_from("<Q", data, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 0x3A)
    raw = []
    for i in range(shnum):
        name, typ, flags, addr, off, size, link, info, align, entsize = struct.unpack_from(
            "<IIQQQQIIQQ", data, shoff + i * shentsize)
        raw.append((name, typ, off, size, info))
    stroff, strsize = raw[shstrndx][2], raw[shstrndx][3]
    strtab = data[stroff:stroff + strsize]
    out = {}
    for name, typ, off, size, info in raw:
        n = strtab[name:strtab.index(b"\0", name)].decode()
        out[n] = (typ, b"" if typ == 8 else data[off:off + size], info)  # 8 = NOBITS
    return out


def main():
    if sys.argv[1] == "--dump":
        typ, body, info = sections(sys.argv[2])[sys.argv[3]]
        for i in range(0, len(body), 16):
            print(f"{i:06x}  {body[i:i + 16].hex(' ')}")
        return
    a, b = sections(sys.argv[1]), sections(sys.argv[2])
    for name in sorted(set(a) | set(b)):
        if name not in a or name not in b:
            print(f"{name:28s} only in {'A' if name in a else 'B'}")
            continue
        x, y = a[name][1], b[name][1]
        if x == y:
            verdict = "identical"
        elif len(x) == len(y):
            verdict = f"{sum(p != q for p, q in zip(x, y))} bytes differ"
        else:
            verdict = "sizes differ"
        print(f"{name:28s} {len(x):6d} {len(y):6d}  {verdict}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Decode the .nv.info* sections of a cubin (ELF64).

usage: nvinfo.py CUBIN [SECTION-SUBSTRING]
"""
import struct
import sys

ATTR = {
    0x01: "PAD", 0x02: "IMAGE_SLOT", 0x03: "JUMPTABLE_RELOCS", 0x04: "CTAIDZ_USED", 0x05: "MAX_THREADS",
    0x06: "IMAGE_OFFSET", 0x07: "IMAGE_SIZE", 0x08: "TEXTURE_NORMALIZED", 0x09: "SAMPLER_INIT",
    0x0a: "PARAM_CBANK", 0x0b: "SMEM_PARAM_OFFSETS", 0x0c: "CBANK_PARAM_OFFSETS", 0x0d: "SYNC_STACK",
    0x0e: "TEXID_SAMPID_MAP", 0x0f: "EXTERNS", 0x10: "REQNTID", 0x11: "FRAME_SIZE", 0x12: "MIN_STACK_SIZE",
    0x13: "SAMPLER_FORCE_UNNORMALIZED", 0x14: "BINDLESS_IMAGE_OFFSETS", 0x15: "BINDLESS_TEXTURE_BANK",
    0x16: "BINDLESS_SURFACE_BANK", 0x17: "KPARAM_INFO", 0x18: "SMEM_PARAM_SIZE", 0x19: "CBANK_PARAM_SIZE",
    0x1a: "QUERY_NUMATTRIB", 0x1b: "MAXREG_COUNT", 0x1c: "EXIT_INSTR_OFFSETS", 0x1d: "S2RCTL_INSTR_OFFSETS",
    0x1e: "CRS_STACK_SIZE", 0x1f: "NEED_CNP_WRAPPER", 0x20: "NEED_CNP_PATCH", 0x21: "EXPLICIT_CACHING",
    0x22: "ISTYPEP_USED", 0x23: "MAX_STACK_SIZE", 0x24: "SUQ_USED", 0x25: "LD_CACHEMODE_INSTR_OFFSETS",
    0x26: "LOAD_CACHE_REQUEST", 0x27: "ATOM_SYS_INSTR_OFFSETS", 0x28: "COOP_GROUP_INSTR_OFFSETS",
    0x29: "COOP_GROUP_MASK_REGIDS", 0x2a: "SW1850030_WAR", 0x2b: "WMMA_USED", 0x2c: "HAS_PRE_V10_OBJECT",
    0x2d: "ATOMF16_EMUL_INSTR_OFFSETS", 0x2e: "ATOM16_EMUL_INSTR_REG_MAP", 0x2f: "REGCOUNT",
    0x30: "SW2393858_WAR", 0x31: "INT_WARP_WIDE_INSTR_OFFSETS", 0x32: "SHARED_SCRATCH", 0x33: "STATISTICS",
    0x34: "INDIRECT_BRANCH_TARGETS", 0x35: "SW2861232_WAR", 0x36: "SW_WAR", 0x37: "CUDA_API_VERSION",
    0x38: "NUM_MBARRIERS", 0x39: "MBARRIER_INSTR_OFFSETS", 0x3a: "COROUTINE_RESUME_ID_OFFSETS",
    0x3b: "SAM_REGION_STACK_SIZE", 0x3c: "PER_REG_TARGET_PERF_STATS",
}


def sections(data):
    shoff, = struct.unpack_from("<Q", data, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 0x3A)
    hdr = [struct.unpack_from("<IIQQQQIIQQ", data, shoff + i * shentsize) for i in range(shnum)]
    strtab = data[hdr[shstrndx][4]:hdr[shstrndx][4] + hdr[shstrndx][5]]
    for name, typ, flags, addr, off, size, link, info, align, entsize in hdr:
        yield strtab[name:strtab.index(b"\0", name)].decode(), data[off:off + size]


def decode(body):
    pos = 0
    while pos + 4 <= len(body):
        fmt, attr, val = struct.unpack_from("<BBH", body, pos)
        pos += 4
        name = ATTR.get(attr, f"0x{attr:02x}")
        if fmt == 4:  # sized payload
            payload = body[pos:pos + val]
            pos += val
            words = [f"0x{w:x}" for w, in struct.iter_unpack("<I", payload[:len(payload) // 4 * 4])]
            yield f"{name:28s} [{val:3d}] {' '.join(words)}"
        else:
            yield f"{name:28s} fmt{fmt} 0x{val:x}"


def main():
    data = open(sys.argv[1], "rb").read()
    want = sys.argv[2] if len(sys.argv) > 2 else ""
    for name, body in sections(data):
        if name.startswith(".nv.info") and want in name:
            print(f"-- {name}")
            for line in decode(body):
                print("   " + line)


if __name__ == "__main__":
    main()

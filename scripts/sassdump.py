#!/usr/bin/env python3
"""List the instructions of a cubin with their two 64-bit encoding words.

usage: sassdump.py NVDISASM CUBIN [REGEX]     (prints: section offset lo hi text)
"""
import re
import subprocess
import sys


def instructions(nvdisasm, cubin):
    out = subprocess.run([nvdisasm, "-hex", "-c", cubin], capture_output=True, text=True).stdout
    section, pending = None, None
    for line in out.splitlines():
        m = re.match(r"\s*\.section\s+(\.text\.\S+?),", line)
        if m:
            section = m.group(1)
            continue
        m = re.match(r"\s*/\*([0-9a-f]{4})\*/\s+(.*?);\s*/\* (0x[0-9a-f]{16}) \*/", line)
        if m:
            pending = (section, int(m.group(1), 16), m.group(2).strip(), int(m.group(3), 16))
            continue
        m = re.match(r"\s*/\* (0x[0-9a-f]{16}) \*/", line)
        if m and pending:
            yield pending[0], pending[1], pending[3], int(m.group(1), 16), pending[2]
            pending = None


def main():
    pattern = re.compile(sys.argv[3]) if len(sys.argv) > 3 else None
    for section, off, lo, hi, text in instructions(sys.argv[1], sys.argv[2]):
        if not pattern or pattern.search(text):
            print(f"{section:18s} {off:04x}  {lo:016x} {hi:016x}  {text}")


if __name__ == "__main__":
    main()

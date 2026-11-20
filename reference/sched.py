#!/usr/bin/env python3
"""sched.py CUBIN FUNCTION [FROM TO]: the instructions of a function with their scheduling fields (nvdisasm text + decoded control code)."""
import os, re, struct, subprocess, sys
if len(sys.argv) < 3:
    sys.exit(__doc__)
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../toolkits/cuda-12.9.2/bin/nvdisasm')
if not os.path.exists(D):
    sys.exit('%s is not there: this needs NVIDIA\'s disassembler, which scripts/fetch-all.sh redist fetches with the toolkits' % os.path.normpath(D))
path, func = sys.argv[1], sys.argv[2]
lo_, hi_ = (int(sys.argv[3], 16), int(sys.argv[4], 16)) if len(sys.argv) > 4 else (0, 1 << 30)
text = subprocess.run([D, '-c', '-hex', path], capture_output=True, text=True).stdout
on = False; pending = None
for line in text.splitlines():
    if '.section' in line and '.text.' in line:
        on = line.split('.text.')[1].split(',')[0] == func; continue
    if not on: continue
    m = re.search(r'/\*([0-9a-f]{4,})\*/\s+(.*?);\s*/\* 0x([0-9a-f]{16}) \*/', line)
    if m:
        pending = (int(m.group(1), 16), m.group(2).strip(), int(m.group(3), 16)); continue
    m = re.search(r'/\* 0x([0-9a-f]{16}) \*/', line)
    if m and pending:
        off, asm, lo = pending; hi = int(m.group(1), 16); pending = None
        if not (lo_ <= off <= hi_): continue
        s = hi >> 41
        stall, yld, wr, rd, wait, reuse = s & 15, (s >> 4) & 1, (s >> 5) & 7, (s >> 8) & 7, (s >> 11) & 63, (s >> 17) & 15
        print('%04x  B%s R%s W%s %s S%02d reuse%x  %-52s %016x %016x' % (off, ''.join(str(i) if wait >> i & 1 else '-' for i in range(6)), rd if rd != 7 else '-', wr if wr != 7 else '-', 'y' if yld else '-', stall, reuse, asm[:52], lo, hi))

#!/usr/bin/env python3
"""probe.py SM CUBIN...: how many classes of the description match each instruction of real code, by rule."""
import collections, struct, sys
sys.path.insert(0, __import__('os').path.dirname(__file__))
import md as M

def instructions(path):
    """(function, offset, word) for every instruction of a cubin."""
    d = open(path, 'rb').read()
    shoff = struct.unpack_from('<Q', d, 0x28)[0]; shentsize, shnum, shstrndx = struct.unpack_from('<HHH', d, 0x3A)
    secs = [struct.unpack_from('<IIQQQQ', d, shoff + i * shentsize) for i in range(shnum)]
    stroff = secs[shstrndx][4]
    for name, typ, flags, addr, off, size in secs:
        n = d[stroff + name:d.index(b'\0', stroff + name)].decode()
        if n.startswith('.text.') and typ == 1:
            for i in range(0, size - 15, 16):
                lo, hi = struct.unpack_from('<QQ', d, off + i)
                yield n[6:], i, lo | (hi << 64)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    sm = int(sys.argv[1]); md = M.load(sm); dec = M.Decoder(md)
    seen = set(); counts = collections.Counter(); examples = {}; classes = collections.Counter()
    n = 0
    for path in sys.argv[2:]:
        for func, off, w in instructions(path):
            n += 1
            if w in seen: continue
            seen.add(w)
            cands = dec.candidates(w)
            full = dec.decode(w)
            key = (len(full),) + tuple(sorted(c.name for c in full))[:4] if len(full) != 1 else (1,)
            counts[key] += 1
            examples.setdefault(key, (w, [c.name for c in cands][:8]))
            if len(full) == 1: classes[full[0].name] += 1
    print('%d instructions, %d distinct' % (n, len(seen)))
    print('classes matching in full (not counting alternates): count')
    for key, c in sorted(counts.items(), key=lambda kv: -kv[1])[:40]:
        w, cn = examples[key]
        print('  %s: %d   e.g. %032x of %s' % (key, c, w, cn))
    print('%d classes seen' % len(classes))

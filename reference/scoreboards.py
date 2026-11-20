#!/usr/bin/env python3
"""scoreboards.py SM CUBIN...: how native code uses the scoreboards.

1. Which instructions name a scoreboard for having read their operands, and for their result: by
   kind, and by mnemonic for the kinds that are not memory. The kinds that name one for having read
   are those that read their operands late (`machine.LATE`).
2. The stall count of an instruction that names a scoreboard when the very next instruction waits
   for that scoreboard: never less than two, on either architecture. (On a V100 a wait one cycle
   after the scoreboard is named does not wait.)"""
import collections, multiprocessing, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def work(args):
    sm, path = args
    import elf, machine
    from machine import READ_BARRIER, WAIT, WRITE_BARRIER, stall_of
    mc = work.mc = getattr(work, 'mc', None) or machine.Machine()
    names, close = collections.Counter(), collections.Counter()
    e = elf.Elf(open(path, 'rb').read())
    for s in e.sections:
        if not (s.name.startswith('.text.') and s.type == 1):
            continue
        words = [int.from_bytes(s.data[i:i + 16], 'little') for i in range(0, len(s.data) - 15, 16)]
        for j, w in enumerate(words):
            d = mc.decode(sm, w)
            if d is None:
                continue
            rb, wb = (w >> READ_BARRIER) & 7, (w >> WRITE_BARRIER) & 7
            names[(d.kind, d.mnemonic, 'all')] += 1
            names[(d.kind, d.mnemonic, 'read')] += rb != 7
            names[(d.kind, d.mnemonic, 'result')] += wb != 7
            mask = (words[j + 1] >> WAIT) & 63 if j + 1 < len(words) else 0
            for what, sb in (('read', rb), ('result', wb)):
                if sb != 7 and mask >> sb & 1:
                    close[(what, d.kind, min(stall_of(w), 4))] += 1
        mc._cache[sm].clear()
    return names, close


if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    sm, paths = int(sys.argv[1]), sys.argv[2:]
    names, close = collections.Counter(), collections.Counter()
    with multiprocessing.Pool(24) as pool:
        for n, c in pool.imap_unordered(work, [(sm, p) for p in paths]):
            names.update(n)
            close.update(c)
    kinds = collections.defaultdict(collections.Counter)
    for (kind, mnemonic, what), n in names.items():
        kinds[kind][what] += n
    print('by kind: instructions, of which name a scoreboard for having read, for their result')
    for kind, c in sorted(kinds.items(), key=lambda kv: -kv[1]['all']):
        print('  %-10s %9d %8d %8d' % (kind, c['all'], c['read'], c['result']))
    print('mnemonics of the kinds that are not memory, that name one:')
    for (kind, mnemonic, what), n in sorted(names.items()):
        if what != 'all' and not kind.startswith('MIO') and n:
            print('  %-10s %-10s %-6s %8d of %d' % (kind, mnemonic, what, n, names[(kind, mnemonic, 'all')]))
    print('the very next instruction waits for the scoreboard an instruction names: by what for, kind, its stall count')
    for (what, kind, stall), n in sorted(close.items()):
        print('  %9d  %-7s %-10s stall %d%s' % (n, what, kind, stall, '+' if stall == 4 else ''))

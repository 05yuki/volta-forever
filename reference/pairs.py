#!/usr/bin/env python3
"""pairs.py SM CUBIN...: the operands that take two registers or more as one value (a 64-bit value,
a load of 8 or 16 bytes): is the first register even? By letter (R, U), by whether read or written
and by how many registers. In all native code of both architectures it is: the translator keeps
the two halves of a uniform value side by side, the lower one even (`translate.placement`)."""
import sys, os, collections, multiprocessing
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def work(args):
    sm, path = args
    import elf, machine
    from machine import registers
    mc = work.mc = getattr(work, 'mc', None) or machine.Machine()
    out, odd = collections.Counter(), {}
    e = elf.Elf(open(path, 'rb').read())
    for s in e.sections:
        if not (s.name.startswith('.text.') and s.type == 1):
            continue
        lows = set()
        for at in range(0, len(s.data) - 15, 16):
            w = int.from_bytes(s.data[at:at + 16], 'little')
            d = mc.decode(sm, w)
            if d is None:
                continue
            uses = mc._uses[(sm, d.c.name)]
            for way, entries in (('read', uses.reads), ('written', uses.writes)):
                for entry in entries:
                    letter, operand, span = entry[0], entry[1], entry[2]
                    if letter in 'RU' and operand in d.ops and d.ops[operand] != machine.ZERO[letter]:
                        n = registers(span, d.ops)
                        if n >= 2:
                            key = (letter, way, n, 'even' if d.ops[operand] % 2 == 0 else 'odd')
                            out[key] += 1
                            if letter == 'U':
                                lows.add(d.ops[operand])
                            if key[3] == 'odd' and (key, d.c.name) not in odd:
                                odd[(key, d.c.name)] = (os.path.basename(path), s.name[6:50], at)
        out[('functions', 'where a uniform register is the upper half of one pair and the lower half of another', 0, '')] += bool({u for u in lows if u - 1 in lows})
    return out, odd

if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    sm, paths = int(sys.argv[1]), sys.argv[2:]
    total, odd = collections.Counter(), {}
    with multiprocessing.Pool(32) as pool:
        for out, o in pool.imap_unordered(work, [(sm, p) for p in paths]):
            total.update(out)
            for k, v in o.items():
                odd.setdefault(k, v)
    for key in sorted(total, key=str):
        print('%9d  %s' % (total[key], ' '.join(str(x) for x in key if x != '')))
    for (key, name), where in sorted(odd.items(), key=str)[:12]:
        print('   odd:', key, name, where)

#!/usr/bin/env python3
"""late.py SM CUBIN...: what native code does for an instruction that reads its operands some time
after it is issued (a memory access, a conversion).

Between such an instruction and the next one, in straight-line code, that writes one of its operand
registers: was there a wait for the scoreboard on which it says it has read, a wait for the one on
which it says its result is there, or neither; and for neither, the fewest cycles between the two,
with examples. This is what the translator's `late` rests on: on sm_70 no register that a pending
load from global memory reads is written without such a wait (the few instructions with neither are
under opposite predicates, or behind a later memory instruction that says it has read: they are
served in order)."""
import sys, os, collections, multiprocessing
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def work(args):
    sm, path = args
    import elf, machine, translate
    from machine import READ_BARRIER, WAIT, WRITE_BARRIER, stall_of
    mc = work.mc = getattr(work, 'mc', None) or machine.Machine()
    out, least, examples = collections.Counter(), {}, {}
    e = elf.Elf(open(path, 'rb').read())
    for s in e.sections:
        if not (s.name.startswith('.text.') and s.type == 1):
            continue
        words = [int.from_bytes(s.data[i:i + 16], 'little') for i in range(0, len(s.data) - 15, 16)]
        ds = [mc.decode(sm, w) for w in words]
        if any(d is None for d in ds):
            continue
        n = len(ds)
        targets = set()
        for i, d in enumerate(ds):
            rel = translate.relative(mc, sm, d)
            if rel is not None:
                targets.add(i + 1 + translate.signed(rel[0], d.word) * rel[1] // 16)
        for j, d in enumerate(ds):
            if d.kind not in machine.LATE:
                continue
            sources = {r for l, r, _ in d.reads if l == 'R' and r != 255}
            if not sources:
                continue
            rb, wb = (d.word >> READ_BARRIER) & 7, (d.word >> WRITE_BARRIER) & 7
            t, waited_r, waited_w = 0, False, False
            for k in range(j + 1, n):
                t += stall_of(words[k - 1])
                if k in targets:
                    break
                dk = ds[k]
                mask = (dk.word >> WAIT) & 63
                if rb != 7 and mask >> rb & 1:
                    waited_r = True
                if wb != 7 and mask >> wb & 1:
                    waited_w = True
                hit = {r for l, r in dk.writes if l == 'R'} & sources
                if hit:
                    how = 'waited for its read' if waited_r else 'waited for its result' if waited_w else 'no wait'
                    kind = 'late' if dk.kind in machine.LATE else 'plain'
                    key = (d.mnemonic, 'says read' if rb != 7 else 'does not say', how, 'next writer: ' + kind)
                    out[key] += 1
                    if how == 'no wait':
                        if key not in least or t < least[key]:
                            least[key] = t
                            examples[key] = (path, s.name, j, k)
                    break
                if dk.kind == 'BRU':
                    break
    return out, least, examples

if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    sm, paths = int(sys.argv[1]), sys.argv[2:]
    total, least, examples = collections.Counter(), {}, {}
    with multiprocessing.Pool(24) as pool:
        for out, l, ex in pool.imap_unordered(work, [(sm, p) for p in paths]):
            total.update(out)
            for k, v in l.items():
                if k not in least or v < least[k]:
                    least[k], examples[k] = v, ex[k]
    for key, n in sorted(total.items()):
        print('%8d  %-6s %-13s %-22s %-22s %s' % (n, key[0], key[1], key[2], key[3], 'fewest cycles: %d' % least[key] if key in least else ''))
    import elf, machine
    from machine import READ_BARRIER, USCHED, WRITE_BARRIER, WAIT
    from check import disassemble
    shown = 0
    for key in sorted(examples, key=lambda k: least[k]):
        if key[3] != 'next writer: plain' or shown >= 8:
            continue
        shown += 1
        path, name, j, k = examples[key]
        e = elf.Elf(open(path, 'rb').read())
        s = e.section(name)
        a, b = max(0, j - 3), k + 2
        words = [int.from_bytes(s.data[16 * i:16 * i + 16], 'little') for i in range(a, min(b, len(s.data) // 16))]
        texts = disassemble(words, sm)[0]
        print('\n%s, %d cycles: %s %s' % (' / '.join(key), least[key], os.path.basename(path), name[6:70]))
        for i, w in enumerate(words):
            stall, y = (w >> USCHED) & 15, (w >> (USCHED + 4)) & 1
            wr, rd, wait = (w >> WRITE_BARRIER) & 7, (w >> READ_BARRIER) & 7, (w >> WAIT) & 63
            print('  %s%05x  stall %2d %s  wr %s rd %s wait %-3s  %s' % ('>' if a + i in (j, k) else ' ', 16 * (a + i), stall, 'Y' if y else ' ', '-' if wr == 7 else wr, '-' if rd == 7 else rd,
                  ''.join(str(x) for x in range(6) if wait >> x & 1) or '-', texts[i]))

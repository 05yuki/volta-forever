#!/usr/bin/env python3
"""waits_moved.py IN.sm75.cubin OUT.sm75.cubin [FUNCTION...]: the same code with another,
equivalent schedule (in the functions named, or in all).

Where a load names a scoreboard, a ULDC of a constant follows, and the first instruction that
waits for that scoreboard reads what was loaded (all in a straight line, nothing else naming the
scoreboard in between), the wait is moved from the reader onto the ULDC. A wait holds for
everything after it, so the program is the same; compilers do put the wait for one instruction on
an earlier one (977 ULDCs of PyTorch's sm_75 code wait).

A translation must keep such a wait. A translator that loses it (a ULDC made MOVs that do not
wait, as this one did until 2026-11-17) gives wrong results. Measured that day on a V100 with the
translator as it was: k_memory.sm75.cubin with the waits
moved in k_global_atomic and k_global_atomic64 gave wrong sums in both, three runs of three (4 of
65,536 bytes and 3 of 131,072 differ from the native build: what was added had been read before
it was loaded); the same file with the compiler's schedule was right in its seven kernels.
k_memory.waits-moved.sm75.cubin is that file with the waits moved in all its kernels (6 waits),
for tests/gpu/run.sh (`translated-waits-moved`). Prints the places changed.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../reference'))
import elf, machine, translate
from machine import READ_BARRIER, WAIT, WRITE_BARRIER


def main():
    source, target, only = sys.argv[1], sys.argv[2], set(sys.argv[3:])
    mc = machine.Machine()
    e = elf.Elf(open(source, 'rb').read())
    moved = 0
    for s in e.sections:
        if not (s.name.startswith('.text.') and s.type == 1) or (only and s.name[len('.text.'):] not in only):
            continue
        words = [int.from_bytes(s.data[i:i + 16], 'little') for i in range(0, len(s.data), 16)]
        decoded = [mc.decode(75, w) for w in words]
        starts = {0}
        for i, d in enumerate(decoded):
            if d is not None and d.kind == 'BRU':
                starts.add(i + 1)
                rel = translate.relative(mc, 75, d)
                if rel is not None:
                    starts.add(translate.signed(rel[0], d.word) * rel[1] // 16 + i + 1)
        named = lambda w: {(w >> WRITE_BARRIER) & 7, (w >> READ_BARRIER) & 7} - {7}
        for i, (w, d) in enumerate(zip(words, decoded)):
            board = (w >> WRITE_BARRIER) & 7
            if d is None or board == 7 or d.kind not in ('MIO_SLOW', 'MIO_FAST'):
                continue
            constant = None
            for k in range(i + 1, min(i + 24, len(words))):
                if k in starts or decoded[k] is None:
                    break
                if ((words[k] >> WAIT) & 63) >> board & 1:
                    loaded = {r for letter, r in d.writes if letter == 'R'}
                    if constant is not None and loaded & {r for letter, r, role in decoded[k].reads if letter == 'R'}:
                        words[constant] |= 1 << (WAIT + board)
                        words[k] &= ~(1 << (WAIT + board))
                        moved += 1
                        print('%s: the wait for scoreboard %d (the load at 0x%x) moved from 0x%x onto the ULDC at 0x%x' % (s.name[6:], board, 16 * i, 16 * k, 16 * constant))
                    break
                if board in named(words[k]):
                    break
                if decoded[k].c.name == 'uldc_const__RCR' and constant is None and not (words[k] >> WAIT) & 63:
                    constant = k
        s.data = b''.join(w.to_bytes(16, 'little') for w in words)
    open(target, 'wb').write(e.write())
    print('%d waits moved' % moved)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""guard_written.py IN.sm75.cubin OUT.sm75.cubin [FUNCTION...]: the same program, with
instructions that write the predicate that guards them (in the functions named, or in all).

An instruction on uniform registers under a uniform guard (`@UP0 UIADD3 UR6, UR4, URZ, URZ`)
that can give a predicate too (the carry of a sum, whether the result of a logical operation is
other than zero) and gives none is made to give it into the predicate that guards it:
`@UP0 UIADD3 UR6, UP0, UR4, URZ, URZ`. This is done only where nothing reads that predicate
again: in a straight line after the instruction, the next one that names the predicate writes it,
under no guard, and does not read it. The program computes what it computed.

A translation that puts instructions of its own after such an instruction, under the
instruction's guard, gets them wrong: the predicate no longer holds what the instruction ran
under. This translator did, until 2026-11-20, where it keeps a uniform register in the thread's
local memory: the store after the instruction was under the guard, and did not happen when the
sum had run and cleared the predicate; the uniform register kept its old value. No compiled code
at hand has such an instruction: of the 103.5 million instructions of sm_75 code of PyTorch and
of 42 other libraries, 323 write the predicate that guards them, and none of those writes a
uniform register. That is why the test is made by hand.

k_guards.guard-written.sm75.cubin is k_guards.sm75.cubin with this done in all its kernels, for
tests/gpu/run.sh (`translated-guard-written`). Prints the places changed.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../reference'))
import elf, machine


def put(mc, c, word, operand, value):
    """`word`, of sm_75 class `c`, with its operand `operand` set to `value`."""
    for kind, fields, data in mc.d75.plan(c):
        if kind == 'operand' and data == operand:
            return fields[0].put(word, value)
    sys.exit('%s has no operand %s' % (c.name, operand))


def unread_after(decoded, i, g):
    """Whether nothing reads uniform predicate `g` as instruction `i` leaves it: in a straight
    line, the next instruction that names it writes it under no guard and does not read it."""
    for d in decoded[i + 1:]:
        if d is None or any(letter == 'Q' and x == g for letter, x, role in d.reads):
            return False
        guarded = d.ops.get('UPg', 7) != 7 or d.ops.get('UPg@not') or d.ops.get('Pg', 7) != 7 or d.ops.get('Pg@not')
        if any(letter == 'Q' and x == g for letter, x in d.writes) and not guarded:
            return True
        if d.kind == 'BRU':
            return False
    return False


def main():
    source, target, only = sys.argv[1], sys.argv[2], set(sys.argv[3:])
    mc = machine.Machine()
    e = elf.Elf(open(source, 'rb').read())
    changed = 0
    for s in e.sections:
        if not (s.name.startswith('.text.') and s.type == 1) or (only and s.name[len('.text.'):] not in only):
            continue
        words = [int.from_bytes(s.data[i:i + 16], 'little') for i in range(0, len(s.data), 16)]
        decoded = [mc.decode(75, w) for w in words]
        for i, d in enumerate(decoded):
            if d is None or d.ops.get('UPg', 7) == 7 or d.ops.get('UPu') != 7 or d.ops.get('UPv', 7) != 7:
                continue
            if not any(letter == 'U' for letter, x in d.writes) or not unread_after(decoded, i, d.ops['UPg']):
                continue
            g = d.ops['UPg']
            words[i] = put(mc, d.c, d.word, 'UPu', g)
            after = mc.decode(75, words[i])
            assert after is not None and after.c.name == d.c.name and ('Q', g) in after.writes and after.ops['UPg'] == g, hex(16 * i)
            assert {k: v for k, v in after.ops.items() if k != 'UPu'} == {k: v for k, v in d.ops.items() if k != 'UPu'}, hex(16 * i)
            changed += 1
            print('%s+0x%x: %s gives its predicate into UP%d, which guards it' % (s.name[6:], 16 * i, d.mnemonic, g))
        s.data = b''.join(w.to_bytes(16, 'little') for w in words)
    open(target, 'wb').write(e.write())
    print('%d instructions changed' % changed)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main()

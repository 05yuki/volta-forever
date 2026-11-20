#!/usr/bin/env python3
"""uses_check.py SM CUBIN...: what each instruction reads and writes, as the translator has it from
the descriptions (machine.py, latency.Uses), against what nvdisasm itself works out.

nvdisasm draws the lives of the registers of a function (`-plr`): for every instruction, a mark under
each register it reads (v), writes (^) or both (x). That is NVIDIA's own answer to the question the
scheduling of a translation rests on: a register the translator does not know an instruction reads is
a wait it does not insert (the upper half of a 64-bit address, missed until 2026-11-16: a store went
out before the addition that made that half had finished).

For every instruction of the cubins: the registers and predicates of the two must be the same sets,
and for sm_75 the uniform registers and predicates too (nvdisasm 13.5 draws those as well).
Differences are counted by class of instruction and by what differs, most frequent first, with one
instruction as an example. What could not be compared is counted too, by what it is: nvdisasm draws
nothing for the NOPs that fill the end of a function; anything else there is a tool that failed. SM is 70 (cubins in the CUDA 12 container, which nvdisasm 12.9.2 reads) or
75 (the CUDA 13 container, nvdisasm 13.5).
"""
import collections, os, re, struct, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import elf
import machine

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = {70: '../toolkits/cuda-12.9.2/bin/nvdisasm', 75: '../toolkits/cuda-13.5.0/bin/nvdisasm'}


def charts(sm, path):
    """{function: {offset: what the instruction there reads and what it writes, of registers, predicates,
    uniform registers and uniform predicates in turn (eight sets)}}"""
    if not os.path.exists(os.path.join(HERE, TOOL[sm])):
        sys.exit('%s is not there: this needs NVIDIA\'s disassembler, which scripts/fetch-all.sh redist fetches with the toolkits' % os.path.normpath(os.path.join(HERE, TOOL[sm])))
    r = subprocess.run([os.path.join(HERE, TOOL[sm]), '-plr', '-lrm', 'narrow', '-c', path], capture_output=True, text=True, errors='replace')
    if r.returncode and 'Invalid register count' in r.stderr:
        # nvdisasm draws no chart for a file in which a function declares 255 registers. What an
        # instruction reads and writes does not depend on that count: a copy that declares 254.
        e = elf.Elf(open(path, 'rb').read())
        for s in e.sections:
            if s.name.startswith('.text.') and s.type == 1 and s.info >> 24 == 255:
                s.info = (254 << 24) | (s.info & 0xffffff)
        info = e.section('.nv.info')
        if info is not None:
            # (The count is also in the attributes of the file: symbol, count.)
            d, pos = bytearray(info.data), 0
            while pos + 4 <= len(d):
                fmt, attr, val = d[pos], d[pos + 1], struct.unpack_from('<H', d, pos + 2)[0]
                if fmt == 4 and attr == 0x2f and pos + 12 <= len(d) and struct.unpack_from('<I', d, pos + 8)[0] == 255:
                    struct.pack_into('<I', d, pos + 8, 254)
                pos += 4 + val if fmt == 4 else 4
            info.data = bytes(d)
        with tempfile.NamedTemporaryFile(suffix='.cubin') as copy:
            copy.write(e.write())
            copy.flush()
            r = subprocess.run([os.path.join(HERE, TOOL[sm]), '-plr', '-lrm', 'narrow', '-c', copy.name], capture_output=True, text=True, errors='replace')
    if r.returncode:
        # Said, not passed over: its instructions are counted as not compared. (nvdisasm needs
        # 2.6 GB for the largest cubin of PyTorch: too many at once and some are killed.)
        return None, 'nvdisasm ended with %d on %s: %s' % (r.returncode, path, r.stderr.strip()[:200])
    out, name, columns = {}, None, None
    lines = r.stdout.splitlines()
    for k, line in enumerate(lines):
        m = re.search(r'\.section\s+\.text\.([^,\s]+)', line)
        if m:
            name, columns = m.group(1), None
            out[name] = {}
            continue
        if '//' not in line or name is None:
            continue
        groups = line.split('//', 1)[1].split('|')[1:-1]
        if not groups:
            continue
        if '#' in groups[0] and columns is None:
            # The header: under `#`, the units of each register; the lines above have the tens and,
            # past 99 registers, the hundreds.
            above = []
            for back in (1, 2):
                parts = lines[k - back].split('//', 1)[1].split('|')[1:-1] if '//' in lines[k - back] else []
                if not any(ch.isdigit() for part in parts for ch in part):
                    break
                above.append(parts)
            columns = []
            for g, group in enumerate(groups):
                numbers = {}
                for at, ch in enumerate(group):
                    if ch.isdigit() and '#' in group and at > group.index('#'):
                        digits = ''.join(parts[g][at] for parts in reversed(above) if g < len(parts) and at < len(parts[g]) and parts[g][at].isdigit())
                        numbers[at] = int(digits + ch)
                columns.append(numbers)
            continue
        m = re.search(r'/\*([0-9a-f]+)\*/', line)
        if m and columns is not None:
            sets = []
            for g in range(4):
                # The registers, the predicates, and where a function has them the uniform ones.
                group = groups[g] if g < len(groups) else ''
                marks = {columns[g][at]: ch for at, ch in enumerate(group) if at in columns[g] and ch in 'v^x'} if g < len(columns) else {}
                sets += [{r for r, ch in marks.items() if ch in 'vx'}, {r for r, ch in marks.items() if ch in '^x'}]
            out[name][int(m.group(1), 16)] = tuple(sets)
    return out, None


def main():
    sm, paths = int(sys.argv[1]), sys.argv[2:]
    mc = machine.Machine()
    checked, same, calls, differences, examples = 0, 0, 0, collections.Counter(), {}
    absent, failures = collections.Counter(), []
    from check import disassemble
    for path in paths:
        e = elf.Elf(open(path, 'rb').read())
        chart, failure = charts(sm, path)
        if failure:
            failures.append(failure)
        mc._cache[sm].clear()       # (the decoded instructions of the last cubin: gigabytes over a corpus)
        for s in e.sections:
            if not (s.name.startswith('.text.') and s.type == 1):
                continue
            marks = (chart or {}).get(s.name[len('.text.'):], {})
            for at in range(0, len(s.data) - 15, 16):
                w = int.from_bytes(s.data[at:at + 16], 'little')
                d = mc.decode(sm, w)
                if at not in marks or d is None:
                    # Not compared: nvdisasm drew nothing for it (it draws nothing for the NOPs that
                    # fill the end of a function, or it failed), or the descriptions do not have the
                    # instruction. Counted by what it is, so that a tool that failed shows.
                    absent['not decoded' if d is None else 'nvdisasm failed' if chart is None else d.mnemonic] += 1
                    continue
                checked += 1
                ours = tuple(x for letter in 'RPUQ' for x in ({r for l, r, _ in d.reads if l == letter}, {r for l, r in d.writes if l == letter}))
                theirs = marks[at]
                if tuple(ours) == tuple(theirs):
                    same += 1
                    continue
                if d.c.name.startswith('call_'):
                    # nvdisasm marks what the callee may read and change by convention: not operands.
                    calls += 1
                    continue
                for what, a, b in zip(('reads R', 'writes R', 'reads P', 'writes P', 'reads UR', 'writes UR', 'reads UP', 'writes UP'), ours, theirs):
                    if a != b:
                        # By how far from the instruction's own operands: R+1 for the register after one of them.
                        named = {v for k, v in d.ops.items() if isinstance(v, int)}
                        def place(r):
                            return 'operand' if r in named else 'operand+1' if r - 1 in named else 'operand+2..3' if (r - 2 in named or r - 3 in named) else 'another'
                        key = (d.c.name, what, 'nvdisasm also: ' + ','.join(sorted({place(r) for r in b - a})) if b - a else '',
                               'only here: ' + ','.join(sorted({place(r) for r in a - b})) if a - b else '')
                        differences[key] += 1
                        if key not in examples:
                            examples[key] = (w, sorted(a), sorted(b))
    print('%d instructions checked, %d the same as nvdisasm, %d calls (not compared), %d classes of instruction differ somewhere, %d instructions not in the charts' % (
        checked, same, calls, len({k[0] for k in differences}), sum(absent.values())))
    print('not in the charts: %s' % (', '.join('%s %d' % kv for kv in absent.most_common(8)) or 'none'))
    for failure in failures:
        print('FAILED: ' + failure)
    for key, n in differences.most_common(60):
        w, a, b = examples[key]
        text = disassemble([w], sm)[0][0]
        print('%8d  %-28s %-8s %s %s\n          %s  (here %s, nvdisasm %s)' % (n, key[0], key[1], key[2], key[3], text[:70], a, b))


if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main()

#!/usr/bin/env python3
"""idioms.py SM CUBIN...: how code passes return addresses, in real code.

A call gives its callee the address to return to as a number: an immediate that is the offset, in the
section, of the instruction after the call (a relative call), or that offset and the offset of a
LEPC instruction, from which the absolute address is made (a call through a register). Code that is
moved must have those immediates changed. This counts the shapes found, to see whether a fixed rule
recognises all of them.
"""
import collections, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import md as M
from probe import instructions


def main():
    sm, paths = int(sys.argv[1]), sys.argv[2:]
    md = M.load(sm)
    dec = M.Decoder(md)
    cache = {}
    def info(w):
        r = cache.get(w)
        if r is None:
            cs = dec.decode(w)
            c = cs[0] if len(cs) == 1 else None
            ops = dec.operands(c, w) if c else {}
            imm = next((ops[n] for n, o in c.operands.items() if o.type in ('UImm', 'SImm') and o.args and o.args.startswith('32') and n in ops), None) if c else None
            r = cache[w] = (c.name if c else '?', c.mnemonic if c else '?', imm, ops.get('Rd'))
        return r
    shapes = collections.Counter()
    examples = {}
    def function(path, name, code):
        infos = [info(w) for _, w in code]
        offsets = set(o for o, _ in code)
        immediates = collections.defaultdict(list)      # value -> indices of instructions with that 32-bit immediate
        for i, (cname, mnemonic, imm, rd) in enumerate(infos):
            if imm is not None and mnemonic == 'MOV':
                immediates[imm].append(i)
        for i, (cname, mnemonic, imm, rd) in enumerate(infos):
            off = code[i][0]
            if cname.startswith('call_rel'):
                movs = [j for j in immediates.get(off + 16, []) if j < i]
                where = 'none' if not movs else 'just before' if movs[-1] == i - 1 else '%d before' % (i - movs[-1]) if i - movs[-1] <= 8 else 'far before'
                key = 'relative call: MOV of the return offset %s%s' % (where, ', and %d more with that value' % (len(immediates[off + 16]) - 1) if len(immediates.get(off + 16, [])) > 1 else '')
            elif cname.startswith('call_abs'):
                ret = immediates.get(off + 16, [])
                lepc = [j for j in range(max(0, i - 12), i) if infos[j][1] == 'LEPC']
                own = immediates.get(code[lepc[-1]][0], []) if lepc else []
                key = 'call through a register: %s, MOV of the return offset %s, MOV of the offset of LEPC %s' % (
                    'LEPC %d before' % (i - lepc[-1]) if lepc else 'no LEPC', 'yes' if any(lepc and lepc[-1] < j < i for j in ret) else 'no',
                    'yes' if any(lepc and lepc[-1] < j < i for j in own) else 'no')
            elif mnemonic == 'LEPC':
                nxt = [j for j in range(i + 1, min(len(infos), i + 12)) if infos[j][0].startswith('call_abs')]
                key = 'LEPC: %s' % ('a call through a register within 12 instructions' if nxt else 'no call after it')
            elif mnemonic in ('BRX', 'JMX', 'BRXU', 'JMXU', 'JMP', 'RET', 'RPCMOV', 'CALL'):
                key = '%s (%s)' % (mnemonic, cname)
            else:
                continue
            shapes[key] += 1
            examples.setdefault(key, '%s %s+0x%x' % (os.path.basename(path), name[:50], off))
    current, code, cpath = None, [], None
    for path in paths:
        for name, off, w in instructions(path):
            if (path, name) != current:
                if code:
                    function(cpath, current[1], code)
                current, code, cpath = (path, name), [], path
            code.append((off, w))
    if code:
        function(cpath, current[1], code)
    for key, n in shapes.most_common():
        print('%8d  %s      e.g. %s' % (n, key, examples[key]))


if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main()

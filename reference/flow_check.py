#!/usr/bin/env python3
"""flow_check.py CUBIN...: where each instruction of sm_75 code may lead, as the translator has it
against what the text of the instructions and nvdisasm's listing say.

The translator works out which predicates hold something from the flow of a function
(`translate.Function.liveness`: the successors of each instruction, from the descriptions of the
instruction set, the jump tables of the container and its own rules for calls and returns).
`validate.py` works the same lives out again from another flow (`validate.flow`: the mnemonics
that nvdisasm prints, the targets it lists for branches through a register). The two must agree:
an edge that the translator lacks is a path on which a predicate it took for free may be read.

    2026-11-17: BRA.DIV and BRA.CONV, which go by whether the threads of a warp are together,
    were taken by the translator as always taken (1,437 and 255 of them in a third of PyTorch):
    the code after them, where a warp that is not together calls a helper, was in no life.

Prints the instructions whose successors differ, by mnemonic, with an example of each; the exit
status is 0 when none differs.
"""
import collections, multiprocessing, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import elf, machine, translate, validate


def one(path):
    mc = machine.Machine()
    lives, tables, recounted = validate.alive(path)
    e = elf.Elf(open(path, 'rb').read())
    out, examples, checked = collections.Counter(), {}, 0
    for s in e.sections:
        if not (s.name.startswith('.text.') and s.type == 1):
            continue
        name = s.name[len('.text.'):]
        info = e.section('.nv.info.' + name)
        try:
            f = translate.Function(mc, name, s.data, s.info >> 24, translate.tables_of(info.data) if info is not None else {})
        except translate.Refused:
            continue
        f.liveness()
        texts = validate.texts(s.data, 75)
        leads, inside, anywhere = validate.flow(texts, tables.get(name, {}))
        n = len(texts)
        checked += n
        for k in range(n):
            a = sorted({16 * x for x in f.successors[k]})
            b = sorted({x for x in leads.get(16 * k, []) if x < 16 * n})
            if a != b:
                t = texts[k] or '?'
                mnemonic = ('@ ' + t.split()[1]) if t.startswith('@') else t.split()[0]
                kind = 'more for validate.py' if set(a) < set(b) else 'more for the translator' if set(b) < set(a) else 'other'
                out[(mnemonic, kind)] += 1
                examples.setdefault((mnemonic, kind), '%s %s+0x%x: %s; translator %s, validate.py %s' % (
                    os.path.basename(path), name[:30], 16 * k, t[:40], [hex(x) for x in a][:4], [hex(x) for x in b][:4]))
        mc._cache[75].clear()
    return out, examples, checked


def main():
    paths = sys.argv[1:]
    if not paths:
        print(__doc__)
        return 2
    total, examples, checked = collections.Counter(), {}, 0
    with multiprocessing.Pool(min(32, len(paths))) as pool:
        for out, found, n in pool.imap_unordered(one, paths):
            total.update(out)
            checked += n
            for key, example in found.items():
                examples.setdefault(key, example)
    print('%d instructions: %d lead elsewhere for the translator than for validate.py' % (checked, sum(total.values())))
    for (mnemonic, kind), n in total.most_common(40):
        print('%8d  %-20s %-24s e.g. %s' % (n, mnemonic, kind, examples[(mnemonic, kind)]))
    return 1 if total else 0


if __name__ == '__main__':
    sys.exit(main())

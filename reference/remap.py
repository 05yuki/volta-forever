#!/usr/bin/env python3
"""remap.py TABLE: rules that turn an sm_75 instruction with uniform registers or predicates into the
sm_70 instruction that says the same with ordinary ones, each checked with nvdisasm.

A rule pairs a class of sm_75 with a class of sm_70 and says which field of the first goes to which
field of the second (found by what the fields hold: the same operand once uniform names and types
are made ordinary). It is kept only if, for every instruction of that class in TABLE (of both.py),
the word it produces reads, as sm_70, like the original read as sm_75 with the uniform registers
renamed.
"""
import hashlib, collections, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import md as M
from check import disassemble

UNIFORM_REGISTER = M.UNIFORM_TYPES
MNEMONIC = {'VOTEU': 'VOTE', 'S2UR': 'S2R'}
KEEP = ()                                   # mnemonics that start with U and are not uniform
ALIAS = {'idx': 'pmode'}                    # a modifier that has another name on sm_70
FIXED_SCHEDULING = ('BITS_6_121_116_req_bit_set', 'BITS_3_115_113_src_rel_sb', 'BITS_3_112_110_dst_wr_sb', 'BITS_8_124_122_109_105_opex')
SCHEDULING = sum(((1 << b) for b in range(105, 122)), 0)     # stall, barriers, wait mask: kept as they are
BASE = 100                                                   # where the check puts the uniform registers


def ordinary(text):
    text = re.sub(r'\bUR([a-z0-9]?)\b', r'R\1', text)
    text = re.sub(r'\bUP([a-z0-9]?)\b', r'P\1', text)
    return re.sub(r'_UR([a-z])\b', r'_R\1', text)


def steps(md, dec, c):
    """The fields of a class that hold something of the instruction: {what it holds: (fields, kind, operand)}."""
    out = {}
    for kind, fields, data in dec.plan(c):
        names = [f.name for f in fields]
        if kind in ('opcode', 'unused', 'const', 'default') or names[0] in FIXED_SCHEDULING:
            continue
        if kind == 'operand':
            key, operand = data, data
        elif kind == 'attribute':
            key, operand = '%s@%s' % data, data[0]
        elif kind == 'scaled':
            key, operand = '%s SCALE %d' % data, data[0]
        elif kind == 'table':
            # A table by what it holds (the two descriptions number their tables differently), the
            # same from one run to the next.
            key, operand = 'TABLE<%s>(%s)' % (hashlib.sha1(repr(sorted(md.tables[data[0]])).encode()).hexdigest()[:10], ','.join(data[1])), None
        else:
            key, operand = '%s(%s)' % (data[0], ','.join(data[1])), None
        out[key] = (fields, kind, operand)
    return out


def default_raw(md, c, name):
    """The raw value of an operand that the instruction does not give."""
    o = c.operands.get(name)
    if o is None:
        return 0
    values = md.enums.get(o.type)
    if o.args:
        arg = o.args.strip()
        if values is not None and arg.strip('"') in values:
            return values[arg.strip('"')]
        m = re.search(r'/\s*(0x[0-9a-fA-F]+|\d+)', arg)
        if m:
            return int(m.group(1), 0)
    if values is not None:
        if len(set(values.values())) == 1:
            return next(iter(values.values()))
        if o.type in md.defaults:
            return values[md.defaults[o.type]]
        if o.type in M.REGISTER_TYPES:
            return 255
        if o.type == 'Predicate':
            return 7
    return 0


class Rule:
    def __init__(self, md75, d75, a, md70, d70, b, drop=()):
        """`drop`: fields of the sm_75 class that the caller deals with itself (they are in
        `self.dropped`, by what they hold)."""
        self.a, self.b, self.ok, self.why, self.allowed = a, b, True, '', []
        sa, sb = steps(md75, d75, a), steps(md70, d70, b)
        self.moves, taken, self.dropped = [], set(), {}
        for key, (fields, kind, operand) in sa.items():
            if key in drop or (kind == 'table' and 'TABLE' in drop):
                self.dropped[key] = fields
                continue
            target = sb.get(ordinary(key))
            o = a.operands.get(operand) if operand else None
            conv = 'raw'
            if o is not None and kind == 'operand' and o.type in UNIFORM_REGISTER:
                conv = 'register'
            elif o is not None and kind == 'operand' and o.type == 'UniformPredicate':
                conv = 'predicate'
            if kind == 'operand' and o is not None and md75.enums.get(o.type) is not None and len(set(md75.enums[o.type].values())) == 1:
                # A modifier with one value. Where sm_70 has no such field, or one with a single value
                # too, sm_70's own default stands for it; where sm_70's has several, it is carried over.
                other = b.operands.get(ordinary(operand))
                if target is None or other is None or md70.enums.get(other.type) is None or len(set(md70.enums[other.type].values())) == 1:
                    continue
            if target is None and ALIAS.get(key) in sb:
                target = sb[ALIAS[key]]
                self.aliased = getattr(self, 'aliased', []) + [(key, ALIAS[key])]
            if target is None:
                self.ok, self.why = False, 'nothing on sm_70 for %s' % key
                return
            if conv == 'raw' and [len(f) for f in fields] != [len(f) for f in target[0]]:
                other = b.operands.get(ordinary(operand)) if operand else None
                if (kind == 'operand' and o is not None and other is not None and o.type == other.type == 'SImm'
                        and len(fields) == len(target[0]) == 1 and len(fields[0]) < len(target[0][0])
                        and (o.args or '').partition('/')[2] == (other.args or '').partition('/')[2]):
                    # A signed number in a wider field (the offset of a generic load or store has
                    # 24 bits beside a uniform register, 32 without one): the same number.
                    conv = 'signed'
                else:
                    self.ok, self.why = False, 'width of %s' % key
                    return
            if conv == 'raw' and kind == 'operand':
                # The same type, or enumerations with the same values.
                ta, tb = a.operands[operand], b.operands[ALIAS.get(operand, ordinary(operand)) if ordinary(operand) not in b.operands else ordinary(operand)]
                ea, eb = md75.enums.get(ta.type), md70.enums.get(tb.type)
                if ea is not None and eb is not None and ea != eb:
                    # Two enumerations: the values that have the same names in both.
                    def names(e, v):
                        return frozenset(n for n, x in e.items() if x == v and not n.startswith('INVALID'))
                    same = frozenset(v for v in set(ea.values()) if names(ea, v) and names(ea, v) == names(eb, v))
                    if not same:
                        self.ok, self.why = False, 'type of %s' % key
                        return
                    self.allowed.append((fields[0], same))
                elif (ea, ta.args if ea is None else None) != (eb, tb.args if eb is None else None):
                    self.ok, self.why = False, 'type of %s' % key
                    return
            taken.add(ordinary(key) if ordinary(key) in sb else ALIAS.get(key))
            self.moves.append((fields, target[0], conv))
        # What the sm_70 class has and the sm_75 one does not: its default.
        self.fixed = 0
        for kind, fields, data in d70.plan(b):
            if kind in ('const', 'default', 'opcode'):
                self.fixed = fields[0].put(self.fixed, data)
        for key, (fields, kind, operand) in sb.items():
            if key in taken:
                continue
            if kind == 'operand':
                self.fixed = fields[0].put(self.fixed, default_raw(md70, b, operand))
            elif kind != 'attribute':
                self.ok, self.why = False, 'sm_70 has %s' % key
                return

    def apply(self, word, register=lambda u: BASE + u, predicate=lambda q: q):
        """The sm_70 word for `word`, or None when a field has a value that sm_70 does not have."""
        if any(f.get(word) not in ok for f, ok in self.allowed):
            return None
        out = self.fixed | (word & SCHEDULING)
        for src, dst, conv in self.moves:
            for fs, fd in zip(src, dst):
                v = fs.get(word)
                if conv == 'register':
                    v = 255 if v == 63 else register(v)
                elif conv == 'predicate':
                    v = 7 if v == 7 else predicate(v)
                elif conv == 'signed' and v >> (len(fs) - 1):
                    v -= 1 << len(fs)
                out = fd.put(out, v)
        return out


def expected(text):
    """What nvdisasm must print for the translation of an instruction it printed as `text`."""
    text = re.sub(r'\.reuse\b', '', text)
    text = re.sub(r'\bURZ\b', 'RZ', text)
    text = re.sub(r'\bUR(\d+)\b', lambda m: 'R%d' % (BASE + int(m.group(1))), text)
    text = re.sub(r'\bUPT\b', 'PT', text)
    text = re.sub(r'\bUP(\d)\b', r'P\1', text)
    guard, body = ('', text)
    if text.startswith('@'):
        guard, body = text.split(' ', 1)
        guard += ' '
    name, dot, rest = body.partition('.') if '.' in body.split(' ')[0] else body.partition(' ')
    name = MNEMONIC.get(name, name[1:] if name.startswith('U') and name not in KEEP else name)
    return guard + name + dot + rest if dot == '.' else guard + name + ' ' + rest


def plain(text):
    """nvdisasm names some uses of IMAD after what they do (a move, an addition, a shift): the same instruction."""
    return re.sub(r'\bIMAD\.(MOV|IADD|SHL)\b', 'IMAD', text) if text else text


def candidates(md75, d75, md70, d70):
    by_mnemonic = collections.defaultdict(list)
    for c in md70.classes:
        if not c.alternate:
            by_mnemonic[c.mnemonic].append(c)
    out = {}
    for a in md75.classes:
        if a.alternate or not any(o.type in UNIFORM_REGISTER or o.type == 'UniformPredicate' for o in a.operands.values()):
            continue
        m = a.mnemonic
        target = MNEMONIC.get(m, m[1:] if m.startswith('U') and m[1:] in by_mnemonic else m)
        rules = [Rule(md75, d75, a, md70, d70, b) for b in by_mnemonic.get(target, [])]
        good = [r for r in rules if r.ok]
        out[a.name] = (good, [r for r in rules if not r.ok])
    return out


def main():
    md75, md70 = M.load(75), M.load(70)
    d75, d70 = M.Decoder(md75), M.Decoder(md70)
    cands = candidates(md75, d75, md70, d70)
    words = collections.defaultdict(list)
    for line in open(sys.argv[1]):
        f = line.rstrip('\n').split('\t')
        if f[2] in cands:
            words[f[2]].append((int(f[0]), int(f[1], 16), f[4]))
    NOP = 0x000fc000000000000000000000007918
    verdict = {}
    for name in sorted(words, key=lambda n: -sum(w[0] for w in words[n])):
        good, bad = cands[name]
        total = sum(w[0] for w in words[name])
        if not good:
            reasons = collections.Counter(r.why for r in bad)
            verdict[name] = ('no candidate', total, '; '.join('%s' % k for k, v in reasons.most_common(2)))
            continue
        result = None
        sample = words[name]
        for rule in good:
            out = [rule.apply(w) for _, w, _ in sample]
            if any(w is None for w in out):
                result = result or ('differs', total, '%s: a value sm_70 does not have' % rule.b.name)
                continue
            texts = []
            for i in range(0, len(out), 2048):
                t, err = disassemble(out[i:i + 2048], 70)
                if any(x is None for x in t):
                    t = [disassemble([w], 70)[0][0] for w in out[i:i + 2048]]
                texts += t
            wrong = [(text75, got) for (n, w, text75), got in zip(sample, texts) if plain(got) != plain(expected(text75))]
            if not wrong:
                result = ('rule', total, rule.b.name)
                break
            result = ('differs', total, '%s: %d of %d, e.g. %s  ->  %s' % (rule.b.name, len(wrong), len(sample), wrong[0][0], wrong[0][1]))
        verdict[name] = result
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rules-sm75.tsv'), 'w') as f:
        f.write('# sm_75 class, sm_70 class, instructions of the sample: rules checked with nvdisasm by remap.py\n')
        for name, (kind, total, detail) in sorted(verdict.items()):
            if kind == 'rule':
                f.write('%s\t%s\t%d\n' % (name, detail, total))
    counts = collections.Counter()
    for name, (kind, total, detail) in verdict.items():
        counts[kind] += total
    print('instructions with uniform operands in the table: %s' % dict(counts))
    for name, (kind, total, detail) in sorted(verdict.items(), key=lambda kv: -kv[1][1]):
        print('  %8d  %-36s %-12s %s' % (total, name, kind, detail[:170]))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main()

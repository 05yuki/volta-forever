#!/usr/bin/env python3
"""classes.py [-v]: which instruction classes of sm_75 are, bit for bit, classes of sm_70?

For every class of the sm_75 description, the sm_70 classes with the same opcode are compared with it:
the bits of every field, what each field holds (operand and its type, with the values of the type
when it is an enumeration; constant; table, by its contents), the unused bits, the conditions, and
the properties that say which operands are read and written. Prints a summary, and with -v every
class with what differs.
"""
import collections, hashlib, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import md as M


def digest(x):
    return hashlib.sha1(repr(x).encode()).hexdigest()[:10]


class View:
    """A class seen as comparable pieces."""

    def __init__(self, md, c):
        self.c = c
        self.fields = {}       # bits -> what the field holds
        self.domains = {}      # operand -> (type, values or None)
        for lhs, rhs in c.encoding:
            bits = tuple(tuple(md.fields[n].bits) for n in lhs)
            if rhs == '!':
                self.unused = frozenset(b for group in bits for b in group)
                continue
            if rhs == 'Opcode':
                self.opcode_bits = bits
                continue
            text = re.sub(r'TABLES_\w+', lambda m: 'TABLE<%s>' % digest(sorted(md.tables[m.group(0)])), rhs)
            self.fields[bits] = re.sub(r'\s+', '', text)
        for name, o in c.operands.items():
            values = md.enums.get(o.type)
            self.domains[name] = (o.type, o.args, o.star, None if values is None else frozenset(values.items()))
        conditions = re.sub(r'TABLES_\w+', lambda m: 'TABLE<%s>' % digest(sorted(md.tables[m.group(0)])), c.conditions)
        self.conditions = re.sub(r'\s+', ' ', conditions)
        self.properties = {k: re.sub(r'\s+', '', v) for k, v in c.properties.items() if k != 'SIDL_NAME'}
        self.predicates = {k: re.sub(r'\s+', '', v) for k, v in c.predicates.items()}
        self.format = re.sub(r'\s+', ' ', c.format)


def differences(a, b):
    """What differs between two views (a: sm_75, b: sm_70), as short strings."""
    out = []
    if a.fields.keys() != b.fields.keys():
        only_a = [a.fields[k] for k in a.fields.keys() - b.fields.keys()]
        only_b = [b.fields[k] for k in b.fields.keys() - a.fields.keys()]
        out.append('fields: only 75 %s, only 70 %s' % (sorted(only_a), sorted(only_b)))
    for k in a.fields.keys() & b.fields.keys():
        if a.fields[k] != b.fields[k]:
            out.append('field holds %s / %s' % (a.fields[k], b.fields[k]))
    if getattr(a, 'unused', None) != getattr(b, 'unused', None):
        ua, ub = getattr(a, 'unused', frozenset()), getattr(b, 'unused', frozenset())
        out.append('unused bits: only 75 %s, only 70 %s' % (sorted(ua - ub), sorted(ub - ua)))
    for name in sorted(a.domains.keys() | b.domains.keys()):
        da, db = a.domains.get(name), b.domains.get(name)
        if da == db:
            continue
        if da is None or db is None:
            out.append('operand %s only on %s' % (name, '75' if db is None else '70'))
        elif da[:3] != db[:3]:
            out.append('operand %s: %s%s / %s%s' % (name, da[0], da[1] or '', db[0], db[1] or ''))
        else:
            extra, missing = sorted(da[3] - db[3]), sorted(db[3] - da[3])
            out.append('type %s of %s: only 75 %s, only 70 %s' % (da[0], name, extra[:6], missing[:6]))
    if a.conditions != b.conditions:
        out.append('conditions')
    if a.properties != b.properties:
        keys = sorted(k for k in a.properties.keys() | b.properties.keys() if a.properties.get(k) != b.properties.get(k))
        out.append('properties %s' % keys)
    if a.predicates != b.predicates:
        out.append('predicates')
    if a.format != b.format:
        out.append('format')
    return out


def compare():
    """{sm_75 class name: (status, sm_70 class name or None, differences)}."""
    md70, md75 = M.load(70), M.load(75)
    by_opcode = collections.defaultdict(list)
    for c in md70.classes:
        by_opcode[c.opcode].append(View(md70, c))
    result = {}
    for c in md75.classes:
        a = View(md75, c)
        wide = len(md75.fields[c.opcode_field]) == 13
        if wide and c.opcode >> 12:
            result[c.name] = ('no such opcode on sm_70 (bit 91)', None, [])
            continue
        candidates = by_opcode.get(c.opcode & 0xfff, [])
        if not candidates:
            result[c.name] = ('no such opcode on sm_70', None, [])
            continue
        best = min(candidates, key=lambda b: (len(differences(a, b)), b.c.name != c.name))
        diff = differences(a, best)
        # Bit 91 is part of the opcode on sm_75 (0 here) and must then be a bit sm_70 does not use.
        if wide and 91 not in getattr(best, 'unused', ()):
            diff.append('bit 91 is used on sm_70')
        diff = [d for d in diff if d != 'unused bits: only 75 [], only 70 [91]']
        result[c.name] = ('same' if not diff else 'differs', best.c.name, diff)
    return md70, md75, result


if __name__ == '__main__':
    md70, md75, result = compare()
    verbose = '-v' in sys.argv
    for alternate in (False, True):
        names = [c.name for c in md75.classes if c.alternate == alternate]
        counts = collections.Counter(result[n][0] for n in names)
        print('%s classes of sm_75: %d: %s' % ('alternate' if alternate else 'main', len(names), dict(counts)))
    kinds = collections.Counter()
    for c in md75.classes:
        status, other, diff = result[c.name]
        if c.alternate or status != 'differs':
            continue
        key = ' | '.join(sorted(set(re.sub(r"\[.*?\]|<\w+>|'[^']*'", '', d)[:60] for d in diff)))
        kinds[key] += 1
        if verbose:
            print('%-40s %-34s %s' % (c.name, other, '; '.join(diff)[:400]))
    print('how the classes that differ do (main classes):')
    for key, n in kinds.most_common(40):
        print('  %4d  %s' % (n, key[:230]))

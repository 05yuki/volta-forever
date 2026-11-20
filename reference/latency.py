#!/usr/bin/env python3
"""latency.py SM CUBIN...: the shortest distances, in the stall counts of real code, between an
instruction that writes a register or a predicate and one that reads it.

The compiler spaces a reader from the writer by at least the latency its tables give (sm*_2.txt of
the descriptions). This measures what it actually leaves, by kind of writer and reader, in code it
compiled: the smallest sum of stall counts from the writer to the reader, within straight-line code
(nothing carried across a branch or a branch target), for writers that have no scoreboard (those
that do are waited for).
"""
import collections, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import md as M
from probe import instructions


def operation_sets(sm):
    text = open(os.path.join(M.DATA, 'sm%d_2.txt' % sm)).read()
    sets = {}
    for m in re.finditer(r'^\s*([A-Za-z_0-9]+)\s*=\s*\{([^}]*)\}[^;{}]*;', text, re.M):
        sets.setdefault(m.group(1), set(x.strip() for x in m.group(2).split(',') if x.strip()))
    return sets


def kind_of(sets, mnemonic):
    """The group of the latency tables an operation is in."""
    if mnemonic in sets.get('HMMA_OP', ()): return 'HMMA'
    if mnemonic in sets.get('IMMA_OP', ()): return 'IMMA'
    if mnemonic in sets.get('IMAD_OP', ()): return 'IMAD'
    if mnemonic in sets['fmalighter_pipe']: return 'FMAI'
    if mnemonic in sets['int_pipe'] or mnemonic in sets['fe_pipe']: return 'FXU'
    if mnemonic in sets['fma64lite_pipe'] or mnemonic in sets['fp16_pipe']: return 'FMALITE'
    if mnemonic in sets['fma64heavy_pipe']: return 'FMA64HEAVY'
    if mnemonic in sets.get('udp_pipe', ()): return 'UDP'
    if mnemonic in sets['cbu_pipe']: return 'BRU'
    if mnemonic in sets['mio_pipe'] or mnemonic in sets.get('ttu_pipe', ()):
        return 'MIO_SLOW' if mnemonic in sets['MIO_SLOW_OPS'] else 'MIO_FAST'
    return 'OTHER'


# The predicates an instruction sets, by the names the descriptions give them: its results (`u`, `v`),
# uniform ones too. (Until 2026-11-16 the uniform ones were missing here and taken as read: a
# uniform predicate then looked alive from the start of its function, and found no ordinary
# predicate to stand for it.)
WRITTEN_PREDICATES = ('Pu', 'Pv', 'Pd', 'nPd', 'UPu', 'UPv')


class Size:
    """The size in bits of what an instruction reads or writes, where the description gives it as an
    expression of the operands of the instruction (`e ? 64 : 32` for an address, which is of 64
    bits when the instruction says so; a load of 4, 8 or 16 bytes by its `sz`): C, with the values
    of enumerations written `Enumeration@NAME."""

    TOKEN = re.compile(r'\s*(?:(0x[0-9a-fA-F]+|\d+)|`(\w+)@(?:"([^"]*)"|([\w.]+))|(\w+)|(\|\||&&|==|!=|<=|>=|<<|>>|[-+*/?:()<>&|!]))')
    LEVELS = [('||',), ('&&',), ('|',), ('&',), ('==', '!='), ('<', '<=', '>', '>='), ('<<', '>>'), ('+', '-'), ('*', '/')]

    def __init__(self, md, text):
        self.text, self.tokens, at = text, [], 0
        while at < len(text.rstrip()):
            m = self.TOKEN.match(text, at)
            if m is None:
                raise ValueError('size %r: not understood at %d' % (text, at))
            number, enum, quoted, plain, name, op = m.groups()
            if number is not None:
                self.tokens.append(('number', int(number, 0)))
            elif enum is not None:
                self.tokens.append(('number', md.enums[enum][quoted if quoted is not None else plain]))
            elif name is not None:
                self.tokens.append(('name', name))
            else:
                self.tokens.append(('op', op))
            at = m.end()
        self.at = 0
        self.tree = self._ternary()
        if self.at != len(self.tokens):
            raise ValueError('size %r: something left over' % text)
        self.names = sorted({t[1] for t in self.tokens if t[0] == 'name'})

    def _peek(self, *ops):
        return self.at < len(self.tokens) and self.tokens[self.at][0] == 'op' and self.tokens[self.at][1] in ops

    def _ternary(self):
        condition = self._binary(0)
        if self._peek('?'):
            self.at += 1
            yes = self._ternary()
            assert self._peek(':'), self.text
            self.at += 1
            return ('?', condition, yes, self._ternary())
        return condition

    def _binary(self, level):
        if level == len(self.LEVELS):
            return self._unary()
        left = self._binary(level + 1)
        while self._peek(*self.LEVELS[level]):
            op = self.tokens[self.at][1]
            self.at += 1
            left = (op, left, self._binary(level + 1))
        return left

    def _unary(self):
        if self._peek('!', '-'):
            op = self.tokens[self.at][1]
            self.at += 1
            return (op + 'x', self._unary())
        if self._peek('('):
            self.at += 1
            inside = self._ternary()
            assert self._peek(')'), self.text
            self.at += 1
            return inside
        kind, value = self.tokens[self.at]
        assert kind in ('number', 'name'), self.text
        self.at += 1
        return (kind, value)

    def bits(self, ops, node=None):
        """Its value for an instruction with the operands `ops`."""
        node = self.tree if node is None else node
        op = node[0]
        if op == 'number':
            return node[1]
        if op == 'name':
            return ops[node[1]]
        if op == '?':
            return self.bits(ops, node[2]) if self.bits(ops, node[1]) else self.bits(ops, node[3])
        if op == '!x':
            return int(not self.bits(ops, node[1]))
        if op == '-x':
            return -self.bits(ops, node[1])
        a = self.bits(ops, node[1])
        if op == '||':
            return int(bool(a or self.bits(ops, node[2])))
        if op == '&&':
            return int(bool(a and self.bits(ops, node[2])))
        b = self.bits(ops, node[2])
        return {'|': lambda: a | b, '&': lambda: a & b, '==': lambda: int(a == b), '!=': lambda: int(a != b),
                '<': lambda: int(a < b), '<=': lambda: int(a <= b), '>': lambda: int(a > b), '>=': lambda: int(a >= b),
                '<<': lambda: a << b, '>>': lambda: a >> b, '+': lambda: a + b, '-': lambda: a - b, '*': lambda: a * b,
                '/': lambda: a // b}[op]()


def registers(span, ops):
    """How many registers an operand takes: `span` is that number, or a `Size` to work out for the
    operands `ops` of the instruction. Something of no size still names a register: one."""
    if isinstance(span, int):
        return span
    return max(1, -(-span.bits(ops) // 32))


class Uses:
    """What the instructions of a class read and write, from its description."""

    def __init__(self, md, c):
        self.reads, self.writes = [], []      # (resource letter, operand name, registers it spans: see `registers`)
        def names(prop):
            return re.findall(r'INDEX\((\w+)\)', c.properties.get(prop, ''))
        def span(pred, operand):
            text = c.predicates.get(pred, '').strip()
            if re.fullmatch(r'\d+', text):
                return max(1, -(-int(text) // 32))
            if not text:
                return 1
            if pred == 'ISRC_A_SIZE' and operand == 'Ra' and re.search(r'_uniform__Ra32', c.name):
                # An address with a uniform part and a register of 32 bits: the description says 64
                # for the address as a whole. The uniform part has them, in two registers; the
                # register read is one (as nvdisasm's chart of the lives of registers marks them:
                # uses_check.py).
                return 1
            size = Size(md, text)
            # Every name in it must be an operand of the class, with a value in each instruction.
            assert all(n in c.operands for n in size.names), (c.name, pred, text)
            return size
        written = {n: span(p, n) for prop, p in (('IDEST_OPERAND_MAP', 'IDEST_SIZE'), ('IDEST2_OPERAND_MAP', 'IDEST2_SIZE')) for n in names(prop)}
        read = {n: span(p, n) for prop, p in (('ISRC_A_OPERAND_MAP', 'ISRC_A_SIZE'), ('ISRC_B_OPERAND_MAP', 'ISRC_B_SIZE'), ('ISRC_C_OPERAND_MAP', 'ISRC_C_SIZE')) for n in names(prop)}
        for name, o in c.operands.items():
            if o.type in M.REGISTER_TYPES or o.type in M.UNIFORM_TYPES:
                letter = 'R' if o.type in M.REGISTER_TYPES else 'U'
                if name in written:
                    self.writes.append((letter, name, written[name]))
                else:
                    self.reads.append((letter, name, read.get(name, 1)))   # addresses and the like: read
            elif o.type in ('Predicate', 'UniformPredicate'):
                letter = 'P' if o.type == 'Predicate' else 'Q'
                (self.writes if name in WRITTEN_PREDICATES else self.reads).append((letter, name, 1))


FINE = False     # -f: by mnemonic of the writer, and a table for compare.py


def main():
    global FINE
    if sys.argv[1] == '-f':
        FINE = True
        del sys.argv[1]
    sm, paths = int(sys.argv[1]), sys.argv[2:]
    md = M.load(sm)
    dec = M.Decoder(md)
    sets = operation_sets(sm)
    cache = {}
    def info(w):
        r = cache.get(w)
        if r is None:
            cs = dec.decode(w)
            if len(cs) != 1:
                r = None
            else:
                c = cs[0]
                ops = dec.operands(c, w)
                uses = Uses(md, c)
                kind = kind_of(sets, c.mnemonic)
                zero = {'R': 255, 'U': 63, 'P': 7, 'Q': 7}
                reads = [(l, ops[n] + i, 'guard' if n == 'Pg' else 'data') for l, n, s in uses.reads if n in ops and ops[n] != zero[l] for i in range(registers(s, ops))]
                writes = [(l, ops[n] + i) for l, n, s in uses.writes if n in ops and ops[n] != zero[l] for i in range(registers(s, ops))]
                sched = (w >> 105) & 0x1f
                barrier = (w >> 110) & 7       # the scoreboard the result is announced on, 7 for none
                relative = [n for n, o in c.operands.items() if o.type == 'RSImm' and n in ops]
                target = None
                if relative:
                    # The field holds the distance from the next instruction, in units of 4 bytes, signed.
                    for step, fields, data in dec.plan(c):
                        if step == 'scaled' and data[0] == relative[0]:
                            v = fields[0].get(w)
                            if v >> (len(fields[0]) - 1):
                                v -= 1 << len(fields[0])
                            target = v * data[1]
                r = (kind, reads, writes, sched & 0xf, sched >> 4, barrier, target, c.mnemonic)
            cache[w] = r
        return r
    least = {}           # (writer kind, reader kind, resource, role) -> (distance, example)
    histogram = collections.defaultdict(collections.Counter)
    current, code = None, []
    def function(name, code):
        targets = set()
        for off, w in code:
            r = info(w)
            if r and r[6] is not None:
                targets.add(off + 16 + r[6])
        t, last = 0, {}
        for off, w in code:
            r = info(w)
            if off in targets or r is None:
                last = {}
            if r is None:
                continue
            kind, reads, writes, stall, trans, barrier, target, mnemonic = r
            for letter, reg, role in reads:
                p = last.get((letter, reg))
                if p is not None:
                    key = (p[2] if FINE else p[1], kind, letter, role)
                    d = t - p[0]
                    histogram[key][d] += 1
                    if key not in least or d < least[key][0]:
                        least[key] = (d, '%s+0x%x: %s then %s' % (name[:40], off, p[2], mnemonic))
            for letter, reg in writes:
                last[(letter, reg)] = (t, kind, mnemonic) if barrier == 7 else None
            t += stall
            if kind == 'BRU':
                last = {}
    for path in paths:
        for name, off, w in instructions(path):
            if name != current:
                if code:
                    function(current, code)
                current, code = name, []
            code.append((off, w))
    if code:
        function(current, code)
    if FINE:
        for key in sorted(least):
            h = histogram[key]
            print('\t'.join([key[2], key[3], key[0], key[1], str(least[key][0]), str(h[least[key][0]]), str(sum(h.values())), ','.join('%d:%d' % (d, n) for d, n in sorted(h.items()) if d <= 40)]))
        return
    print('sm_%d: least distance from writer to reader (and how often: least, +1, +2), by kind' % sm)
    for key in sorted(least, key=lambda k: (k[2], k[3], k[0], k[1])):
        d, example = least[key]
        h = histogram[key]
        total = sum(h.values())
        if total < 20:
            continue
        print('  %s %-5s %-10s -> %-10s  %2d  (%d, %d, %d of %d)   %s' % (key[2], key[3], key[0], key[1], d, h[d], h[d + 1], h[d + 2], total, example))


if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main()

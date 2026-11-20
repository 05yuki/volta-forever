#!/usr/bin/env python3
"""Control-flow graph, reaching definitions, and whole-place analysis for Q.

Given a function (its decoded sm_75 instructions and its jump tables), this module builds:
- the control-flow graph: successors and predecessors of each instruction
- reaching definitions for registers (with ENTRY sentinel for the function's entry state)
- whole places and uniform branches (section 7 of docs/pairs.md)
- which internal calls read and write which registers (for the checker)

The helpers `relative`, `signed`, and `tables_of` from translate.py are the only parts of
translate.py used here.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from translate import relative, signed
from machine import ZERO

# Sentinel: a reaching definition from the function entry (the register was never written
# on this way). Condition 4 says "on no way to i is one of these four registers as the
# function was entered with it": the checker must object when this sentinel is in the set.
ENTRY = -1


# ---------------------------------------------------------------------------
# Helpers: predicates and guards
# ---------------------------------------------------------------------------

def is_unguarded(d):
    """True when instruction d runs unconditionally (no predicate guard)."""
    ops = d.ops
    pg_true = ops.get('Pg', 7) == 7 and not ops.get('Pg@not', 0)
    upg_true = ops.get('UPg', 7) == 7 and not ops.get('UPg@not', 0)
    return pg_true and upg_true


def guard_predicate(d):
    """Return (letter, predicate_register, negated) if the instruction is guarded, else None."""
    ops = d.ops
    if 'Pg' in ops and ops['Pg'] != 7:
        return ('P', ops['Pg'], bool(ops.get('Pg@not', 0)))
    if 'UPg' in ops and ops['UPg'] != 7:
        return ('UP', ops['UPg'], bool(ops.get('UPg@not', 0)))
    return None


def is_guarded(d):
    return not is_unguarded(d)


def writes_zero(d):
    """True if d, under no predicate, writes zero to all its destination registers."""
    if is_guarded(d):
        return False
    ops = d.ops
    m = d.mnemonic
    if m == 'CS2R' and ops.get('SRa', 0) == 255:
        return True
    if m == 'IMAD' and ops.get('Ra', -1) == 255 and ops.get('Rc', -1) == 255:
        return True
    if m == 'IMAD' and ops.get('Ra', -1) == 255 and ops.get('Rb', -1) == 255 and ops.get('Rc', -1) == 255:
        return True
    if m == 'MOV':
        if 'Rb' in ops and ops['Rb'] == 255:
            return True
        if 'Sb' in ops and ops['Sb'] == 0 and 'Rb' not in ops:
            return True
    if m == 'LOP3' and ops.get('imm8', -1) == 0:
        return True
    return False


# State values for register states in the checker.
# CONFLICT is distinct from None; it means "T on one way, Q on another" and nothing replaces it.
CONFLICT = 'X'


# ---------------------------------------------------------------------------
# Control flow
# ---------------------------------------------------------------------------

# Which mnemonics park or remove lanes.
LANES = {'BSYNC', 'BREAK', 'KILL', 'PBK', 'PCNT', 'PEXIT', 'PLONGJMP', 'PRET', 'SSY', 'SYNC', 'BRK', 'CONT', 'LONGJMP', 'RTT'}


class FlowError:
    def __init__(self, index, reason):
        self.index, self.reason = index, reason
    def __repr__(self):
        return 'FlowError(%d, %s)' % (self.index, self.reason)


class CFG:
    """Control-flow graph for a function."""

    def __init__(self, src, tables=None):
        self.src = src
        self.n = len(src)
        self.tables = tables or {}
        self.error = None
        self.succ = [[] for _ in range(self.n)]
        self.pred = [[] for _ in range(self.n)]
        self.entries = set()
        self.has_calls = False
        self.call_targets = {}   # call index -> callee entry index (or None for external)
        self.parks_lanes = set()
        self.aligned = set()
        self.kind = ['plain'] * self.n

    def build(self, mc):
        src, n = self.src, self.n
        tables = self.tables

        # Classify each instruction and resolve relative targets
        named = [None] * n
        for i, d in enumerate(src):
            rel = relative(mc, 75, d)
            if rel is not None:
                f, sc = rel
                t = 16 * i + 16 + signed(f, d.word) * sc
                named[i] = t // 16 if t % 16 == 0 and 0 <= t < 16 * n else None
        self.named = named

        # Collect meeting places: targets of BSSY, PBK, SSY, PCNT
        meets = sorted({named[i] for i in range(n) if src[i].mnemonic in ('BSSY', 'PBK', 'SSY', 'PCNT') and named[i] is not None})

        for i, d in enumerate(src):
            m, name, ops = d.mnemonic, d.c.name, d.ops
            ug = is_unguarded(d)

            if d.c.name == 'hmma_x8_' or d.c.name.startswith('ldsm'):
                if ug:
                    self.aligned.add(i)
            if m == 'WARPSYNC':
                self.aligned.add(i)

            after = [i + 1] if i + 1 < n else []

            if m == 'BRA':
                pp_true = ops.get('Pp', 7) == 7 and not ops.get('Pp@not', 0)
                cond = ops.get('cond', 0)
                # BRA under no predicate with no condition or with .U (cond=1) is unconditional.
                # BRA.DIV (cond=2) and BRA.CONV (cond=3) go by what the lanes are doing.
                if ug and pp_true and cond in (0, 1):
                    kind = 'jump'
                    out = [named[i]] if named[i] is not None else []
                else:
                    kind = 'cond'
                    out = after + ([named[i]] if named[i] is not None else [])
            elif m == 'EXIT':
                kind = 'exit'
                out = [] if ug else after
                if not ug:
                    self.parks_lanes.add(i)
            elif m == 'RET':
                kind = 'other'
                out = [] if ug else after
            elif m == 'CALL':
                kind = 'call'
                # Model call as: successor is {i+1}. Callee's body not in CFG.
                out = after
                if name.startswith('call_rel'):
                    self.has_calls = True
                    self.call_targets[i] = named[i]
                else:
                    self.call_targets[i] = None  # external
            elif m == 'BRX':
                if name == 'brx_' and 16 * i in tables:
                    kind = 'other'
                    targets = [t // 16 for t in tables[16 * i] if t % 16 == 0 and 0 <= t < 16 * n]
                    out = targets + ([] if ug else after)
                else:
                    kind = 'unknown'
                    self.error = FlowError(i, '%s without a known jump table' % m)
                    out = after
            elif m in ('JMX', 'JMP'):
                kind = 'unknown'
                self.error = FlowError(i, '%s to unknown address' % m)
                out = after
            elif m in ('KILL', 'BREAK'):
                kind = 'other'
                # Lanes that leave go to meeting places (where BSYNC reconverges them).
                out = ([] if ug else after) + meets
                self.parks_lanes.add(i)
            elif m == 'BSYNC':
                kind = 'other'
                self.parks_lanes.add(i)
                out = after
            elif m in LANES:
                kind = 'other'
                self.parks_lanes.add(i)
                # Other lane-parking instructions also go to meeting places.
                out = after + meets
            elif m == 'BSSY':
                kind = 'other'
                out = after  # no edge to the meeting place
            else:
                kind = 'plain'
                out = after

            self.kind[i] = kind
            self.succ[i] = sorted(set(out))

        # Build predecessors
        for i in range(n):
            for j in self.succ[i]:
                self.pred[j].append(i)

        # Entries
        self.entries.add(0)
        for i in range(n):
            for j in self.succ[i]:
                if j != i + 1:
                    self.entries.add(j)

        # Compute what each internal call reads and writes
        self._compute_called()

        # Whole places
        self._compute_whole()

        return self

    def removes(self, i):
        """Instruction i may leave fewer (or more) lanes active."""
        d = self.src[i]
        return d.mnemonic in LANES or (d.mnemonic in ('EXIT', 'RET') and not is_unguarded(d))

    def is_conditional_branch(self, i):
        return self.kind[i] == 'cond'

    def is_uniform(self, i):
        """A conditional branch that all lanes take the same way (section 7)."""
        return getattr(self, '_uniform', [False] * self.n)[i]

    # ---- what internal calls read and write --------------------------------

    def _compute_called(self):
        """For each internal call, compute what registers its callee reads and writes.
        A callee that calls another function: conservatively everything."""
        src, n = self.src, self.n
        self.called = {}  # call_index -> (reads: set, writes: set) or None (anything)

        for call_i, target in self.call_targets.items():
            if target is None:
                self.called[call_i] = None
                continue
            # Walk the callee's body
            body, work, simple = set(), [target], True
            while work:
                k = work.pop()
                if k in body:
                    continue
                body.add(k)
                if self.kind[k] == 'call':
                    simple = False
                    break
                work += self.succ[k]
            if not simple:
                self.called[call_i] = None
                continue
            # Compute what the callee reads (live at entry) and writes
            alive = {k: set() for k in body}
            changed = True
            while changed:
                changed = False
                for k in sorted(body, reverse=True):
                    after = set()
                    for t in self.succ[k]:
                        if t in alive:
                            after |= alive[t]
                    kills = {r for l, r in src[k].writes if l == 'R'} if is_unguarded(src[k]) else set()
                    now = {r for l, r, *_ in src[k].reads if l == 'R'} | (after - kills)
                    if now != alive[k]:
                        alive[k], changed = now, True
            writes = {r for k in body for l, r in src[k].writes if l == 'R'}
            self.called[call_i] = (alive.get(target, set()), writes)

    def call_reads(self, i, r):
        """Does the call at i (or its callee) read register r?"""
        c = self.called.get(i)
        return c is None or r in c[0]

    def call_writes(self, i, r):
        """Does the call at i (or its callee) write register r?"""
        c = self.called.get(i)
        return c is None or r in c[1]

    # ---- whole places (section 7) ------------------------------------------

    def _compute_whole(self):
        n, src = self.n, self.src

        # leads_to_aligned: straight-line code from i reaches an aligned instruction.
        # No conditional branch, no lane-removing instruction, no call on the way.
        lead = [None] * n

        def leads(i):
            path = []
            while True:
                if i is None or i >= n or lead[i] is False or i in path:
                    result = False
                    break
                if lead[i] is True or i in self.aligned:
                    result = True
                    break
                path.append(i)
                k = self.kind[i]
                if k == 'plain' and not self.removes(i):
                    i = i + 1 if i + 1 < n else None
                elif k == 'jump':
                    i = self.named[i]
                else:
                    result = False
                    break
            for p in path:
                lead[p] = result
            if i is not None and 0 <= i < n and lead[i] is None:
                lead[i] = result
            return result

        for i in range(n):
            if lead[i] is None:
                leads(i)
        self.lead = [bool(x) for x in lead]

        # uniform: a conditional branch at a whole place where one way leads to aligned
        self._uniform = [False] * n
        for i in range(n):
            if self.kind[i] == 'cond':
                target = self.named[i]
                ft = i + 1 if i + 1 < n else None
                if (target is not None and self.lead[target]) or (ft is not None and self.lead[ft]):
                    self._uniform[i] = True

        # Whole places: computed as a least fixed point.
        # A place is whole by lead, or by being entered only by whole ways.
        # A WAY from p to q is whole when:
        #   p is whole, and
        #   p does not remove lanes, and
        #   p is 'plain' or 'jump', or (p is 'cond' and p is uniform).
        # A PLACE is whole when:
        #   lead[i], OR
        #   every way into it is whole.

        whole = list(self.lead)

        # After: what follows an aligned instruction in straight-line code,
        # up to and including the next branch. Walk forward, stopping at
        # entries from elsewhere, calls, lane-removing instructions.
        for a in self.aligned:
            j = a
            while True:
                whole[j] = True
                d = src[j]
                if self.kind[j] in ('cond', 'jump'):
                    break
                if self.removes(j) or self.kind[j] == 'call':
                    break
                nxt = j + 1
                if nxt >= n or nxt not in set(self.succ[j]):
                    break
                if nxt in self.entries and nxt != a:
                    # entered from elsewhere: the AFTER rule doesn't carry
                    # (a multi-entry place is whole only if every way in is)
                    break
                j = nxt

        changed = True
        while changed:
            changed = False
            for i in range(n):
                if whole[i]:
                    continue
                preds = self.pred[i]
                if not preds:
                    continue
                if all(self._way_whole(p, whole) for p in preds):
                    whole[i] = True
                    changed = True

        self.whole = whole

    def _way_whole(self, p, whole):
        """The way FROM instruction p is whole: 32 lanes leave it together."""
        if not whole[p] or self.removes(p):
            return False
        k = self.kind[p]
        if k in ('plain', 'jump'):
            return True
        if k == 'cond' and self._uniform[p]:
            return True
        return False


# ---------------------------------------------------------------------------
# Reaching definitions (with ENTRY sentinel)
# ---------------------------------------------------------------------------

def reaching_definitions(cfg, watch=None):
    """Compute reaching definitions for registers.

    watch: if given, only track these registers (for performance).
    Returns: reach[i] = {register: frozenset of defining instructions}.
    ENTRY (-1) in the set means "as the function was entered with it" (never written on this way).

    A call that may write register r adds None to the set (conservatively).
    We use ENTRY for the function entry and None for call writes.
    """
    n = cfg.n
    src = cfg.src

    # What each instruction writes for each register
    writes_at = [[] for _ in range(n)]
    kills_at = [set() for _ in range(n)]
    for i, d in enumerate(src):
        ug = is_unguarded(d)
        for letter, reg in d.writes:
            if letter == 'R' and (watch is None or reg in watch):
                writes_at[i].append(reg)
                if ug:
                    kills_at[i].add(reg)
        # Calls
        if cfg.kind[i] == 'call':
            c = cfg.called.get(i)
            if c is None:
                # External or complex call: may write anything
                for reg in (watch or set()):
                    if reg not in kills_at[i]:
                        writes_at[i].append(reg)
                        # Don't kill: we don't know for sure
            else:
                _, call_writes = c
                for reg in call_writes:
                    if (watch is None or reg in watch) and reg not in writes_at[i]:
                        writes_at[i].append(reg)
                        # Calls write unconditionally in their body, but the
                        # call itself is conditional on whether the callee actually
                        # writes it. Conservatively: don't kill (add to defs).

    # Initialize with ENTRY for all watched registers
    if watch is not None:
        init = {r: frozenset([ENTRY]) for r in watch}
    else:
        # Find all registers that appear
        all_regs = set()
        for d in src:
            for l, r, *_ in d.reads:
                if l == 'R':
                    all_regs.add(r)
            for l, r in d.writes:
                if l == 'R':
                    all_regs.add(r)
        init = {r: frozenset([ENTRY]) for r in all_regs}

    reach = [None] * n
    reach[0] = dict(init)

    work = [0]
    in_work = [False] * n
    in_work[0] = True

    while work:
        i = work.pop()  # pop from end for speed (DFS-like)
        in_work[i] = False

        state = reach[i]
        if state is None:
            continue

        # Compute output state
        out = state
        if writes_at[i]:
            out = dict(state)
            for r in writes_at[i]:
                if r in kills_at[i]:
                    out[r] = frozenset([i])
                else:
                    out[r] = (state.get(r, frozenset()) | frozenset([i]))

        for s in cfg.succ[i]:
            old = reach[s]
            if old is None:
                reach[s] = dict(out)
                if not in_work[s]:
                    work.append(s)
                    in_work[s] = True
            else:
                changed = False
                for r, defs in out.items():
                    prev = old.get(r)
                    if prev is None:
                        old[r] = defs
                        changed = True
                    elif not defs <= prev:
                        old[r] = prev | defs
                        changed = True
                if changed and not in_work[s]:
                    work.append(s)
                    in_work[s] = True

    return reach

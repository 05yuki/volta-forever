#!/usr/bin/env python3
"""pending.py SM CUBIN...: how many instructions native code leaves pending on one scoreboard.

An instruction that takes its time (a load, a shuffle, a tensor-core product) names a scoreboard
for its result, or for having read its operands, and what depends on it waits for that scoreboard.
Does a scoreboard count the instructions that named it, and how many? This looks at what the
compiler does: in straight-line code, for every wait on a scoreboard, how many instructions named
that scoreboard for their result since the last wait on it (a result may take any time: a load
that misses the cache), how many named it for having read (which takes a few cycles, whatever
the instruction), and how many of either in the `WINDOW` cycles just before the wait (by the
stalls).

What it showed: in a quarter of PyTorch's and cuBLAS's sm_70 code, up to 12
instructions name one scoreboard in the 24 cycles before a wait on it (four loads from global
memory within 17 cycles, then the instruction that reads the second waits), and up to 75 between
two waits. A scoreboard counts, and well beyond three: the sequences of the translation may put
several shuffles on one. (The question came from a wrong reading of an experiment, in which
products seemed to fail from the fourth instruction on a scoreboard: what failed was something
else, a result read too soon after a product.)
"""
import collections, multiprocessing, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import elf, machine
from machine import READ_BARRIER, WAIT, WRITE_BARRIER, stall_of

WINDOW = 24


def one(job):
    sm, path = job
    mc = machine.Machine()
    near, since, kinds, examples = collections.Counter(), collections.Counter(), collections.Counter(), {}
    results, reads = collections.Counter(), collections.Counter()
    e = elf.Elf(open(path, 'rb').read())
    for s in e.sections:
        if not (s.name.startswith('.text.') and s.type == 1):
            continue
        words = [int.from_bytes(s.data[i:i + 16], 'little') for i in range(0, len(s.data) - 15, 16)]
        decoded = [mc.decode(sm, w) for w in words]
        # Where straight-line code starts again: after a branch, and where a branch arrives.
        starts = {0}
        for i, d in enumerate(decoded):
            if d is None:
                starts.add(i + 1)
                continue
            if d.kind == 'BRU':
                starts.add(i + 1)
                rel = machine_relative(mc, sm, d)
                if rel is not None:
                    starts.add(rel // 16 + i + 1)
        named = [[] for _ in range(6)]             # for each scoreboard: (time, kind, for its result) of what named it since the last wait on it
        t = 0
        for i, (w, d) in enumerate(zip(words, decoded)):
            if i in starts:
                named = [[] for _ in range(6)]
            if d is None:
                continue
            wait = (w >> WAIT) & 63
            for b in range(6):
                if wait >> b & 1 and named[b]:
                    recent = [k for when, k, _ in named[b] if t - when <= WINDOW]
                    near[len(recent)] += 1
                    since[len(named[b])] += 1
                    wrote = [k for _, k, result in named[b] if result]
                    results[len(wrote)] += 1
                    reads[len(named[b]) - len(wrote)] += 1
                    if len(wrote) >= 3:
                        kinds[(len(wrote), '+'.join(sorted(set(wrote))))] += 1
                        examples.setdefault(len(wrote), '%s %s+0x%x' % (os.path.basename(path), s.name[6:36], 16 * i))
                    named[b] = []
            wr, rd = (w >> WRITE_BARRIER) & 7, (w >> READ_BARRIER) & 7
            if wr < 6:
                named[wr].append((t, d.kind, True))
            if rd < 6 and rd != wr:
                named[rd].append((t, d.kind, False))
            t += stall_of(w)
        mc._cache[sm].clear()
    return near, since, kinds, examples, results, reads


def machine_relative(mc, sm, d):
    """The byte offset a relative branch adds to the address after it, or None."""
    import translate
    rel = translate.relative(mc, sm, d)
    if rel is None:
        return None
    return translate.signed(rel[0], d.word) * rel[1]


def main():
    sm, paths = int(sys.argv[1]), sys.argv[2:]
    near, since, kinds, examples = collections.Counter(), collections.Counter(), collections.Counter(), {}
    results, reads = collections.Counter(), collections.Counter()
    with multiprocessing.Pool(32) as pool:
        for a, b, c, d, r, q in pool.imap_unordered(one, [(sm, p) for p in paths]):
            near.update(a)
            since.update(b)
            kinds.update(c)
            results.update(r)
            reads.update(q)
            for k, v in d.items():
                examples.setdefault(k, v)
    def table(title, counts, last=12):
        print(title)
        for n in sorted(counts):
            if n <= last:
                print('  %2d  %9d   %s' % (n, counts[n], examples.get(n, '') if counts is results and n >= 3 else ''))
        more = sum(v for n, v in counts.items() if n > last)
        if more:
            print('  more %8d   (up to %d)' % (more, max(counts)))
    table('waits on a scoreboard, by how many instructions named it for their result since the last wait on it:', results)
    table('by how many named it for having read:', reads)
    table('by how many named it, for either, in the %d cycles before the wait:' % WINDOW, near)
    print('the groups of three results or more, by the kinds of their instructions:')
    for (n, k), count in sorted(kinds.items(), key=lambda kv: (-kv[0][0], -kv[1]))[:12]:
        print('  %2d  %9d   %s' % (n, count, k))


if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main()

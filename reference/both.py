#!/usr/bin/env python3
"""both.py OUT CUBIN...: decode the instructions of sm_75 code with the sm_75 and the sm_70 descriptions.

Writes OUT (a table: count, word, sm_75 class, sm_70 class or '-', nvdisasm's text as sm_75, and as
sm_70 when it differs) for every distinct instruction word, and prints a summary by sm_75 class:
how many instructions are also sm_70 instructions with the same text, bit for bit.
"""
import collections, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import md as M
from probe import instructions
from check import disassemble


def main():
    out, paths = sys.argv[1], sys.argv[2:]
    md75, md70 = M.load(75), M.load(70)
    d75, d70 = M.Decoder(md75), M.Decoder(md70)
    count = collections.Counter()
    for path in paths:
        for _, _, w in instructions(path):
            count[w] += 1
    words = list(count)
    t75, t70 = [], []
    for i in range(0, len(words), 4096):
        t75 += disassemble(words[i:i + 4096], 75)[0]
    # nvdisasm stops at the first word it does not know: as sm_70, one word at a time would be slow,
    # so only the words the sm_70 description accepts are given to it, in pieces.
    c75 = [d75.decode(w) for w in words]
    c70 = [[c for c in d70.decode(w) if d70.unused_clear(c, w)] for w in words]
    # As sm_70, the words the sm_70 description does not accept are replaced by NOP: nvdisasm stops at
    # a word it does not know, and every word must stay at its address (branch targets are printed
    # from it).
    NOP = 0x000fc000000000000000000000007918
    accepted = set(i for i, cs in enumerate(c70) if len(cs) == 1)
    text70 = {}
    for i in range(0, len(words), 4096):
        piece = list(range(i, min(i + 4096, len(words))))
        texts, err = disassemble([words[j] if j in accepted else NOP for j in piece], 70)
        if any(t is None for t in texts):          # nvdisasm refused one: the piece again, one word at a time, in place
            texts = []
            for j in piece:
                alone = [NOP] * (j - i) + [words[j] if j in accepted else NOP]
                texts.append(disassemble(alone, 70)[0][-1])
        for j, t in zip(piece, texts):
            if j in accepted:
                text70[j] = t
    by_class = collections.defaultdict(collections.Counter)
    with open(out, 'w') as f:
        for i, w in enumerate(words):
            a = c75[i][0].name if len(c75[i]) == 1 else '?%d' % len(c75[i])
            b = c70[i][0].name if len(c70[i]) == 1 else '-'
            if b == '-':
                verdict = 'not sm_70'
            elif text70.get(i) is None:
                verdict = 'description accepts, nvdisasm refuses'
            elif text70[i] == t75[i]:
                verdict = 'same'
            else:
                verdict = 'other text'
            by_class[a][verdict] += count[w]
            f.write('%d\t%032x\t%s\t%s\t%s\t%s\n' % (count[w], w, a, b, t75[i], '' if verdict in ('same', 'not sm_70') else text70.get(i)))
    total = collections.Counter()
    for a, v in by_class.items():
        total.update(v)
    n = sum(total.values())
    print('%d instructions, %d distinct: %s' % (n, len(words), ', '.join('%s %d (%.1f %%)' % (k, c, 100.0 * c / n) for k, c in total.most_common())))
    print('sm_75 classes whose instructions are not all the same on sm_70, by count:')
    rows = sorted(((sum(v.values()) - v['same'], a, v) for a, v in by_class.items()), reverse=True)
    for other, a, v in rows[:70]:
        if other:
            print('  %9d  %-38s %s' % (other, a, dict(v)))


if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main()

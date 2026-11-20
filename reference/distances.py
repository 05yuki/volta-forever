#!/usr/bin/env python3
"""distances.py TABLE > distances-sm70.tsv: the distance to keep between a writer and a reader, from
what native sm_70 code keeps (TABLE: merge.py's, of latency.py -f over the corpus).

The compiler never puts a reader closer than the latency of its writer, and very often puts it
exactly there: the distance kept is the smallest one that many samples have. A handful of samples
below it are readers that the measurement took for readers and are not (an operand that is not
really read): a distance counts only if it has 20 samples and one in two hundred of the pair's.

One line per kind of writer and kind of reader (the compiler's latency groups), from all the
mnemonics of the kind together, and one per mnemonic whose own distance, well sampled, is longer.
"""
import collections, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from latency import operation_sets, kind_of


def kept(histogram):
    """The smallest distance with enough samples, or None when no distance has enough."""
    total = sum(histogram.values())
    for d in sorted(histogram):
        if histogram[d] >= 20 and histogram[d] * 200 >= total:
            return d
    return None


def main():
    sets = operation_sets(70)
    by_kind = collections.defaultdict(collections.Counter)
    by_mnemonic = {}
    for line in open(sys.argv[1]):
        f = line.rstrip('\n').split('\t')
        letter, role, mnemonic, reader = f[:4]
        h = collections.Counter({int(p.split(':')[0]): int(p.split(':')[1]) for p in f[7].split(',')})
        by_mnemonic[(letter, role, mnemonic, reader)] = h
        by_kind[(letter, role, kind_of(sets, mnemonic), reader)].update(h)
    print('# resource, role, kind or mnemonic of the writer, kind of the reader, distance, samples at it, samples')
    kinds = {}
    for key in sorted(by_kind):
        d = kept(by_kind[key])
        if d is not None:
            kinds[key] = d
            print('\t'.join(list(key) + [str(d), str(by_kind[key][d]), str(sum(by_kind[key].values()))]))
    for key in sorted(by_mnemonic):
        letter, role, mnemonic, reader = key
        d = kept(by_mnemonic[key])
        base = kinds.get((letter, role, kind_of(sets, mnemonic), reader))
        if d is not None and base is not None and d > base and sum(by_mnemonic[key].values()) >= 500:
            print('\t'.join([letter, role, '=' + mnemonic, reader, str(d), str(by_mnemonic[key][d]), str(sum(by_mnemonic[key].values()))]))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main()

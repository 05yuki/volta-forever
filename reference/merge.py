#!/usr/bin/env python3
"""merge.py TABLE...: one table of `latency.py -f` from several: for each writer (mnemonic), kind of
reader, resource and role, the least distance seen, how often, the samples, and how many samples at
each distance up to 40."""
import collections, sys
if len(sys.argv) < 2:
    sys.exit(__doc__)
total, histogram = collections.Counter(), collections.defaultdict(collections.Counter)
for name in sys.argv[1:]:
    for line in open(name):
        f = line.rstrip('\n').split('\t')
        if len(f) < 7:
            continue
        key = tuple(f[:4])
        total[key] += int(f[6])
        if len(f) > 7 and f[7]:
            for pair in f[7].split(','):
                d, n = pair.split(':')
                histogram[key][int(d)] += int(n)
        else:
            histogram[key][int(f[4])] += int(f[5])
for key in sorted(total):
    h = histogram[key]
    least = min(h)
    print('\t'.join(list(key) + [str(least), str(h[least]), str(total[key]), ','.join('%d:%d' % (d, n) for d, n in sorted(h.items()))]))

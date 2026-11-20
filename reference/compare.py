#!/usr/bin/env python3
"""compare.py LAT70 LAT75 [SAMPLES]: where is sm_75 code scheduled tighter than native sm_70 code?
The inputs are the tables of `latency.py -f` for code of each architecture."""
import sys
if len(sys.argv) < 3:
    sys.exit(__doc__)

def load(name):
    d = {}
    for line in open(name):
        f = line.rstrip('\n').split('\t')
        if len(f) == 7:
            d[tuple(f[:4])] = (int(f[4]), int(f[5]), int(f[6]))
    return d

a, b = load(sys.argv[1]), load(sys.argv[2])
enough = int(sys.argv[3]) if len(sys.argv) > 3 else 200
rows = [(b[k][0] - a[k][0], k, a[k], b[k]) for k in sorted(set(a) & set(b)) if a[k][2] >= enough and b[k][2] >= enough]
shorter = [r for r in rows if r[0] < 0]
print('writer -> kind of reader, with %d samples on both sides: %d pairs, %d shorter on sm_75, %d equal, %d longer'
      % (enough, len(rows), len(shorter), sum(r[0] == 0 for r in rows), sum(r[0] > 0 for r in rows)))
for diff, key, x, y in sorted(shorter):
    print('  %s %-5s %-12s -> %-9s  sm_70 %2d (%d of %d)   sm_75 %2d (%d of %d)' % (key[0], key[1], key[2], key[3], x[0], x[1], x[2], y[0], y[1], y[2]))
only75 = [(k, b[k]) for k in sorted(set(b) - set(a)) if b[k][2] >= enough]
print('%d pairs seen only in sm_75 code (with %d samples)' % (len(only75), enough))

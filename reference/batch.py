#!/usr/bin/env python3
"""batch.py OUT_DIR CUBIN...: translate many cubins (one process), writing OUT_DIR/<name> and one line
per function in OUT_DIR/<name>.report: `ok` or the reason it was refused (with offsets left out)."""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import machine, translate
if len(sys.argv) < 3:
    sys.exit(__doc__)

out_dir, paths = sys.argv[1], sys.argv[2:]
mc = machine.Machine()
for path in paths:
    name = os.path.basename(path)
    try:
        listing, lives = [], []
        out, report = translate.translate(open(path, 'rb').read(), mc, listing, lives)
    except Exception as e:                      # a defect of the prototype: the whole file is lost
        with open(os.path.join(out_dir, name + '.report'), 'w') as f:
            f.write('CRASH\t%s: %s\n' % (type(e).__name__, str(e)[:200]))
        continue
    open(os.path.join(out_dir, name), 'wb').write(out)
    with open(os.path.join(out_dir, name + '.report'), 'w') as f:
        for function, why in report.items():
            f.write('%s\t%s\n' % ('ok' if not why else re.sub(r'0x[0-9a-f]+', 'N', why), function))
    with open(os.path.join(out_dir, name + '.map'), 'w') as f:
        for row in listing:
            f.write('%s\t%x\t%x\t%s\n' % row)
    translate.write_lives(os.path.join(out_dir, name + '.map.lives'), lives)
    kinds = {}
    for _, _, _, what in listing:
        kinds[what] = kinds.get(what, 0) + 1
    with open(os.path.join(out_dir, name + '.kinds'), 'w') as f:
        for what, n in sorted(kinds.items()):
            f.write('%d\t%s\n' % (n, what))

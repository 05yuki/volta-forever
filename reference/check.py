#!/usr/bin/env python3
"""check.py SM CUBIN...: compare this directory's decoder with nvdisasm on the instructions of real code.

For every distinct instruction word: the class found must be unique, its mnemonic must be the one
nvdisasm prints, and the registers and predicates decoded (those that are not the zero register or
the true predicate, which nvdisasm often leaves out) must be the ones in nvdisasm's text.
"""
import collections, os, re, struct, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import md as M
from probe import instructions

NVDISASM = os.path.join(M.DATA, '../../../toolkits/cuda-12.9.2/bin/nvdisasm')
TOKEN = re.compile(r'(?<![A-Za-z0-9_.])(UR\d+|R\d+|UP[0-6]|P[0-6]|B\d+)(?![A-Za-z0-9_])')


def disassemble(words, sm):
    """nvdisasm's text for each word (None where it reports an error)."""
    if not os.path.exists(NVDISASM):
        sys.exit('%s is not there: this needs NVIDIA\'s disassembler, which scripts/fetch-all.sh redist fetches with the toolkits' % os.path.normpath(NVDISASM))
    out = {}
    with tempfile.NamedTemporaryFile(suffix='.bin') as f:
        for w in words:
            f.write(struct.pack('<QQ', w & (1 << 64) - 1, w >> 64))
        f.flush()
        r = subprocess.run([NVDISASM, '-b', 'SM%d' % sm, f.name], capture_output=True, text=True)
    for m in re.finditer(r'/\*([0-9a-f]+)\*/\s+(.*?);', r.stdout):
        out[int(m.group(1), 16) // 16] = m.group(2).strip()
    return [out.get(i) for i in range(len(words))], r.stderr


def registers(dec, c, word):
    """The registers and predicates of the instruction, as nvdisasm names them, without the defaults."""
    names = []
    for name, value in dec.operands(c, word).items():
        o = c.operands.get(name)
        if o is None:
            continue
        if o.type in M.REGISTER_TYPES and value != 255:
            names.append('R%d' % value)
        elif o.type in M.UNIFORM_TYPES and value != 63:
            names.append('UR%d' % value)
        elif o.type == 'Predicate' and value != 7:
            names.append('P%d' % value)
        elif o.type == 'UniformPredicate' and value != 7:
            names.append('UP%d' % value)
        elif o.type in ('BarrierRegister', 'BD'):
            names.append('B%d' % value)
    return sorted(names)


def main():
    sm = int(sys.argv[1])
    md = M.load(sm)
    dec = M.Decoder(md)
    words, seen = [], set()
    for path in sys.argv[2:]:
        for _, _, w in instructions(path):
            if w not in seen:
                seen.add(w)
                words.append(w)
    texts = []
    for i in range(0, len(words), 4096):   # an error stops nvdisasm: small pieces lose little
        t, _ = disassemble(words[i:i + 4096], sm)
        texts += t
    stats = collections.Counter()
    shown = collections.Counter()
    for w, text in zip(words, texts):
        classes = dec.decode(w)
        if text is None:
            stats['nvdisasm gives nothing'] += 1
            continue
        if len(classes) != 1:
            stats['%d classes' % len(classes)] += 1
            if shown['classes'] < 8:
                shown['classes'] += 1
                print('%d classes: %032x  %s  %s' % (len(classes), w, text, [c.name for c in classes]))
            continue
        c = classes[0]
        body = text.split(None, 1)[1] if text.startswith('@') else text
        mnemonic = body.split()[0].split('.')[0]
        if mnemonic != c.mnemonic:
            stats['mnemonic differs'] += 1
            if shown[(c.mnemonic, mnemonic)] < 1:
                shown[(c.mnemonic, mnemonic)] += 1
                print('mnemonic: %032x  %-50s  %s (%s)' % (w, text, c.name, c.mnemonic))
            continue
        mine, theirs = registers(dec, c, w), sorted(TOKEN.findall(text))
        if mine != theirs:
            stats['registers differ'] += 1
            if shown[c.name] < 1:
                shown[c.name] += 1
                print('registers: %032x  %-50s  %s: %s' % (w, text, c.name, ' '.join(mine)))
            continue
        stats['same'] += 1
    print('sm_%d: %d distinct instructions: %s' % (sm, len(words), ', '.join('%s %d' % kv for kv in stats.most_common())))


if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main()

#!/bin/bash
# The independent validator over a corpus, cubin by cubin, keeping nothing but counts:
#   tests/sass/validate-corpus.sh OUT CUBIN_OR_DIRECTORY...
# Each sm_75 cubin is translated by the reference translator (reference/translate.py; with $VF_BORROW as
# it is in the environment: unset is what the layer does by itself), validated (validate.py), and its
# translation and maps are removed at once (a run over PyTorch with its maps kept is 8 GB).
# In OUT: `counts.tsv` (cubin, problems, instructions and functions checked; `?` when the validator did
# not end with a count: it stopped, or ran out of $LIMIT seconds; `refused` when the translator itself
# stopped), `problems/` (the validator's whole output for every cubin that is not at 0), `checks.tsv`
# (what was checked and what was not, summed over the corpus: the validator's own count lines). A cubin
# already in `counts.tsv` is not done again: the run can be taken up where it stopped.
# $JOBS processes at once (default 16: the host of this tree is a file server), $LIMIT seconds a cubin
# (default 3600), $VALIDATOR a directory with another validate.py and what it needs (default: this
# tree's reference/), $TRANSLATOR the same for translate.py.
# The last line says how many cubins are at 0 problems; the exit status is 0 only if all are.
set -uo pipefail
W=$(cd "$(dirname "$0")/../.." && pwd)
out=$1; shift
[ $# -gt 0 ] || { echo "usage: $0 OUT CUBIN_OR_DIRECTORY..." >&2; exit 2; }
validator=${VALIDATOR:-$W/reference}
translator=${TRANSLATOR:-$W/reference}
mkdir -p "$out/problems" "$out/work" || exit 2
out=$(cd "$out" && pwd)
touch "$out/counts.tsv"
export out validator translator LIMIT=${LIMIT:-3600}
one() {
    cubin=$1
    key=$(basename "$(dirname "$cubin")")/$(basename "$cubin")
    cut -f1 "$out/counts.tsv" | grep -qxF "$key" && return 0
    tmp=$(mktemp -d "$out/work/XXXXXXXX")
    sizes=
    if ! python3 -I "$translator/translate.py" "$cubin" "$tmp/t.cubin" "$tmp/t.map" > "$tmp/translate.out" 2>&1 || [ ! -s "$tmp/t.cubin" ]; then
        cp "$tmp/translate.out" "$out/problems/$(echo "$key" | tr / _).translate.txt"
        result=refused
    else
        timeout "$LIMIT" python3 -I "$validator/validate.py" "$cubin" "$tmp/t.cubin" "$tmp/t.map" > "$tmp/validate.out" 2>&1
        result=$(sed -n '1s/^.* checked, \([0-9][0-9]*\) problems.*$/\1/p' "$tmp/validate.out")
        result=${result:-?}
        sizes=$(sed -n '1s/^.*: \([0-9][0-9]*\) instructions of \([0-9][0-9]*\) functions checked.*$/\1\t\2/p' "$tmp/validate.out")
        [ "$result" = 0 ] || cp "$tmp/validate.out" "$out/problems/$(echo "$key" | tr / _).txt"
        # The validator's count lines: "  N (kind) what".
        sed -n 's/^  \([0-9][0-9]*\) \((.*\)$/\1\t\2/p' "$tmp/validate.out" > "$tmp/checks"
        flock "$out/checks.lock" cat "$tmp/checks" >> "$out/checks.raw"
    fi
    rm -rf "$tmp"
    flock "$out/counts.lock" printf '%s\t%s\t%s\n' "$key" "$result" "$sizes" >> "$out/counts.tsv"
}
export -f one
for path in "$@"; do
    if [ -d "$path" ]; then find "$path" -name '*.cubin' | sort
    elif [ -f "$path" ]; then echo "$path"
    else echo "$0: $path: neither a cubin nor a directory" >&2; exit 2
    fi
done > "$out/list"
total=$(wc -l < "$out/list")
xargs -a "$out/list" -P "${JOBS:-16}" -I{} bash -c 'one "$1"' _ {}
rmdir "$out/work" 2>/dev/null
[ -f "$out/checks.raw" ] && awk -F'\t' '{ n[$2] += $1 } END { for (k in n) printf "%d\t%s\n", n[k], k }' "$out/checks.raw" | sort -t$'\t' -k1,1nr > "$out/checks.tsv"
zero=$(awk -F'\t' '$2 == "0"' "$out/counts.tsv" | wc -l)
done_=$(wc -l < "$out/counts.tsv")
echo "$zero of $total cubins at 0 problems ($done_ done; with problems: $(awk -F'\t' '$2 ~ /^[1-9]/' "$out/counts.tsv" | wc -l), validator did not end: $(awk -F'\t' '$2 == "?"' "$out/counts.tsv" | wc -l), translator stopped: $(awk -F'\t' '$2 == "refused"' "$out/counts.tsv" | wc -l))"
awk -F'\t' '{ i += $3; f += $4 } END { printf "%d instructions of %d functions checked\n", i, f }' "$out/counts.tsv"
[ -f "$out/checks.tsv" ] && { echo "not checked, summed:"; grep "(not checked)" "$out/checks.tsv" | head -12; }
[ "$zero" = "$total" ]

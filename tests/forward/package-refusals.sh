#!/bin/bash
# What scripts/package.sh refuses to pack, without a GPU and without touching dist/:
#   tests/forward/package-refusals.sh
# Each case packs a copy of this tree's dist/ (built with scripts/build.sh and
# scripts/build-forwarders.sh), in a temporary directory, with one thing wrong:
#   - a forwarding library is missing (libcublasLt.so.13): refused, and named;
#   - a forwarding library is older than the sources: refused, and named;
#   - the report is that of a run for one library only: refused.
# An archive of such a tree would not do what INSTALL.md says, or would carry libraries of an
# earlier generator under a new version. Prints PASS or FAIL for each case; exit status 1 on a FAIL.
set -uo pipefail
W=$(cd "$(dirname "$0")/../.." && pwd)
[ -f "$W/dist/forward/libcublasLt.so.13" ] && [ -f "$W/dist/lib/libcuda.so.1" ] || { echo "missing: a built dist/ (scripts/build.sh, scripts/build-forwarders.sh)"; exit 2; }
tmp=$(mktemp -d "${TMPDIR:-/tmp}/vf-package-refusals.XXXXXX") || exit 2
trap 'rm -rf "$tmp"' EXIT
failed=0
# refused NAME WANTED: the copy in $tmp/NAME must be refused (status 2) with a line that has WANTED.
refused() {
    local name=$1 wanted=$2 log=$tmp/$1.log status
    DIST=$tmp/$name ALLOW_DIRTY=1 bash "$W/scripts/package.sh" > "$log" 2>&1
    status=$?
    if [ $status = 2 ] && grep -q -- "$wanted" "$log" && ! ls "$tmp/$name"/volta-forever-*.tar.* > /dev/null 2>&1; then
        echo "PASS  $name"
    else
        echo "FAIL  $name (status $status, wanted a refusal with: $wanted)"; sed 's/^/    /' "$log" | tail -5; failed=1
    fi
}
copy() {  # copy NAME: the built tree as it is (dates kept)
    mkdir -p "$tmp/$1" && cp -a "$W/dist/lib" "$W/dist/bin" "$W/dist/forward" "$tmp/$1/"
}

copy missing-library
mv "$tmp/missing-library/forward/libcublasLt.so.13" "$tmp/missing-library/forward/libcublasLt.so.13.aside"
refused missing-library "lacks 1 of the .* forwarding libraries.*libcublasLt.so.13"

copy old-library
touch -d 2000-01-01 "$tmp/old-library/forward/libcufft.so.12"
refused old-library "libcufft.so.12 is older than"

copy partial-report
grep -v "^libcusolver.so.12 " "$W/dist/forward/REPORT.txt" > "$tmp/partial-report/forward/REPORT.txt"
touch -r "$W/dist/forward/REPORT.txt" "$tmp/partial-report/forward/REPORT.txt"
refused partial-report "lacks 1 of the .* forwarding libraries.*libcusolver.so.12"

exit $failed

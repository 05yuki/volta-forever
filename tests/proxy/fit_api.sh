#!/bin/bash
# The second translation at launch, by every way a program can come to it, without a GPU:
#   tests/proxy/fit_api.sh [PROXY_DIR]
# PROXY_DIR is the directory of the libcuda.so.1 to test (default: dist/lib of this tree). A
# stand-in for the driver (fit_mock_driver.c) refuses the launches of the first image loaded;
# fit_api.c says what each case checks. Needs gcc and tests/kernels/k_fit_test.135.sm75.cubin.
# Prints one line a case, then PASS or FAIL; exit status 1 on FAIL.
set -uo pipefail
W=$(cd "$(dirname "$0")/../.." && pwd)
proxy=${1:-$W/dist/lib}
G=$W/tests/proxy
out=$(mktemp -d "${TMPDIR:-/tmp}/vf-fit-api-test.XXXXXX") || exit 2
trap 'rm -rf "$out"' EXIT

gcc -shared -fPIC -O0 -Wl,-Bsymbolic -o "$out/mock.so" "$G/fit_mock_driver.c" || exit 2
gcc -O2 -Wall -o "$out/fit_api" "$G/fit_api.c" -ldl -lpthread || exit 2

cubin=$W/tests/kernels/k_fit_test.135.sm75.cubin
[ -f "$cubin" ] || { echo "missing: $cubin"; exit 2; }

# The stand-in as the driver; nothing kept, nothing said.
export VF_REAL_LIBCUDA="$out/mock.so"
export VF_PROXY_CACHE=0
export VF_LOG_LEVEL=0
export VF_EMULATION_WARNINGS=0
"$out/fit_api" "$proxy/libcuda.so.1" "$out/mock.so" "$cubin"
status=$?

if [ $status -eq 0 ]; then echo "PASS  fit_api"; else echo "FAIL  fit_api"; fi
exit $status

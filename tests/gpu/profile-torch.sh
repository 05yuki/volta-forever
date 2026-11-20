#!/bin/bash
# Profile PyTorch operations through the layer and natively on a V100.
# Usage: profile-torch.sh GPU [PATTERN...]
#
# Writes the profiler's files to $VF_OUT/logs/profile-layer.PID and profile-native.PID (one per
# process). The layer build is from the tree this script is in; the native wheel (sm_70 code)
# runs through the same proxy, which leaves its images as they are and times its kernels the
# same way.
#
# Fails (exit 1) when a run fails, when a profile file is missing or empty, when it lacks one of
# the expected columns, when no kernel has at least one launch and a time above zero, or when no
# kernel of the native wheel is labelled "native" in its origin column.
set -uo pipefail
W=$(cd "$(dirname "$0")/../.." && pwd)
. "$W/tests/gpu/env.sh"
out=${VF_OUT:-$VF_TEST_ROOT/profile}
layer_site=${VF_TORCH:-$VF_TEST_ROOT/torch-cu130/site}
native_site=${VF_TORCH_NATIVE:-$VF_TEST_ROOT/torch-cu126/site}
gpu=${1:-0}
[ $# -gt 0 ] && shift
for site in "$layer_site" "$native_site"; do
    [ -d "$site/torch" ] || { echo "missing: PyTorch in $site"; exit 2; }
done
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR=$out/tmp XDG_CACHE_HOME=$out/cache TORCH_HOME=$out/cache/torch CUDA_CACHE_PATH=$out/cache/cuda
mkdir -p "$out/logs" "$out/tmp" "$out/cache" "$out/cache/torch" "$out/cache/cuda" "$out/cache/proxy"

proxy=$W/dist/lib
map=$(vf_torch_map "$VF_TEST_ROOT")

mkdir -p "$out/logs"
rm -f "$out"/logs/profile-layer.[0-9]* "$out"/logs/profile-native.[0-9]*

echo "=== profiling: layer run ==="
# Run through the layer with profiling. The proxy cache must exist and be owned by root.
VF_PROXY_CACHE=$out/cache/proxy
[ -d "$VF_PROXY_CACHE" ] || mkdir -p "$VF_PROXY_CACHE"

if [ -f "$map" ]; then
    VF_PROFILE=$out/logs/profile-layer \
    LD_AUDIT=$proxy/libvf_audit.so VF_AUDIT_MAP_FILE=$map \
    LD_LIBRARY_PATH=$proxy VF_PROXY_CACHE=$VF_PROXY_CACHE VF_LOG=$out/logs/profile-layer-sm70.log VF_LOG_LEVEL=2 \
    PYTHONPATH=$layer_site python3 "$W/tests/gpu/torch-bench.py" "$out/logs/profile-layer-bench.tsv" "$@" \
        >"$out/logs/profile-layer.log" 2>&1
else
    VF_PROFILE=$out/logs/profile-layer \
    LD_LIBRARY_PATH=$proxy VF_PROXY_CACHE=$VF_PROXY_CACHE VF_LOG=$out/logs/profile-layer-sm70.log VF_LOG_LEVEL=2 \
    PYTHONPATH=$layer_site python3 "$W/tests/gpu/torch-bench.py" "$out/logs/profile-layer-bench.tsv" "$@" \
        >"$out/logs/profile-layer.log" 2>&1
fi
layer_exit=$?
echo "  layer exit: $layer_exit"
ls -la "$out/logs/profile-layer."* 2>/dev/null

echo ""
echo "=== profiling: native run ==="
# Run the native wheel through the proxy for profiling. The proxy still looks at the images (so
# that it can tell the ones that already have sm_70 code), finds nothing to adapt and loads them
# as they are; it intercepts the launches for timing.
VF_PROFILE=$out/logs/profile-native \
LD_LIBRARY_PATH=$proxy VF_LOG=$out/logs/profile-native-sm70.log VF_LOG_LEVEL=2 \
PYTHONPATH=$native_site python3 "$W/tests/gpu/torch-bench.py" "$out/logs/profile-native-bench.tsv" "$@" \
    >"$out/logs/profile-native.log" 2>&1
native_exit=$?
echo "  native exit: $native_exit"
ls -la "$out/logs/profile-native."* 2>/dev/null

echo ""
echo "=== checking the profiles ==="
fail=0
[ "$layer_exit" -eq 0 ] || { echo "FAIL: the layer run exited with $layer_exit ($out/logs/profile-layer.log)"; fail=1; }
[ "$native_exit" -eq 0 ] || { echo "FAIL: the native run exited with $native_exit ($out/logs/profile-native.log)"; fail=1; }

# check_profile WHAT PREFIX NEED_NATIVE: the files PREFIX.PID exist, are not empty, have the
# columns, and hold a kernel launched at least once with a time above zero; with NEED_NATIVE=1,
# one of those kernels is labelled native.
check_profile() {
    python3 - "$1" "$2" "$3" <<'PY'
import glob, sys
what, prefix, need_native = sys.argv[1], sys.argv[2], sys.argv[3] == '1'
columns = ['kernel', 'origin', 'launches', 'total_ms', 'mean_ms', 'grids', 'blocks', 'regs',
           'static_smem', 'dynamic_smem', 'blocks_per_sm', 'limited_by', 'by_regs', 'by_smem',
           'by_blocks', 'by_warps', 'emulated']
files = sorted(glob.glob(prefix + '.[0-9]*'))
def fail(why):
    print('FAIL: %s: %s' % (what, why))
    sys.exit(1)
if not files:
    fail('no profile file %s.PID' % prefix)
timed = native = 0
for path in files:
    lines = open(path).read().splitlines()
    if not lines:
        fail('%s is empty' % path)
    header = lines[0].split('\t')
    missing = [c for c in columns if c not in header]
    if missing:
        fail('%s lacks the columns %s' % (path, ', '.join(missing)))
    col = {c: header.index(c) for c in columns}
    for line in lines[1:]:
        f = line.split('\t')
        if len(f) <= max(col.values()):
            fail('%s: a line with %d fields: %r' % (path, len(f), line[:80]))
        try:
            launches, total = int(f[col['launches']]), float(f[col['total_ms']])
        except ValueError:
            fail('%s: a launch count or a time that is not a number: %r' % (path, line[:80]))
        if launches >= 1 and total > 0:
            timed += 1
            if f[col['origin']] == 'native':
                native += 1
if timed == 0:
    fail('no kernel with at least one launch and a time above zero')
if need_native and native == 0:
    fail('no kernel of the native wheel is labelled native')
print('ok: %s: %d files, %d kernels launched and timed%s' % (
    what, len(files), timed, ', %d native' % native if need_native else ''))
PY
}
check_profile "layer profile" "$out/logs/profile-layer" 0 || fail=1
check_profile "native profile" "$out/logs/profile-native" 1 || fail=1

echo ""
echo "=== done ==="
echo "Layer bench TSV: $out/logs/profile-layer-bench.tsv"
echo "Native bench TSV: $out/logs/profile-native-bench.tsv"
echo "Profile files: $out/logs/profile-layer.* $out/logs/profile-native.*"
[ "$fail" -eq 0 ] && echo "PASS" || echo "FAIL"
exit "$fail"

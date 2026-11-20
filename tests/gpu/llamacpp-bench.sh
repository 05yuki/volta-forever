#!/bin/bash
# Speed of llama.cpp on a machine with a V100, official CUDA 13.5 build through the proxy against
# a build of the same sources for sm_70 (see tests/README.md for how to make it):
#   tests/gpu/llamacpp-bench.sh [GPU [MODEL...]]
# Runs llama-bench (prompt processing of 512 and 2048 tokens, generation of 128, with and without flash
# attention, all layers on the GPU) and prints its tables. Only meaningful on an idle GPU. The first
# run of a model through the proxy compiles its kernels, which llama-bench does not count.
#
# GPU is the PCI-order index (default 0); MODEL defaults to the two test models. The native build is
# looked for in $VF_NATIVE_LLAMA (default $VF_TEST_ROOT/llama-b12418-sm70/build/bin) and
# skipped if absent. Options for llama-bench: $VF_BENCH (default: -p 512,2048 -n 128 -r 3).
set -uo pipefail
W=$(cd "$(dirname "$0")/../.." && pwd)
. "$W/tests/gpu/env.sh"
D=$W/demos/llamacpp/b12418
B=$D/llama/llama-b12418
T=$W/toolkits/cuda-12.9.2/lib
out=${VF_OUT:-$VF_TEST_ROOT}
native=${VF_NATIVE_LLAMA:-$VF_TEST_ROOT/llama-b12418-sm70/build/bin}
gpu=${1:-0}
[ $# -gt 0 ] && shift
[ $# -gt 0 ] || set -- "$W/demos/models/qwen2.5-0.5b-instruct-q4_0.gguf" "$W/demos/models/qwen2.5-0.5b-instruct-q8_0.gguf"
read -r -a options <<<"${VF_BENCH:--p 512,2048 -n 128 -r 3}"
mkdir -p "$out/logs"
[ -e "$W/dist/forward/libcublas.so.13" ] || { echo "missing: $W/dist/forward (run scripts/build-forwarders.sh)"; exit 2; }
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu
. "$W/tests/gpu/cache.sh"
export CUDA_CACHE_PATH=${VF_JIT_CACHE:-$VF_TEST_ROOT/jit-cache} CUDA_CACHE_MAXSIZE=4294967296
busy=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader -i "$gpu")
echo "GPU $gpu before the runs (utilisation, memory used): $busy"
for model in "$@"; do
    name=$(basename "$model" .gguf)
    echo "== $name, official CUDA 13.5 build through the proxy"
    LD_LIBRARY_PATH=${VF_PROXY:-$W/dist/lib}:$W/dist/forward:$D/cudart/cudart-llama-b12418-bin-ubuntu-cuda-13.5-x64:$T \
        "$B/llama-bench" -m "$model" -ngl 99 -fa 0,1 "${options[@]}" -o md 2>"$out/logs/bench-proxy-$name.err" | grep -E "^\|"
    if [ -x "$native/llama-bench" ]; then
        echo "== $name, build for sm_70"
        LD_LIBRARY_PATH=$native:$T "$native/llama-bench" -m "$model" -ngl 99 -fa 0,1 "${options[@]}" -o md \
            2>"$out/logs/bench-native-$name.err" | grep -E "^\|"
    fi
done

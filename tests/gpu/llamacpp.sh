#!/bin/bash
# The llama.cpp demonstrator, on a machine with a V100:
#   tests/gpu/llamacpp.sh [GPU [LAYERS [MODEL]]]
# Runs the official llama.cpp build for CUDA 13.5 (demos/llamacpp, fetched by scripts/fetch-all.sh demos)
# through the proxy, with and without flash attention, and compares it with the same build on the CPU:
#
# - logits: the next-token distributions over a few thousand tokens of text, processed 512 at a time
#   (the batched kernels: tensor-core flash attention, quantized matrix products). The mean KL
#   divergence from the CPU's distributions and the share of positions with the same most likely token
#   must be what two correct implementations show: on the text that this test had until 2026-11-20,
#   the native sm_70 build of the same llama.cpp gave 0.0016 and 96.7 %, the layer 0.0018 and 96.9 %.
# - text: 48 tokens generated from a short prompt, token by token. Reported, not judged: about 3 % of
#   the most likely tokens differ between any two implementations, so the texts can diverge. The
#   tokens per second printed are not the speed of generation on a first run after a build: the
#   modules first used while generating are adapted and analysed then (about a second), which
#   is most of the time of 48 tokens. Measured over 48 and 304 tokens: 2.1 to 2.4 ms per token in
#   every mode once that is done.
#
# The times printed include the compilation of the kernels on a first run; tests/gpu/llamacpp-bench.sh
# measures speed.
#
# GPU is the PCI-order index (default 0), LAYERS the number of layers put on the GPU (default 99: all),
# MODEL a GGUF file (default: a 0.5B model that fits beside a job that fills the GPU). The first run
# compiles the kernels (minutes); the compiled code is kept in $VF_JIT_CACHE (default
# $VF_TEST_ROOT/jit-cache). The tree tested is the one this script is in; logs go under $VF_OUT.
#
# Library path: the proxy, then the libraries of CUDA 12.9.2 behind the names of CUDA 13 (dist/forward, from scripts/build-forwarders.sh), then the
# CUDA 13.5 libraries distributed with llama.cpp, then what cuBLAS 12.9.2 itself needs.
set -uo pipefail
W=$(cd "$(dirname "$0")/../.." && pwd)
. "$W/tests/gpu/env.sh"
D=$W/demos/llamacpp/b12418
B=$D/llama/llama-b12418
T=$W/toolkits/cuda-12.9.2/lib
out=${VF_OUT:-$VF_TEST_ROOT}
gpu=${1:-0} layers=${2:-99}
model=${3:-$W/demos/models/qwen2.5-0.5b-instruct-q4_0.gguf}
max_divergence=0.005 min_same_top=95
jit_cache=${VF_JIT_CACHE:-$VF_TEST_ROOT/jit-cache}
mkdir -p "$out/logs" "$jit_cache"
[ -e "$W/dist/forward/libcublas.so.13" ] || { echo "missing: $W/dist/forward (run scripts/build-forwarders.sh)"; exit 2; }
export LD_LIBRARY_PATH=${VF_PROXY:-$W/dist/lib}:$W/dist/forward:$D/cudart/cudart-llama-b12418-bin-ubuntu-cuda-13.5-x64:$T
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu
. "$W/tests/gpu/cache.sh"
export CUDA_CACHE_PATH=$jit_cache CUDA_CACHE_MAXSIZE=4294967296
export VF_LOG=$out/logs/llamacpp-proxy.log VF_LOG_LEVEL=2
: >"$VF_LOG"
log=$out/logs/llamacpp
failed=0

# ---- logits ---------------------------------------------------------------------------------------------
# A fixed text (two documents of this repository as they were on 2026-11-20), so that the figures of
# one day can be compared with those of another.
cp "$W/tests/gpu/llamacpp-text.txt" "$log-text.txt"
logits() {  # logits NAME OPTION... : next-token distributions of the text, or their comparison with the CPU's
    local name=$1; shift
    "$B/llama-perplexity" -m "$model" -f "$log-text.txt" -c 512 -b 512 --chunks 8 -t 16 \
        --kl-divergence-base "$log-cpu.logits" "$@" >"$log-logits-$name.out" 2>"$log-logits-$name.err"
}
if logits cpu --device none; then
    for fa in on off; do
        name=logits-fa-$fa
        if ! logits fa-$fa -ngl "$layers" -fa $fa --kl-divergence; then
            echo "FAIL  $name: llama-perplexity failed ($log-$name.err)"; failed=1; continue
        fi
        divergence=$(sed -n 's/^Mean  *KLD: *\([0-9.]*\).*/\1/p' "$log-$name.out")
        same_top=$(sed -n 's/^Same top p: *\([0-9.]*\).*/\1/p' "$log-$name.out")
        pass=$(sed -n 's/.*kl_divergence: *\([0-9.]*\) seconds per pass.*/\1/p' "$log-$name.err")
        if awk -v d="$divergence" -v s="$same_top" -v dmax=$max_divergence -v smin=$min_same_top \
            'BEGIN { exit !(d != "" && s != "" && d <= dmax && s >= smin) }'; then verdict=PASS; else verdict=FAIL; failed=1; fi
        echo "$verdict  $name: mean KL divergence from the CPU $divergence (limit $max_divergence), same most likely token $same_top % (at least $min_same_top), first 512 tokens in $pass s"
    done
else
    echo "FAIL  logits: the CPU run failed ($log-logits-cpu.err)"; failed=1
fi
rm -f "$log-cpu.logits"

# ---- text -----------------------------------------------------------------------------------------------
generate() {  # generate NAME OPTION... : 48 tokens after a short prompt
    local name=$1; shift
    "$B/llama-completion" -m "$model" -p "The three laws of robotics are" -n 48 -c 2048 --temp 0 -s 1 -t 16 -no-cnv "$@" \
        >"$log-text-$name.out" 2>"$log-text-$name.err"
}
if generate cpu --device none; then
    for fa in on off; do
        name=text-fa-$fa
        if ! generate fa-$fa -ngl "$layers" -fa $fa; then
            echo "FAIL  $name: llama-completion failed ($log-$name.err)"; failed=1; continue
        fi
        speed=$(sed -n 's/.* eval time.*runs.*, *\([0-9.]*\) tokens per second.*/\1/p' "$log-$name.err" | tail -1)
        if cmp -s "$log-$name.out" "$log-text-cpu.out"; then same="the same text as on the CPU"; else same="a text that differs from the CPU's (diff $log-$name.out $log-text-cpu.out)"; fi
        echo "PASS  $name: generated $same, $speed tokens per second"
    done
else
    echo "FAIL  text: the CPU run failed ($log-text-cpu.err)"; failed=1
fi
exit $failed

#!/bin/bash
# NVFP4 weights with stock llama.cpp on a machine with a V100:
#   tests/gpu/llamacpp-nvfp4.sh [GPU [MODEL]]
#
# Runs the official llama.cpp build for CUDA 13.5 (demos/llamacpp) through the proxy with an
# NVFP4 GGUF model and compares it with the same build on the CPU:
# - logits: next-token distributions over a text, compared with the CPU's (mean KL divergence,
#   share of same most likely token).
# - text: a few tokens generated from a short prompt, token by token.
# - speed: tokens per second for prompt processing and generation (from llama-bench).
#
# The NVFP4 path on a V100: activations are quantized to Q8_1 and the products are integer ones on
# the doubled-value table of E2M1. For one token at a time the kernel uses dp4a, which Volta has.
# For a batch (a prompt) the stock build takes its integer tensor-core kernels, which the layer
# emulates; a native sm_70 build leaves batches of 64 tokens or more to cuBLAS instead
# (ggml_cuda_should_use_mmq). Measured on Qwen3.8-27B: generation at the native build's speed,
# prompts 1.3 times faster than with it (`docs/performance.md`, "NVFP4 weights").
#
# MODEL is the path to an NVFP4 GGUF model. If not given, the script looks for one in
# $VF_TEST_ROOT/models. If none is found, it exits with a message.
#
# GPU is the PCI-order index (default 0).
set -uo pipefail
W=$(cd "$(dirname "$0")/../.." && pwd)
. "$W/tests/gpu/env.sh"
D=$W/demos/llamacpp/b12418
B=$D/llama/llama-b12418
T=$W/toolkits/cuda-12.9.2/lib
out=${VF_OUT:-$VF_TEST_ROOT}
native=${VF_NATIVE_LLAMA:-$VF_TEST_ROOT/llama-b12418-sm70/build/bin}
gpu=${1:-0}
model=${2:-}

# Find the NVFP4 model
if [ -z "$model" ]; then
    candidate=${VF_NVFP4_MODEL_CANDIDATE:-$VF_TEST_ROOT/models/nvfp4.gguf}
    [ -f "$candidate" ] && model=$candidate
fi
[ -n "$model" ] || { echo "No NVFP4 model found. Give an NVFP4 GGUF model as the second argument."; exit 2; }
echo "NVFP4 model: $model ($(du -sh "$model" | cut -f1))"

jit_cache=${VF_JIT_CACHE:-$VF_TEST_ROOT/jit-cache}
mkdir -p "$out/logs" "$jit_cache"
[ -e "$W/dist/forward/libcublas.so.13" ] || { echo "missing: $W/dist/forward (run scripts/build-forwarders.sh)"; exit 2; }
export LD_LIBRARY_PATH=${VF_PROXY:-$W/dist/lib}:$W/dist/forward:$D/cudart/cudart-llama-b12418-bin-ubuntu-cuda-13.5-x64:$T
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu
. "$W/tests/gpu/cache.sh"
export CUDA_CACHE_PATH=$jit_cache CUDA_CACHE_MAXSIZE=4294967296
export VF_LOG=$out/logs/llamacpp-nvfp4-proxy.log VF_LOG_LEVEL=2
: >"$VF_LOG"
log=$out/logs/llamacpp-nvfp4
failed=0

# --- text ---
generate() {
    local name=$1; shift
    "$B/llama-completion" -m "$model" -p "The three laws of robotics are" -n 48 -c 2048 --temp 0 -s 1 -t 16 -no-cnv --load-mode none "$@" \
        >"$log-text-$name.out" 2>"$log-text-$name.err"
}

echo "--- text generation ---"
if generate cpu --device none; then
    for fa in on off; do
        name=text-fa-$fa
        if ! generate fa-$fa -ngl 99 -fa $fa; then
            echo "FAIL  $name: llama-completion failed (see $log-$name.err)"; failed=1; continue
        fi
        speed=$(sed -n 's/.* eval time.*runs.*, *\([0-9.]*\) tokens per second.*/\1/p' "$log-$name.err" | tail -1)
        if cmp -s "$log-$name.out" "$log-text-cpu.out"; then same="the same text as on the CPU"; else same="a text that differs from the CPU's (diff $log-$name.out $log-text-cpu.out)"; fi
        echo "PASS  $name: generated $same, $speed tokens per second (48 tokens, the first ones slower)"
    done
else
    echo "FAIL  text: CPU run failed ($log-text-cpu.err)"; failed=1
fi

# --- speed ---
echo "--- speed (llama-bench) ---"
bench_opts="-p 512 -n 128 -r 3"
echo "== NVFP4 through the proxy"
LD_LIBRARY_PATH=${VF_PROXY:-$W/dist/lib}:$W/dist/forward:$D/cudart/cudart-llama-b12418-bin-ubuntu-cuda-13.5-x64:$T \
    "$B/llama-bench" -m "$model" -ngl 99 -fa 0,1 $bench_opts -o md --load-mode none 2>"$log-bench-proxy.err" | grep -E "^\|"

if [ -x "$native/llama-bench" ]; then
    echo "== NVFP4 native sm_70 build"
    LD_LIBRARY_PATH=$native:$T "$native/llama-bench" -m "$model" -ngl 99 -fa 0,1 $bench_opts -o md --load-mode none \
        2>"$log-bench-native.err" | grep -E "^\|"
fi

# --- logits (correctness) ---
echo "--- logits (correctness against CPU) ---"
max_divergence=0.01 min_same_top=90
if [ -f "$W/tests/gpu/llamacpp-text.txt" ]; then
    cp "$W/tests/gpu/llamacpp-text.txt" "$log-text.txt"
    logits() {
        local name=$1; shift
        "$B/llama-perplexity" -m "$model" -f "$log-text.txt" -c 512 -b 512 --chunks 4 -t 16 \
            --kl-divergence-base "$log-cpu.logits" --load-mode none "$@" >"$log-logits-$name.out" 2>"$log-logits-$name.err"
    }
    if logits cpu --device none; then
        for fa in on off; do
            name=logits-fa-$fa
            if ! logits fa-$fa -ngl 99 -fa $fa --kl-divergence; then
                echo "FAIL  $name: llama-perplexity failed ($log-$name.err)"; failed=1; continue
            fi
            divergence=$(sed -n 's/^Mean  *KLD: *\([0-9.]*\).*/\1/p' "$log-$name.out")
            same_top=$(sed -n 's/^Same top p: *\([0-9.]*\).*/\1/p' "$log-$name.out")
            if awk -v d="$divergence" -v s="$same_top" -v dmax=$max_divergence -v smin=$min_same_top \
                'BEGIN { exit !(d != "" && s != "" && d <= dmax && s >= smin) }'; then verdict=PASS; else verdict=FAIL; failed=1; fi
            echo "$verdict  $name: mean KL divergence from the CPU $divergence (limit $max_divergence), same most likely token $same_top % (at least $min_same_top)"
        done
    else
        echo "FAIL  logits: CPU run failed"; failed=1
    fi
    rm -f "$log-cpu.logits"
else
    echo "SKIP  logits: no test text file"
fi

exit $failed

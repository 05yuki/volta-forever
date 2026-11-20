#!/bin/bash
# Upstream ZML's LLM example on a machine with a V100. No file of the build is changed and no XLA
# flag is set: the audit library (LD_AUDIT) gives the program the proxy, the nvJitLink adapter, and
# NVIDIA's libraries that still have code for Volta in place of its own.
#   tests/gpu/zml.sh [GPU [CAPABILITY...]]
# CAPABILITY is what the program is told about the GPU. "7.5": the answer every caller without Volta
# code gets by default; XLA then leaves Triton off and sends matrix products to cuBLAS. "8.0": Triton
# on; its kernels use Ampere instructions, which the layer rewrites. Default: both, one after the other.
#
# Needs: the build of tests/gpu/zml/build-zml.sh ($VF_ZML, default
# $VF_TEST_ROOT/zml/build), the model ($VF_ZML_MODEL), toolkits/ (CUDA 12.9.2, from
# scripts/fetch-all.sh) and dist/ (scripts/build.sh).
#
# What stands in for what (written to $out/zml/vf-audit.map):
#   libcuda.so.1                 the proxy
#   libnvJitLink.so.13           the adapter, served by nvJitLink 12.9.2
#   cuBLAS, cuBLASLt, cuFFT, cuSOLVER, NVRTC of CUDA 13   the forwarders of dist/forward (scripts/build-forwarders.sh),
#                                which call the libraries of 12.9.2
# cuDNN is the build's own (9.24.0): its kernels come as PTX for a later GPU, which the proxy adapts.
# VF_ZML_CUDNN=9.10.2 puts cuDNN 9.10.2 in its place instead, the last with machine code for Volta
# (toolkits/cudnn-9.10.2-cuda12): it has no libcudnn_engines_tensor_ir, which the program only lists (an
# empty library stands for it), and its version queries answer 9.24.0, because XLA refuses a cuDNN
# older than the one it was built with. Same speed; kept for comparison.
set -uo pipefail
W=$(cd "$(dirname "$0")/../.." && pwd)
. "$W/tests/gpu/env.sh"
T=$W/toolkits/cuda-12.9.2/lib
C=$W/toolkits/cudnn-9.10.2-cuda12/lib
out=${VF_OUT:-$VF_TEST_ROOT}
# (The build to test: the tree's own, or the directory $VF_PROXY, a build staged beside it.)
proxy=${VF_PROXY:-$W/dist/lib}
zml=${VF_ZML:-$VF_TEST_ROOT/zml/build}
model=${VF_ZML_MODEL:-$VF_TEST_ROOT/models/Qwen3.5-9B-AWQ-fp16-text}
[ -d "$model" ] || [ -f "$model" ] || { echo "missing: \$VF_ZML_MODEL ($model); set it to the path of the model"; exit 2; }
cudnn=${VF_ZML_CUDNN:-own}
cudnn_version=${VF_ZML_CUDNN_VERSION:-9.24.0}
sandbox=$zml/llm.runfiles/+cuda_packages+libzml_cuda_linux_amd64/sandbox/lib
gpu=${1:-0}
[ $# -gt 0 ] && shift
[ $# -gt 0 ] || set -- 7.5 8.0
[ "$cudnn" = own ] || [ -e "$C/libcudnn.so.9" ] || { echo "missing: $C/libcudnn.so.9"; exit 2; }
for needed in "$W/dist/forward/libcublas.so.13" "$zml/llm" "$sandbox/libzml_cuda.so" "$model" "$T/libnvJitLink.so.12" "$proxy/libvf_audit.so"; do
    [ -e "$needed" ] || { echo "missing: $needed"; exit 2; }
done
mkdir -p "$out/logs" "$out/zml"

map=$out/zml/vf-audit.map
{
    echo "# Written by tests/gpu/zml.sh."
    echo "libcuda.so.1=$proxy/libcuda.so.1"
    echo "libnvJitLink.so.13=$proxy/libnvJitLink.so.13"
    echo "libnvJitLink.so.12=$T/libnvJitLink.so.12"
    for pair in libcublas.so.12:libcublas.so.13 libcublasLt.so.12:libcublasLt.so.13 libcufft.so.11:libcufft.so.12 \
                libcusolver.so.11:libcusolver.so.12 libnvrtc.so.12:libnvrtc.so.13; do
        name=${pair##*:}
        echo "$name=$W/dist/forward/$name"
    done
    if [ "$cudnn" != own ]; then
        for library in "$sandbox"/libcudnn*.so.9; do
            name=$(basename "$library")
            if [ -e "$C/$name" ]; then
                echo "$name=$C/$name"
            else
                [ -e "$out/zml/$name" ] || gcc -shared -fPIC -Wl,-soname,"$name" -o "$out/zml/$name" "$W/tests/gpu/zml/empty.c" || exit 2
                echo "$name=$out/zml/$name"
            fi
        done
        echo "@cudnn-version=$cudnn_version"
    fi
} >"$map" || exit 2

export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu
. "$W/tests/gpu/cache.sh"
export CUDA_CACHE_PATH=${VF_JIT_CACHE:-$VF_TEST_ROOT/jit-cache} CUDA_CACHE_MAXSIZE=4294967296
unset XLA_FLAGS
echo "GPU $gpu before the runs (utilisation, memory used): $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader -i "$gpu")"
failed=0
for capability in "$@"; do
    log=$out/logs/zml-$capability
    [ "$cudnn" = own ] || log=$log-cudnn-$cudnn
    rm -f "$log.vf.log"
    (
        # 7.5 is what the proxy answers by itself; another value has to be asked for.
        [ "$capability" = 7.5 ] || export VF_CAPABILITY=$capability
        export VF_LOG=$log.vf.log VF_LOG_LEVEL=${VF_LOG_LEVEL:-1} VF_AUDIT_MAP_FILE=$map
        export LD_AUDIT=$proxy/libvf_audit.so LD_LIBRARY_PATH=$sandbox:$T
        script -qfc "$zml/llm --model=$model --seqlen=2048 --prompt='The capital of France is'" "$log.log" >/dev/null 2>&1
    )
    text=$(sed 's/\x1b\[[0-9;]*m//g' "$log.log" | tr -d '\r')
    told=$(printf '%s\n' "$text" | grep -o 'compute_capability: "[0-9.]*"' | head -1)
    speed=$(printf '%s\n' "$text" | grep -o 'decode .*tok/s' | tail -1)
    if printf '%s\n' "$text" | grep -q "Paris" && [ -n "$speed" ] && [ "$told" = "compute_capability: \"$capability\"" ]; then
        echo "PASS  $(basename "$log"): told $capability, answered with Paris, $speed"
    else
        echo "FAIL  $(basename "$log") (told: ${told:-nothing}; $log.log):"
        printf '%s\n' "$text" | grep -v '^ *$' | tail -4 | cut -c1-300
        failed=1
    fi
done
exit $failed

# Sourced by the test scripts, after CUDA_VISIBLE_DEVICES is set.
#
# What the proxy keeps of the modules a program loads, so that the next start neither adapts nor
# translates them again (VF_PROXY_CACHE), goes by default to the home directory of whoever runs:
# on the nodes that is memory, and up to 2 GB of it. It cannot go to the NFS either: the proxy
# refuses a cache whose files do not belong to the user it runs as, and root's files there do not.
# So the tests keep it in a directory under /tmp for this script and this GPU, emptied when the
# script starts and removed when it ends (a script must not `exec` its last command, or the trap
# never runs). $VF_PROXY_CACHE, when it is set, is left as it is: a directory, or 0 for none.
if [ -z "${VF_PROXY_CACHE:-}" ]; then
    VF_PROXY_CACHE=/tmp/vf-cache-$(basename "$0" .sh)-gpu${CUDA_VISIBLE_DEVICES:-x}
    rm -rf "$VF_PROXY_CACHE"
    trap 'rm -rf "$VF_PROXY_CACHE"' EXIT
    export VF_PROXY_CACHE
fi

# Two more things a program writes under the home directory of whoever runs it, which on the nodes
# is memory: the kernels the driver compiles from PTX (~/.nv/ComputeCache: 212 MB were found there
# on 2026-11-19, from tests that did not say otherwise) and the kernels PyTorch compiles by itself
# (~/.cache/torch/kernels). With the other outputs instead, unless the test or the caller has said
# where. (Not llama.cpp's directory of compiled kernels: a test with the driver's default limit of
# 1 GB would empty what took minutes to compile.)
_vf_out=${VF_OUT:-$VF_TEST_ROOT}
if [ -d "$_vf_out" ]; then
    export CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-$_vf_out/nv-cache}
    export PYTORCH_KERNEL_CACHE_PATH=${PYTORCH_KERNEL_CACHE_PATH:-$_vf_out/torch-kernels}
fi
unset _vf_out

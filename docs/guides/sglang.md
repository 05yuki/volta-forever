# SGLang on a Tesla V100

## 1. CUDA

On Ubuntu 24.04 x86-64, run this check for the proprietary NVIDIA R580 driver.

```bash
if ! nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | grep -E 'V100.*, 580[.]'; then
  sudo apt update && sudo apt install -y nvidia-driver-580
fi
```

If you installed the driver, reboot and repeat the check. It should print your V100 and driver version.

SGLang compiles FlashInfer kernels on first launch. Use a complete CUDA 13 toolkit so the compiler and headers come from the same installation. This block installs the toolkit only when CUDA 13 is missing. [NVIDIA installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#ubuntu)

```bash
export PATH="$PATH:${CUDA_HOME:-/usr/local/cuda}/bin"
if ! nvcc --version 2>/dev/null | grep -q 'release 13[.]'; then
  sudo apt update && sudo apt install -y ca-certificates curl
  repo=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64
  curl -fsSL "$repo/cuda-archive-keyring.gpg" | sudo tee /usr/share/keyrings/cuda-archive-keyring.gpg >/dev/null
  echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] $repo/ /" | sudo tee /etc/apt/sources.list.d/cuda-ubuntu2404-x86_64.list
  sudo apt update && sudo apt install -y cuda-toolkit-13
  export PATH="/usr/local/cuda-13/bin:$PATH"
fi
nvcc_path=$(readlink -f "$(command -v nvcc)")
export CUDA_HOME="${nvcc_path%/bin/nvcc}"
```

## 2. VF

The installer fetches the CUDA 12.9.2 cuBLAS libraries used by `vf`. If an existing installation lacks them, run `vf fetch`.

```sh
sudo apt update
sudo apt install -y ca-certificates curl python3 python3-venv xz-utils
curl -fsSLo /tmp/vf-install.sh https://raw.githubusercontent.com/nbald/volta-forever/main/install.sh
bash /tmp/vf-install.sh
export PATH="$HOME/.local/bin:$PATH"
```

## 3. Engine

```sh
sudo apt install -y build-essential git libgl1 libglib2.0-0t64 libibverbs1 libnuma-dev \
  librdmacm1t64 ninja-build python3-dev
mkdir -p ~/sglang-v100 && cd ~/sglang-v100
vf venv .venv
source .venv/bin/activate
pip install sglang
pip install "git+https://github.com/nbald/volta-forever.git#subdirectory=docs/guides/sglang-v100"
```

## 4. Test

Keep the environment active and use the full CUDA toolkit from step 1. The [V100 plugin](./sglang-v100/) selects PyTorch math attention for prefill, SGLang’s JIT expert router and its PyTorch expert reduction. Triton handles decode inside a CUDA graph.

```sh
export PATH="$CUDA_HOME/bin:$PATH"
export VF_CAPABILITY=8.0 SGLANG_MAMBA_CONV_DTYPE=float16 FLASHINFER_USE_CUDA_NORM=1
export TRITON_OVERRIDE_ARCH=sm75 TORCH_COMPILE_DISABLE=1
export NVCC_APPEND_FLAGS='-gencode=arch=compute_80,code=compute_80'
export VF_TRITON_PTXAS="$(python -c 'from pathlib import Path; import triton; print(Path(triton.__file__).parent / "backends/nvidia/bin/ptxas")')"
curl -fsSLo triton-ptxas.py https://raw.githubusercontent.com/nbald/volta-forever/main/scripts/triton-ptxas.py
chmod +x triton-ptxas.py
export TRITON_PTXAS_PATH="$PWD/triton-ptxas.py"
export SGLANG_ENABLE_METRICS_DEVICE_TIMER=1
python -m sglang.launch_server --model-path QuantTrio/Qwen3.5-35B-A3B-AWQ --served-model-name qwen \
  --dtype float16 --attention-backend triton --prefill-attention-backend torch_native --decode-attention-backend triton \
  --sampling-backend pytorch \
  --context-length 512 --max-running-requests 1 --chunked-prefill-size 256 --mem-fraction-static 0.85 \
  --weight-loader-disable-mmap --model-loader-extra-config '{"enable_multithread_load":false}' \
  --cuda-graph-backend-decode full --cuda-graph-bs-decode 1 --cuda-graph-backend-prefill disabled \
  --skip-server-warmup --watchdog-timeout 3600 --enable-metrics \
  --host 127.0.0.1 --port 8000
```

The server downloads the checkpoint on its first launch. After `curl -fsS http://127.0.0.1:8000/ready` succeeds, send a chat from a second terminal. The first request compiles the expert kernels. The tested server returned **Paris** in two output tokens. [Qwen3.5-35B-A3B-AWQ](https://huggingface.co/QuantTrio/Qwen3.5-35B-A3B-AWQ)

```sh
curl -fsS http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen","messages":[{"role":"user","content":"What is the capital of France? Answer with just the city."}],"temperature":0,"max_tokens":32,"chat_template_kwargs":{"enable_thinking":false}}'
```

## 5. Benchmarks

With the server running, use the [SGLang benchmark instructions](./benchmarks/#sglang). After one warm-up, three distinct chats each processed 227 input tokens and generated 64 tokens, with temperature zero, thinking disabled and one request at a time. Each chat used its own `cache_salt`. The server counted zero reused tokens.

| Phase | Three runs (tokens/s) | Median (tokens/s) |
| --- | --- | ---: |
| Prefill | 167.72, 643.91, 644.98 | **643.91** |
| Decode | 67.33, 67.29, 67.30 | **67.30** |

[Measured responses and server counters](./benchmarks/sglang-results.json)

The rates use SGLang’s GPU forward timers for `extend` and `decode`. Prefill counts all 227 input tokens, and decode counts the 63 tokens after the first output. Median time to first token over loopback was 0.374 seconds. The test used a Tesla V100-SXM2-32GB, driver 580.178.04, VF 0.1.1 with the Triton helper above, CUDA 13.5, SGLang 0.5.23 and PyTorch 2.14.1+cu130.

## 6. Additional notes

The 8.0 capability profile selects SGLang’s 4-bit AWQ Marlin experts. The CUDA and Triton flags preserve PTX for VF. The larger-batch router in the tested wheel ignored CUDA error 209 on Volta. The plugin selects the JIT implementation shipped by SGLang. The server captured its batch-one decode graph using 0.60 GiB. An eager diagnostic with eight-token prefill chunks measured 71.14 tokens/s for prefill and 12.18 tokens/s for decode.

Qwen2.5-0.5B-Instruct also answered **Paris** with float16 weights and FlashInfer attention. Three warm 32-token completions had a median end-to-end throughput of 160.8 tokens/s on the same type of V100.

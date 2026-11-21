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
```

## 4. Test

Use the same shell so `CUDA_HOME` points to the toolkit from step 1. On first launch the server downloads Qwen2.5-0.5B-Instruct and compiles kernels.

```sh
export PATH="$CUDA_HOME/bin:$PATH"
python -m sglang.launch_server --model-path Qwen/Qwen2.5-0.5B-Instruct \
  --dtype float16 --attention-backend flashinfer --mem-fraction-static 0.7 \
  --host 127.0.0.1 --port 8000
```

From a second terminal on the same machine, send the request. The tested server returned `Paris` in `choices[0].message.content`.

```sh
curl -fsS http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","messages":[{"role":"user","content":"What is the capital of France? Answer with just the city."}],"temperature":0,"max_tokens":32}'
```

## 5. Benchmarks

After startup, send one warm-up request and time three more on loopback. The prompt is `The capital of France is`, temperature is zero and the completion limit is 32 tokens.

```sh
body='{"model":"Qwen/Qwen2.5-0.5B-Instruct","prompt":"The capital of France is","temperature":0,"max_tokens":32}'
for i in 0 1 2 3; do
  curl -fsS -o /dev/null -w "%{time_total} s\n" http://127.0.0.1:8000/v1/completions \
    -H 'Content-Type: application/json' -d "$body"
done
```

On a Tesla V100-SXM2-32GB with driver 580.178.04, SGLang 0.5.23 and PyTorch 2.14.1+cu130 returned 32 tokens in 0.199, 0.199 and 0.198 seconds. Median end-to-end throughput was **160.8 tokens/s** for one request at a time with float16 weights.

## 6. Additional notes

The small-model benchmark used VF 0.1.0 and the complete CUDA 13.5 toolkit. VF 0.1.1 also reports capability 7.5 through NVML. SGLang uses PyTorch's capability for AWQ. At 7.5 it leaves the experts of `QuantTrio/Qwen3.5-35B-A3B-AWQ` in float16, which requires 60 GiB for the experts alone. At 8.0 it selects AWQ Marlin. A two-block diagnostic with synthetic weights passed Marlin repacking with Ampere PTX, then stopped on an sm_80-only Triton kernel. Generation from the full checkpoint is still unverified.

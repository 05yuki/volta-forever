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

The following command checks the stock SGLang installation with a small model on one V100. Use the same shell so `CUDA_HOME` points to the toolkit from step 1. On first launch the server downloads Qwen2.5-0.5B-Instruct and compiles kernels.

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

An experimental Qwen3.5-35B-A3B AWQ run on one V100 measured **643.91 tokens/s in prefill** and **67.30 tokens/s in decode**. It used a separate SGLang adapter and preserved Triton's PTX. These measurements apply to that earlier configuration. Each request processed 227 uncached tokens, generated 64 tokens and returned the requested city. Decode used a batch-one CUDA graph. [Results and server counters](benchmarks/sglang-results.json) · [Archived configuration](https://github.com/nbald/volta-forever/blob/7ead7334d0909483b63139d253aead22cf97820a/docs/guides/sglang.md)

## 6. Additional notes

The stock 35B path is under validation. The tested SGLang 0.5.23 wheel ships the expert router and reduction as sm80-or-later machine code without PTX. In the alternative WNA16 path, the loader selects a linear-layer weight method for an attention module. Results for the standard installation above cover Qwen2.5-0.5B-Instruct.

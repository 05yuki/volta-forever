# FreeToken on Ubuntu with a Tesla V100

## 1. CUDA

On Ubuntu 24.04 x86-64, run this check for the proprietary NVIDIA R580 driver.

```bash
if ! nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | grep -E 'V100.*, 580[.]'; then
  sudo apt update && sudo apt install -y nvidia-driver-580
fi
```

If you installed the driver, reboot and repeat the check. It should print your V100 and driver version.

FreeToken needs a CUDA 13 compiler for its kernels. If `nvcc --version` reports CUDA 13, use that toolkit. Step 3 provides the compiler when it is missing. [FreeToken requirements](https://github.com/FlashML-org/FreeToken/blob/main/docs/install.md)

## 2. VF

Run the installer as your usual user. It fetches the CUDA 12.9.2 libraries used by volta-forever.

```sh
sudo apt-get update
sudo apt-get install -y curl python3-venv xz-utils
curl -fsSL https://raw.githubusercontent.com/nbald/volta-forever/main/install.sh | bash
export PATH="$HOME/.local/bin:$PATH"
vf version
```

## 3. Engine

Create the environment and install FreeToken. Its console command and Python workers will inherit VF.

```sh
sudo apt-get install -y python3-dev build-essential ninja-build
mkdir -p ~/freetoken-v100 && cd ~/freetoken-v100
vf venv .venv
source .venv/bin/activate
pip install freetoken
ft --version
```

Run this in the active environment. It installs the compiler and headers only if a CUDA 13 toolkit is missing. [NVIDIA compiler](https://pypi.org/project/nvidia-cuda-nvcc/) · [NVIDIA C++ headers](https://pypi.org/project/nvidia-cuda-cccl/)

```bash
export PATH="$PATH:${CUDA_HOME:-/usr/local/cuda}/bin"
if ! nvcc --version 2>/dev/null | grep -q 'release 13[.]'; then
  pip install nvidia-cuda-nvcc nvidia-cuda-cccl
  export CUDA_HOME="$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')/nvidia/cu13"
  export PATH="$CUDA_HOME/bin:$PATH"
  ln -sfn lib "$CUDA_HOME/lib64"
  ln -sfn libcudart.so.13 "$CUDA_HOME/lib/libcudart.so"
fi
```

## 4. Test

Download the Q4_K_M text model (20.5 GiB) to a local drive. The following configuration keeps the experts on the 32 GiB V100 and reserves 1,024 tokens for the short test. [Qwen3.5-35B-A3B GGUF](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF) · [Supported models](https://github.com/FlashML-org/FreeToken/blob/main/docs/models.md)

```sh
model=Qwen3.5-35B-A3B-Q4_K_M.gguf
curl -fL -C - -o "$model" "https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/resolve/main/$model"
export PTXAS_OPTIONS='--override-directive-values --maxrregcount=192'
ft serve --model "$model" --served-model-name qwen --text-model-only --enable-timings-report \
  --moe-strategy fused --dtype float16 --attn triton --num-tokens 1024 --graph 1
```

In a second terminal, wait for `curl -s http://127.0.0.1:1919/health` to report `"status":"ok"`. Then send a short request.

```sh
curl -fsS http://127.0.0.1:1919/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen","messages":[{"role":"user","content":"What is the capital of France? Answer with just the city."}],"temperature":0,"max_tokens":16,"chat_template_kwargs":{"enable_thinking":false}}'
```

The V100 returned `Paris` with `finish_reason` set to `stop`.

## 5. Benchmarks

Run the streaming client against the server. It sends one warm-up and three different prompts, one at a time, at temperature zero. [Benchmark client and prompts](./benchmarks/)

```sh
mkdir -p ~/freetoken-v100/benchmark && cd ~/freetoken-v100/benchmark
base=https://raw.githubusercontent.com/nbald/volta-forever/main/docs/guides/benchmarks
curl -fsSLO "$base/freetoken.py" && curl -fsSLO "$base/prompts.json"
python3 freetoken.py
```

Each measured request read 227 uncached tokens and generated 64. The responses started with the requested cities, Kyoto, Lyon and Lima. Median time to the first token was 262 ms at the server and 270 ms over loopback. [Measurements, generated text and graph counters](./benchmarks/freetoken-graphs.json)

| Scheduler phase | Three runs (tokens/s) | Median (tokens/s) |
| --- | --- | ---: |
| Prefill | 884.56, 874.02, 885.13 | **884.56** |
| Decode | 109.30, 109.26, 97.52 | **109.26** |

FreeToken’s `prompt_ms` spans the first prefill batch through the first sampled token. Its `predicted_ms` covers the following 63 tokens. Both exclude time in the request queue.

On the same physical V100, native llama.cpp built for sm_70 processed the same GGUF file and prompts at **474.57 prefill** and **99.77 decode** tokens/s. FreeToken with VF was 1.86× as fast in prefill and 1.10× as fast in decode for this workload. [Native reference and commands](./benchmarks/llamacpp-reference.json)

`--graph 1` captures the batch-one decode and replays it for each new token. All four answers matched the earlier eager run verbatim. In a separate eight-step comparison on the same server, median time spent submitting the forward pass fell from 58.60 ms to 0.71 ms with CUDA graphs. The corresponding CUDA stream spans were 58.98 and 8.97 ms. [Eager baseline](./benchmarks/freetoken-results.json) · [Graph measurements](./benchmarks/freetoken-graphs.json)

## 6. Additional notes

Validation used Ubuntu 24.04, VF 0.1.1, FreeToken 0.2.2, PyTorch 2.14.1+cu130 and driver 580.178.04 on a Tesla V100-SXM2-32GB. FreeToken placed 18.1 GiB of expert weights on the GPU. The register limit in step 4 lets its Triton GatedDeltaNet prefill kernel run on Volta. A separate test matched all 262,144 outputs and 524,288 updated state values against a CPU reference.

FreeToken Desktop 0.2.0-beta.27 also rendered its setup screen and model catalog under Xvfb. Desktop is optional for the local inference API. [Desktop download](https://www.flashml.ai/)

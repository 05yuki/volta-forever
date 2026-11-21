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
ft serve --model "$model" --served-model-name qwen --text-model-only \
  --moe-strategy fused --dtype float16 --attn triton --num-tokens 1024 --graph 0
```

In a second terminal, wait for `curl -s http://127.0.0.1:1919/health` to report `"status":"ok"`. Then send a short request.

```sh
curl -fsS http://127.0.0.1:1919/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen","messages":[{"role":"user","content":"What is the capital of France? Answer with just the city."}],"temperature":0,"max_tokens":16,"chat_template_kwargs":{"enable_thinking":false}}'
```

## 5. Benchmarks

After the warmup request in step 4, time a second response. Divide its `usage.completion_tokens` by the elapsed seconds to get output tokens/s, including prompt processing and HTTP time.

```sh
curl -fsS -w '\n%{time_total} seconds\n' http://127.0.0.1:1919/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen","messages":[{"role":"user","content":"Write a short sentence about Paris."}],"temperature":0,"max_tokens":32,"chat_template_kwargs":{"enable_thinking":false}}'
```

FreeToken placed 18.1 GiB of expert weights across 40 layers on the V100 and reported 9.01 GiB of VRAM free after initialization. Engine warmup then failed on VF 0.1.0 because the Triton prefill kernel `chunk_gated_delta_rule_fwd_kkt_solve_kernel` exceeded the register limit. That run produced no tokens. With VF 0.1.1 and the register limit in step 4, the complete GatedDeltaNet prefill operator matched all 262,144 outputs and 524,288 updated state values from the CPU reference.

## 6. Additional notes

Validation used Ubuntu 24.04, volta-forever 0.1.0, FreeToken 0.2.2, PyTorch 2.14.1+cu130 and driver 580.178.04 on a Tesla V100-SXM2-32GB. Four real embedding rows matched the CPU exactly. A separate test ran the first expert's Q4_K gate and up projections, SwiGLU and Q5_K down projection. Its 2,048 outputs matched a CPU reference using the same Q8_1 activation quantization to a relative L2 error of 2.9 × 10⁻⁷.

FreeToken Desktop 0.2.0-beta.27 also rendered its setup screen and model catalog under Xvfb. Desktop is optional for the local inference API. [Desktop download](https://www.flashml.ai/)

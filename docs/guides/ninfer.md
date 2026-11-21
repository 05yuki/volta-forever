# Try upstream NInfer on a V100

The official NInfer CLI builds on Ubuntu, but its Qwen runtime requires compute capability 12.0. On our V100, the default VF profile stopped at that check before generating a token.

## 1. CUDA

On Ubuntu 24.04 x86-64, run this check for the proprietary NVIDIA R580 driver.

```bash
if ! nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | grep -E 'V100.*, 580[.]'; then
  sudo apt update && sudo apt install -y nvidia-driver-580
fi
```

If you installed the driver, reboot and repeat the check. It should print your V100 and driver version.

The official x86-64 build needs a toolkit that can compile for `sm_120a`. This check reuses a suitable compiler and otherwise installs NVIDIA’s current CUDA Toolkit. [NVIDIA installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#ubuntu)

```bash
export PATH="$PATH:${CUDA_HOME:-/usr/local/cuda}/bin"
if ! printf '%s\n' '__global__ void probe() {}' | nvcc -arch=sm_120a -cubin -x cu -o /dev/null - 2>/dev/null; then
  sudo apt update && sudo apt install -y ca-certificates curl
  repo=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64
  curl -fsSL "$repo/cuda-archive-keyring.gpg" | sudo tee /usr/share/keyrings/cuda-archive-keyring.gpg >/dev/null
  echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] $repo/ /" | sudo tee /etc/apt/sources.list.d/cuda-ubuntu2404-x86_64.list
  sudo apt update && sudo apt install -y cuda-toolkit
  export PATH="/usr/local/cuda/bin:$PATH"
fi
```

## 2. VF

```sh
sudo apt update && sudo apt install -y ca-certificates curl python3 xz-utils
curl -fsSL https://raw.githubusercontent.com/nbald/volta-forever/main/install.sh | bash
export PATH="$HOME/.local/bin:$PATH"
vf doctor
```

## 3. Engine

Upstream distributes the CLI as source. Build its official x86-64 preset. [NInfer build instructions](https://github.com/Neroued/ninfer#build)

```sh
sudo apt install -y build-essential cmake ninja-build git pkg-config \
  libavformat-dev libavcodec-dev libavutil-dev libswscale-dev libjpeg-dev \
  libhwloc-dev liburing-dev libcurl4-openssl-dev
git clone https://github.com/Neroued/ninfer.git
cd ninfer
cmake --preset rtx5090-release
cmake --build build/rtx5090-release --parallel 4 --target ninfer
```

## 4. Test

Use upstream's 22.8 GB Qwen3.6-35B-A3B artifact. Routed experts use Q4 gate/up and Q5/Q6 down projections. [Model card](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer)

```sh
curl -fL --retry 3 -o qwen3_6_35b_a3b.ninfer \
  https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer/resolve/main/qwen3_6_35b_a3b.ninfer
VF_CAPABILITY=12.0 vf ./build/rtx5090-release/apps/ninfer qwen3_6_35b_a3b.ninfer \
  --prompt 'What is the capital of France? Answer with one word.' \
  --max-context 512 --max-tokens 12 --no-thinking --temperature 0
```

## 5. Benchmarks

The test command prints prompt and generated token counts, prefill speed and decode speed after generation. On a Tesla V100-SXM2-32GB with driver 580.178.04, the CLI built with CUDA 13.5 loaded 19.6 GiB of real weights. An earlier VF build reached `engine ready` and timed out on the first prompt. A later VF build passed NInfer’s numerical check for the prompt’s Q8 projection, taking 2.51 s to compile the kernel on first use.

## 6. Additional notes

With the fixes in VF 0.1.1, a structural test using the original tokenizer and 934 zero-filled tensors completed the 35B graph at context 512, processing 24 prompt tokens and producing two tokens. [VF 0.1.1 release](https://github.com/nbald/volta-forever/releases/tag/v0.1.1)

On VF 0.1.0, NInfer’s default profile stopped at `Qwen3.5 family runtime requires compute capability 12.0`. With `VF_CAPABILITY=12.0`, its upstream token-count test stopped at `cudaErrorSymbolNotFound`. [Upstream capability check](https://github.com/Neroued/ninfer/blob/ebdc67e584bcd9ebe16ee63af9abeb2118bac6cd/src/models/qwen3_5/program/planning/startup.cpp#L575-L580)

The CLI rejected Qwen3.5-35B-A3B Q4_K_M with `NInfer accepts only .ninfer artifacts`. Upstream publishes Qwen3.6-35B-A3B as a separate checkpoint in its own format. [Registered models](https://github.com/Neroued/ninfer#supported-models-and-deployments)

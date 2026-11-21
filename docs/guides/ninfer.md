# Run NInfer on a V100

The official NInfer CLI ran Qwen3.6-35B-A3B on one 32 GB Tesla V100. Set `VF_CAPABILITY=12.0` when launching NInfer so its hardware check accepts the device.

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

The test returned **Paris** on a Tesla V100-SXM2-32GB with driver 580.178.04, VF 0.1.1 and the upstream CLI built with CUDA 13.5. The CLI reported 24 input tokens and two output tokens for this answer.

## 5. Benchmarks

The CLI prints prefill and decode speeds after each request. For a warm benchmark, start the official server and send requests to `/v1/chat/completions`. Each response includes `timings.prompt_per_second`, `timings.predicted_per_second` and the token counts. [NInfer request timings](https://github.com/Neroued/ninfer/blob/master/docs/serving.md#llamacpp-compatible-request-observations)

```sh
cmake --build build/rtx5090-release --parallel 4 --target ninfer-serve
VF_CAPABILITY=12.0 vf ./build/rtx5090-release/apps/ninfer-serve qwen3_6_35b_a3b.ninfer \
  --host 127.0.0.1 --port 9086 --max-context 512 --no-thinking --no-prefix-reuse
```

In a second terminal, run the shared warm-up and three measured chats. The client records the server’s prefill and decode times for each reply.

```sh
base=https://raw.githubusercontent.com/nbald/volta-forever/main/docs/guides/benchmarks
curl -fsSLO "$base/strata-ninfer.py"
curl -fsSLO "$base/prompts.json"
python3 strata-ninfer.py ninfer
```

## 6. Additional notes

This demo needs the linked-kernel fixes included in VF 0.1.1. VF 0.1.0 stopped with `cudaErrorSymbolNotFound` in NInfer’s token-count test even with the 12.0 profile. [VF release](https://github.com/nbald/volta-forever/releases/tag/v0.1.1)

Qwen3.5-35B-A3B Q4_K_M GGUF was rejected with `NInfer accepts only .ninfer artifacts`. The official `qwen3_6_35b_a3b` conversion recipe accepted the Qwen3.5 BF16 tensor names and shapes across all 14 source shards. Inference was verified with the published Qwen3.6 artifact. [NInfer conversion recipes](https://github.com/Neroued/ninfer/blob/master/docs/weight-conversion.md)

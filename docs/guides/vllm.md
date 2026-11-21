# vLLM on a Tesla V100

## 1. CUDA

On Ubuntu 24.04 x86-64, run this check for the proprietary NVIDIA R580 driver.

```bash
if ! nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | grep -E 'V100.*, 580[.]'; then
  sudo apt update && sudo apt install -y nvidia-driver-580
fi
```

If you installed the driver, reboot and repeat the check. It should print your V100 and driver version.

The Python wheels in step 3 supply CUDA 13, including the compiler.

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
mkdir -p ~/vllm-v100 && cd ~/vllm-v100
vf venv .venv
source .venv/bin/activate
pip install vllm
```

## 4. Test

Start the server. On the first launch it downloads Qwen2.5-0.5B-Instruct and compiles kernels, which can take several minutes.

```sh
export CUDA_HOME="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --dtype float16 --max-model-len 512 --enforce-eager \
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

On a Tesla V100-SXM2-32GB with driver 580.178.04, vLLM 0.34.0 and PyTorch 2.14.0+cu130 returned 32 tokens in 0.955, 0.970 and 0.958 seconds. Median end-to-end throughput was **33.4 tokens/s** for one request at a time with float16 weights.

## 6. Additional notes

VF 0.1.0 reported capability 7.0 to vLLM's NVML check, below the 7.5 required by `QuantTrio/Qwen3.5-35B-A3B-AWQ`. VF 0.1.1 reports 7.5 by default. Testing the NVML fix loaded all nine weight files on one V100 (21.3 GiB of GPU memory). This model's FlashInfer RMSNorm also needed a complete CUDA 13.5 toolkit with matching headers. A two-block diagnostic completed prefill and decoding with synthetic weights. Generation from the full checkpoint is still unverified.

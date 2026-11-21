# Run Strata on a V100

## 1. CUDA

On Ubuntu 24.04 x86-64, run this check for the proprietary NVIDIA R580 driver.

```bash
if ! nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | grep -E 'V100.*, 580[.]'; then
  sudo apt update && sudo apt install -y nvidia-driver-580
fi
```

If you installed the driver, reboot and repeat the check. It should print your V100 and driver version.

Strata ships the CUDA runtime and cuBLAS with its engine. The host can run this demo without installing a CUDA toolkit.

## 2. VF

```sh
sudo apt update && sudo apt install -y ca-certificates curl python3 xz-utils
curl -fsSL https://raw.githubusercontent.com/nbald/volta-forever/main/install.sh | bash
export PATH="$HOME/.local/bin:$PATH"
vf doctor
```

## 3. Engine

Run the upstream Linux installer. It installs the terminal in your home directory and checks its signature. [Strata installation](https://github.com/Niko1221/Strata#install)

```sh
curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install-strata.sh
sh install-strata.sh
strata --version
```

## 4. Test

Strata registers Unsloth's Qwen3.6-35B-A3B MTP GGUF in the `UD-Q4_K_XL` profile. Allow 22.9 GB for the weights and at least 32 GB of system RAM. The command selects the CUDA 13 engine, enables lazy kernel loading and requests an 8,192-token context. A successful launch prints a local chat URL where you can ask “What is the capital of France?” [Strata model catalog](https://github.com/Niko1221/Strata/blob/main/docs/MODELS.md#qwen36-35b-a3b)

```sh
CUDA_MODULE_LOADING=LAZY STRATA_CUDA=13 vf strata launch --family qwen36 --model UD-Q4_K_XL \
  --context 8k --vision off --access local --port 9085 --no-api-key
```

## 5. Benchmarks

Once the server answers a prompt, run Strata’s smoke benchmark in a second terminal to measure request latency and answer tokens per second.

```sh
strata bench run --url http://127.0.0.1:9085 --suite serving-smoke --out "strata-benchmark-$(date +%s)"
```

With default eager loading and a 512-token context, Qwen3.6-35B-A3B produced no tokens on a Tesla V100-SXM2-32GB, driver 580.178.04, VF 0.1.0 and Strata CUDA 13 engine 0.1.63. Strata checked all 753 tensors and uploaded 4.40 GiB before reporting `cuBLAS handle failed`. An isolated cuBLAS check returned `CUBLAS_STATUS_ALLOC_FAILED` in eager mode and succeeded in lazy mode. [Strata’s loading default](https://github.com/Niko1221/Strata/blob/v0.2.17/engine/src/program/generate.cpp#L3702-L3711)

## 6. Additional notes

The standard Qwen3.5-35B-A3B Q4_K_M file has 733 tensors. Strata rejected it with `missing qwen35moe.nextn_predict_layers`. Unsloth also publishes a 753-tensor Qwen3.5 file with the MTP layer. [Qwen3.5 MTP weights](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-MTP-GGUF)

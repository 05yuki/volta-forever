# Changes

## Unreleased

- The PTX translator handles packed BF16 subtraction, packed FP8-to-BF16 conversion and BF16-to-FP16 conversion.
- vLLM's standard NVML backend now sees the same capability as its CUDA 13 PyTorch runtime under `vf`. Its NVML query reports 7.5 for Volta by default and honors `VF_CAPABILITY`.
- Under `vf` or `vf venv`, direct CUDA driver clients with no embedded GPU code also see 7.5 on a V100 by default. Direct clients can still read native Volta kernel binary versions. CUDA 12 runtimes keep their previous handling of explicit overrides, and other GPUs retain their physical capabilities.

## 0.1.0 (2026-11-21)

The first version made public. It runs software built with CUDA 13 on a Tesla V100 with the 580
driver:

- `libcuda.so.1`, in front of NVIDIA's driver library: PTX for later GPUs retargeted, with the
  instructions that Volta lacks rewritten (the tensor-core products of Turing, Ampere and
  Blackwell, matrix loads, bf16, FP8, FP4, block-scaled products); machine code for sm_75
  translated to sm_70 when it is loaded; kernels that cannot run refused with a reason; a cache
  of what was worked out.
- Forwarding libraries for 22 libraries of CUDA 13, served by their CUDA 12.9.2 builds; an
  nvJitLink 13 served by nvJitLink 12.9.2; an audit library for programs that open their
  libraries by path.
- `vf`, a launcher (`volta-forever` under its long name): `vf COMMAND` runs a command through
  the layer, `vf venv DIR` makes a virtual environment of Python that runs through it by
  itself, `env` prints the environment for a shell session, `fetch` downloads the CUDA 12.9.2
  libraries from NVIDIA's servers, `doctor` says what is missing; the offline tool is
  `vf-translate`, which the launcher has as `vf inspect`, `extract`, `adapt`, `translate`.
- A program can ask whether the layer is there: `volta_forever_version()` and
  `volta_forever_abi()` in `libcuda.so.1`, `VOLTA_FOREVER` in the launcher's environment.
- Tested with PyTorch 2.14.0, llama.cpp b12418, vLLM 0.34.0, SGLang 0.5.23 and ZML
  (`LIMITS.md`).

Known limits of this version are in `LIMITS.md`; how the translations are checked, and what no
check covers, in `docs/validation.md`.

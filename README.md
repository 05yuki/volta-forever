# volta-forever

**Your Tesla V100 is not dead.** CUDA 13 dropped Volta. volta-forever brings it back.

Run unmodified CUDA 13 programs on a V100: today's PyTorch wheels, vLLM, SGLang, llama.cpp. No
patch, no rebuild. Put two letters in front of your command:

```sh
vf python3 -c "import torch; print(torch.cuda.get_device_name())"    # Tesla V100-SXM2-32GB
vf python3 -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-0.5B-Instruct
vf llama-completion -m model.gguf -p "Hello" -n 48 -ngl 99
```

What a GPU of 2017 lacks is emulated on what it has: the tensor-core instructions of Turing,
Ampere and Blackwell, so your V100 now speaks BF16, FP8 and NVFP4.

> **Preview (0.1.0-preview).** Young software, run on Tesla V100-SXM2-32GB cards with NVIDIA's
> driver 580.178.04. What it cannot run it refuses with a reason; `LIMITS.md` has what does not
> work, what is slower and what was not tried. Not NVIDIA's, and not endorsed by NVIDIA.

## Why

CUDA 13 no longer compiles for Volta GPUs (compute capability 7.0, `sm_70`), and what is built
with it carries code for later GPUs only. On a V100 such a program stops with
`CUDA_ERROR_NO_BINARY_FOR_GPU`, and NVIDIA's own CUDA 13 libraries refuse the card. The card
itself is fine, and NVIDIA's driver still runs it (the 580 branch is the last that does).

volta-forever stands between the program and the driver and rewrites the GPU code as the
program loads it. The program is not touched, and neither is the driver.

## Quick start

You need a Tesla V100, NVIDIA's driver 580, and Linux x86-64 with glibc 2.34 or later.

**1. Get volta-forever.** Unpack an archive and put its `bin` on your path. (There is no
binary release yet: `scripts/package.sh` makes the archive from a built tree, `Build` below;
the launcher also works from the built tree itself, as `scripts/vf`.) The command is `vf`;
`volta-forever` is the same program under its long name.

```sh
tar --zstd -xf volta-forever-VERSION-linux-x86_64.tar.zst
export PATH=$PWD/volta-forever-VERSION-linux-x86_64/bin:$PATH
```

**2. Get NVIDIA's cuBLAS for CUDA 12.9.2.** NVIDIA's libraries are NVIDIA's to distribute, so
none is in the archive. This downloads from NVIDIA's servers the one that a program cannot
keep from CUDA 13 (1 GB):

```sh
vf fetch
```

A program's other CUDA 13 libraries (the cuFFT, cuSOLVER and cuSPARSE of a PyTorch wheel) are
translated like the rest of its code. `vf fetch --all` downloads their CUDA 12.9.2 builds too
(2.7 GB in all), which still have code for Volta, and the layer then uses those instead.

**3. Run your program** with `vf` in front:

```sh
vf python3 train.py
```

That is all. If something is missing, `vf doctor` says what: the card, the driver, the
libraries.

### With pip and virtual environments

Nothing changes for pip: install the ordinary wheels for CUDA 13, they bring their own CUDA
libraries, and the layer puts the right ones behind them when the program runs. A virtual
environment can be made to run through the layer by itself, so that there is no prefix to
remember, and so that a package which compiles GPU code when it is installed asks the GPU
what to compile for through the layer too (a V100 alone answers `sm_70`, which CUDA 13's
compiler refuses):

```sh
vf venv .venv                    # python3 -m venv .venv, and every Python of it runs through the layer
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu130
.venv/bin/python train.py        # no prefix
```

`vf venv` also takes an environment that is there already (made by `python -m venv` or by
`uv venv`), and `vf venv --off .venv` takes the layer out of it again. For a shell session
rather than an environment: `eval "$(vf env)"`.

`INSTALL.md` has the options, the switches and what the launcher sets, for those who would
rather set it by hand.

## What runs

Measured on one Tesla V100-SXM2-32GB, driver 580.178.04, with no switch set, on 2026-11-20 and
21. Every figure below has the model it was measured with and what it is compared with.

**Is the result right?** Compared with the same program on the CPU.

| Program | Model or workload | Result |
|---|---|---|
| PyTorch 2.14.0 and 2.14.1, the stock wheels for CUDA 13 (2.14.1 is what `pip install torch` brings today) | 925 operations and 32 small whole models, the tests' own (`tests/gpu/torch-ops.py`, `torch-models.py`) | 924 operations give the CPU's result, and the CPU does not do the other one; 32 models of 32 |
| llama.cpp b12418, the official build for CUDA 13.5 | Qwen2.5-0.5B-Instruct (GGUF, Q4_0), flash attention on and off | the CPU's text; next-token distributions at a KL divergence of 0.0017 from the CPU's |
| | Qwen3.8-27B with NVFP4 weights (GGUF) | the CPU's text; KL divergence 0.0005 |
| vLLM 0.34.0 (pip) | Qwen2.5-0.5B-Instruct in f16, three prompts, 50 tokens each, greedy | the engine's text and its HTTP server's are the CPU's (`transformers`, f32), token for token |
| SGLang 0.5.23 (pip), its default attention (FlashInfer) | the same | the same |
| ZML (upstream build, its own CUDA libraries) | Qwen3.5-9B in f16, one prompt | it answers with the expected word: compared with no reference, the weakest of these checks |

**What does it cost?** Compared with a build of the same program that has code for Volta,
where one exists.

| Program | Model or workload | Through the layer | The build for Volta |
|---|---|---|---|
| PyTorch 2.14.0 | 89 operations, each timed by itself | 1.00 times the reference's time (median); 0.79 to 1.65 times over the 89 | PyTorch 2.14.0 for CUDA 12.6, the last wheel with code for Volta |
| llama.cpp b12418 | Qwen2.5-0.5B-Instruct, Q4_0 and Q8_0: generation of 128 tokens | 410 to 472 tokens a second | 402 to 482 (the same sources built for `sm_70`) |
| | the same: a prompt of 2,048 tokens | 9,700 to 11,900 tokens a second | 19,600 to 27,800 |
| | Qwen3.8-27B with NVFP4 weights: generation | 34.6 to 34.7 tokens a second | 34.8 to 35.0 |
| | the same: a prompt of 512 tokens | 300 tokens a second | 224 |
| vLLM 0.34.0 | Qwen2.5-0.5B-Instruct in f16, one request at a time | 83 to 99 tokens a second | none exists: with PyTorch's build for CUDA 12.6 under it, vLLM 0.34.0 does not start (its own compiled part and its torchvision are built for CUDA 13) |
| SGLang 0.5.23 | the same | 45 to 52 tokens a second (FlashInfer), 55 to 59 (`--attention-backend triton`) | none to compare with |
| ZML | Qwen3.5-9B in f16 | 27 to 29 tokens a second | none to compare with (upstream builds for CUDA 13 only) |

The largest gaps are where the program runs instructions that a V100 does not have, which
are emulated: attention in f16 with Turing's tensor cores (the 1.65), and llama.cpp's prompts,
whose kernels for batches use integer tensor-core products. Where the layer is the faster of
the two, they do not run the same code: for a batch with NVFP4 weights, llama.cpp's own build
for Volta leaves the work to cuBLAS. `docs/performance.md` has each of these measured apart.

What to expect elsewhere:

- **Most code runs at the speed it would have natively**: it is translated once, when the
  program loads it, and then it is Volta code. The first run of a program is slower (it
  translates and compiles; the results are kept for the next runs).
- **What is emulated costs more**: tensor-core products in the shapes of later GPUs, BF16, FP8,
  the 4-bit formats. `docs/performance.md` has the measurements.
- **A kernel that cannot be made right is refused**, with its name and the reason, when the
  program asks for it. It is never run wrong knowingly. With PyTorch 2.14, 30 of 23,693
  functions are refused; none of them is reached by the 925 operations tested.
- **Not tried**: more than one GPU in a process, other Volta cards than the V100 (Titan V,
  Quadro GV100), drivers after 580.178.04. `LIMITS.md` has the whole list.

## For programs that know about it

A program built for later GPUs is told the compute capability that its code starts at (7.5),
so that it loads that code; the name of the card is still the V100's. What it cannot tell from
the driver's answers is that the card is a Volta with volta-forever in front of it. It can
ask:

```python
import ctypes
try:
    version = ctypes.CDLL("libcuda.so.1").volta_forever_version
    version.restype = ctypes.c_char_p
    print("volta-forever", version().decode())      # a Volta, and the layer in front of it
except AttributeError:
    print("NVIDIA's driver alone")
```

```c
const char *(*version)(void) = dlsym(RTLD_DEFAULT, "volta_forever_version");
if (version) { /* prefer the kernels written for Volta, when the program has some */ }
```

`volta_forever_version()` and `volta_forever_abi()` (the number of this interface: 1) are in
`libcuda.so.1` only when it is the layer's. The launcher (`vf`) also sets `VOLTA_FOREVER` (the
version) and `VOLTA_FOREVER_HOME` in the environment, for scripts. A program that has kernels
written for Volta (PTX for `sm_70`, which the driver compiles as it is) may then prefer them
to code for later GPUs, which would be translated or emulated.

## How it works

```
 program built with CUDA 13
   |
   +-- libcuda.so.1          the proxy: in front of NVIDIA's driver library; adapts every image
   |                         of GPU code that the program loads
   +-- forward/*.so          22 libraries with the names of CUDA 13's (cuBLAS, cuFFT, cuSOLVER,
   |                         cuSPARSE, cuRAND, NPP, nvJPEG, NVRTC), whose functions call the
   |                         CUDA 12.9.2 builds, which still have code for Volta
   +-- libnvJitLink.so.13    nvJitLink 13's interface, served by nvJitLink 12.9.2
   +-- libvf_audit.so        for programs that open their libraries by path: gives them the
                             libraries above instead (the loader's audit interface)
```

- **PTX** (code that the driver compiles) is retargeted, and the instructions that Volta lacks
  are replaced by sequences that it has.
- **Machine code for `sm_75`** is translated to machine code for `sm_70`, function by function.
- **NVIDIA's CUDA 13 libraries** are served by their CUDA 12.9 builds.

The documents:

- **`docs/architecture.md`**: the parts, the path of a module from the program to the driver,
  what a program is told about the GPU, the cache, the switches.
- **`docs/emulations.md`**: what is computed in place of each instruction that a V100 lacks,
  and how exact it is: for the 4-bit formats and the block-scaled products, against results
  computed by an RTX 5090.
- **`docs/translator.md`**: how machine code for `sm_75` becomes machine code for `sm_70`:
  uniform registers, tensor-core products, the timing rules of the hardware, what is refused.
  `docs/pairs.md` is the specification of one part of it.
- **`docs/validation.md`**: how a wrong translation would be found, and what none of the checks
  covers.
- **`docs/performance.md`**: what it costs, and where the time goes.

## Build

```sh
scripts/build.sh                    # the Rust workspace -> dist/lib, dist/bin
scripts/fetch-all.sh redist         # CUDA 12.9.2 and 13.5.0 from NVIDIA's servers (the forwarding
                                    # libraries are generated from their exports and headers)
scripts/build-forwarders.sh         # -> dist/forward
scripts/vf doctor                   # the launcher works from the source tree too
scripts/package.sh                  # -> dist/volta-forever-VERSION-linux-x86_64.tar.zst
```

It needs Rust (edition 2024; built here with 1.100) and gcc. No file of NVIDIA's is in this
repository or in the archive.

## Tests

`tests/README.md` says what each suite runs, against what, and what it needs.

- Without a GPU: the unit tests (`cargo test --workspace`); the Rust translator against the
  Python reference, byte for byte (`tests/sass/compare.sh`); the independent validator
  (`reference/validate.py`) and its controls; the launcher and the proxy against a stand-in
  for the driver.
- On a V100: test kernels against native builds, host models and reference vectors from an RTX
  5090 (`tests/gpu/run.sh`); PyTorch's operations and whole models against the CPU; llama.cpp,
  vLLM, SGLang and ZML.

## Layout

| | |
|---|---|
| `scripts/vf` | the launcher (`scripts/volta-forever` is a link to it) |
| `crates/vf-proxy` | `libcuda.so.1` |
| `crates/vf-adapt` | what to give the driver in place of an image |
| `crates/vf-ptx` | PTX: parsing, the rewriting of instructions for Volta |
| `crates/vf-sass`, `crates/vf-tables` | the machine-code translator and its instruction tables |
| `crates/vf-fatbin` | fat binaries |
| `crates/vf-fwdgen` | the generator of the forwarding libraries |
| `crates/vf-jitlink`, `crates/vf-audit` | the nvJitLink adapter, the audit library |
| `crates/vf-cli` | the offline tool `vf-translate`, which the launcher has as `vf inspect`, `vf extract`, `vf adapt` and `vf translate` |
| `reference/` | the translator's reference implementation in Python, the independent validator and its controls, the generator of the tables, the scripts that derive the timing rules from NVIDIA's compiled code |
| `tests/` | the suites; `tests/kernels/` has the test kernels and the reference vectors |
| `scripts/` | build, package, fetch |
| `docs/` | the documents above |

## Licence

Apache License 2.0 (`LICENSE`), except the instruction tables of `crates/vf-tables`, which
derive from the instruction descriptions of the [denvdis](https://github.com/redplait/denvdis)
project and are distributed, like them, under the GNU Lesser General Public License 2.1
(`crates/vf-tables/LICENSE`). `NOTICE` says what the binaries contain.

NVIDIA and CUDA are trademarks of NVIDIA Corporation. This project is not affiliated with
NVIDIA.

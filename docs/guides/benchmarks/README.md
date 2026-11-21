# Single-request 35B benchmarks

Start the engine using its Ubuntu guide. Run the benchmark on the same computer, with one model loaded and the server idle. Each run uses the model's chat template at temperature zero, with thinking disabled and a limit of 64 output tokens. The warm-up starts with Lisbon. The three measured prompts start with Kyoto, Lyon and Lima and ask for a handover based on different notebook entries. Read the generated text when checking a new engine. [Common prompts](prompts.json)

## vLLM

In a second terminal, download the client and prompts together and run the client. It uses the Python standard library.

```sh
mkdir -p ~/vllm-v100/benchmark && cd ~/vllm-v100/benchmark
base=https://raw.githubusercontent.com/nbald/volta-forever/main/docs/guides/benchmarks
curl -fsSLO "$base/vllm.py" && curl -fsSLO "$base/prompts.json"
python3 vllm.py
```

Each request produces one new observation in each vLLM phase histogram. The client divides the prompt token count by the time in prefill. The first output token is available at the end of prefill, so decode throughput uses the remaining output tokens and the time in decode. It prints each response and the median of the three rates for each phase. [vLLM Ubuntu guide](../vllm.md)

| Prompt | Input tokens | Output tokens | Prefill (tokens/s) | Decode (tokens/s) |
| --- | ---: | ---: | ---: | ---: |
| Kyoto | 227 | 64 | 211.52 | 9.22 |
| Lyon | 227 | 64 | 210.54 | 9.23 |
| Lima | 227 | 64 | 211.74 | 9.23 |
| **Median** | | | **211.52** | **9.23** |

This diagnostic run used eager execution with assembler optimization disabled on a Tesla V100-SXM2-32GB using vLLM 0.34.0, Volta-Forever 0.1.1 and CUDA Toolkit 13.5. Each rate uses vLLM's native counters. The recorded responses and phase durations are available with the checkpoint revision. [Measurement data](vllm-results.json) · [Checkpoint](https://huggingface.co/QuantTrio/Qwen3.5-35B-A3B-AWQ/tree/2f4fac011fc06e736782b0b59394577790da7d79)

## SGLang

Run the same four prompts against the idle server started by the SGLang guide.

```sh
mkdir -p ~/sglang-v100/benchmark && cd ~/sglang-v100/benchmark
base=https://raw.githubusercontent.com/nbald/volta-forever/main/docs/guides/benchmarks
curl -fsSLO "$base/sglang.py" && curl -fsSLO "$base/prompts.json"
python3 sglang.py
```

The client sets a fresh cache salt on every chat. It verifies the input and output token counts and the absence of device, host and storage cache hits. Prefill divides the input tokens by the GPU timer for `extend`. Decode uses the output tokens after the first token and the GPU timer for `decode`. [SGLang Ubuntu guide](../sglang.md) · [Measurement data and configuration](sglang-results.json)


## FreeToken

Use the scheduler timing report in the [FreeToken guide](../freetoken.md#5-benchmarks).
The initial eager run measured 704.55 prefill and 14.85 decode tokens/s. Native llama.cpp
on the same V100 and Q4_K_M checkpoint measured 468.67 and 99.67 tokens/s with the same
three prompts, one request at a time. Each generated 64 tokens from 227 uncached input
tokens. [FreeToken measurements](freetoken-results.json) · [Native reference and reproduction commands](llamacpp-reference.json)

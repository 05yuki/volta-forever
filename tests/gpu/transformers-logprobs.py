#!/usr/bin/env python3
"""Control: transformers on GPU through the layer vs CPU, both float32.

This is the control that shows the method works: the GPU through the layer
should give the CPU's log-probabilities to within 1e-5 or so.

Environment:
  MODEL             path to the model (required)
  VF_DTYPE        float32 or float16 (default: float32)
  MAX_NEW_TOKENS    tokens to generate (default: 64)
"""
import sys, os, json, time, math

MODEL = os.environ['MODEL']
DTYPE = os.environ.get('VF_DTYPE', 'float32')
MAX_NEW_TOKENS = int(os.environ.get('MAX_NEW_TOKENS', '64'))

PROMPTS = [
    "The capital of France is",
    "What is 2 + 2? The answer is",
    "Explain briefly what a GPU is:",
]

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch_dtype = torch.float32 if DTYPE == 'float32' else torch.float16
tokenizer = AutoTokenizer.from_pretrained(MODEL)

# ---- CPU model ----
print(f'Loading CPU model ({DTYPE})...', flush=True)
cpu_model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch_dtype).to('cpu')
cpu_model.eval()

# ---- GPU model ----
print(f'Loading GPU model ({DTYPE})...', flush=True)
gpu_model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch_dtype).to('cuda')
gpu_model.eval()

for i, prompt in enumerate(PROMPTS):
    ids = tokenizer.encode(prompt)
    prompt_tensor_cpu = torch.tensor([ids])
    prompt_tensor_gpu = torch.tensor([ids]).to('cuda')

    # CPU generation
    with torch.no_grad():
        cpu_gen = cpu_model.generate(prompt_tensor_cpu, max_new_tokens=MAX_NEW_TOKENS,
                                     do_sample=False, temperature=None, top_p=None)
    cpu_gen_ids = cpu_gen[0][len(ids):].tolist()

    # GPU generation
    with torch.no_grad():
        gpu_gen = gpu_model.generate(prompt_tensor_gpu, max_new_tokens=MAX_NEW_TOKENS,
                                     do_sample=False, temperature=None, top_p=None)
    gpu_gen_ids = gpu_gen[0][len(ids):].tolist()

    # Token comparison
    n_match = 0
    for j, (ct, gt) in enumerate(zip(cpu_gen_ids, gpu_gen_ids)):
        if ct != gt:
            print(f'  prompt {i}: DIFFER at token {j}: '
                  f'cpu={ct}({tokenizer.decode([ct])!r}) gpu={gt}({tokenizer.decode([gt])!r})', flush=True)
            break
        n_match += 1
    else:
        if len(cpu_gen_ids) == len(gpu_gen_ids):
            print(f'  prompt {i}: ALL MATCH ({len(cpu_gen_ids)} tokens)', flush=True)
        else:
            print(f'  prompt {i}: match to shorter ({min(len(cpu_gen_ids), len(gpu_gen_ids))})', flush=True)
        n_match = min(len(cpu_gen_ids), len(gpu_gen_ids))

    # Teacher-forced logprob comparison: first token prediction
    # Feed just the prompt to both models and compare the logits for the first generated token
    with torch.no_grad():
        cpu_out = cpu_model(prompt_tensor_cpu)
        gpu_out = gpu_model(prompt_tensor_gpu)

    cpu_logits = cpu_out.logits[0, -1].float()  # [vocab_size]
    gpu_logits = gpu_out.logits[0, -1].float().cpu()  # [vocab_size]

    cpu_lp = torch.nn.functional.log_softmax(cpu_logits, dim=0)
    gpu_lp = torch.nn.functional.log_softmax(gpu_logits, dim=0)

    # Compare all log-probs
    diff = (cpu_lp - gpu_lp).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    # Compare top-20
    cpu_top20_vals, cpu_top20_ids = torch.topk(cpu_lp, 20)
    gpu_top20_vals, gpu_top20_ids = torch.topk(gpu_lp, 20)

    top20_max_diff = 0.0
    for k in range(20):
        tid = int(cpu_top20_ids[k])
        cpu_v = float(cpu_lp[tid])
        gpu_v = float(gpu_lp[tid])
        d = abs(cpu_v - gpu_v)
        if d > top20_max_diff:
            top20_max_diff = d

    cpu_argmax = int(torch.argmax(cpu_lp))
    gpu_argmax = int(torch.argmax(gpu_lp))

    print(f'  prompt {i}: first-token logprob comparison:', flush=True)
    print(f'    max|diff| over all vocab: {max_diff:.2e}', flush=True)
    print(f'    mean|diff|: {mean_diff:.2e}', flush=True)
    print(f'    max|diff| over top-20: {top20_max_diff:.2e}', flush=True)
    print(f'    cpu argmax: {cpu_argmax}({tokenizer.decode([cpu_argmax])!r}) '
          f'gpu argmax: {gpu_argmax}({tokenizer.decode([gpu_argmax])!r})', flush=True)
    print(f'    tokens matched: {n_match}', flush=True)

del cpu_model, gpu_model
print('\nDone.', flush=True)

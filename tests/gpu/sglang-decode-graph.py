"""Check SGLang's FP16 decode attention after CUDA graph capture on a V100."""

import math

import torch
from sglang.kernels.ops.attention.decode_attention import decode_attention_fwd


def reference(query, keys, values, indices):
    heads_per_kv = query.shape[1] // keys.shape[1]
    keys = keys[indices.long()].float().repeat_interleave(heads_per_kv, dim=1)
    values = values[indices.long()].float().repeat_interleave(heads_per_kv, dim=1)
    scores = torch.einsum("hd,khd->hk", query[0].float(), keys)
    weights = (scores / math.sqrt(keys.shape[-1])).softmax(-1)
    return torch.einsum("hk,khd->hd", weights, values)


def check(length, splits, generator):
    heads, kv_heads, width = 16, 2, 256
    query = torch.randn(1, heads, width, generator=generator).half()
    keys = torch.randn(512, kv_heads, width, generator=generator).half()
    values = torch.randn(512, kv_heads, width, generator=generator).half()
    indices = torch.randperm(512, generator=generator)[:length].int()
    gpu_q, gpu_k, gpu_v, gpu_indices = (t.cuda() for t in (query, keys, values, indices))
    output = torch.empty_like(gpu_q)
    offsets = torch.tensor([0, length], device="cuda", dtype=torch.int32)
    partitions = torch.tensor([splits], device="cuda", dtype=torch.int32)
    logits = torch.empty((1, heads, splits, width), device="cuda", dtype=torch.float32)
    logsumexp = torch.empty((1, heads, splits), device="cuda", dtype=torch.float32)

    def attention():
        decode_attention_fwd(
            gpu_q, gpu_k, gpu_v, output, offsets, gpu_indices, logits, logsumexp,
            partitions, splits, 1 / math.sqrt(width), 1.0, 1.0,
        )

    attention()
    expected = reference(query, keys, values, indices)
    torch.testing.assert_close(output[0].cpu().float(), expected, rtol=0.01, atol=0.002)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        attention()

    new_query = torch.randn(query.shape, generator=generator).half()
    new_expected = reference(new_query, keys, values, indices)
    assert (new_expected - expected).norm() > 0.2 * expected.norm()
    gpu_q.copy_(new_query)
    output.fill_(float("nan"))
    graph.replay()
    actual = output[0].cpu().float()
    assert torch.isfinite(actual).all()
    relative_error = (actual - new_expected).norm().item() / new_expected.norm().item()
    assert relative_error < 0.005, (length, splits, relative_error)
    print(f"KV length {length}, partitions {splits}, relative L2 error {relative_error:.6f}")


if __name__ == "__main__":
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(62291)
    for length in (23, 227, 288, 512):
        for splits in (1, 4):
            check(length, splits, generator)

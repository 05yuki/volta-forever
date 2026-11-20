#!/usr/bin/env python3
"""Times of PyTorch operations on the GPU, to compare two installations of PyTorch.

    torch-bench.py OUT [PATTERN...]       (run by tests/gpu/torch-bench.sh, which sets the paths)

Writes OUT, one line an operation: its name, the median time of a call in milliseconds (the GPU
waited for after each call), how many calls were timed, and two numbers that depend on the result
(its sum, and the sum of its magnitudes), so that two installations can be seen to have computed
the same thing. With PATTERN, only the
operations whose name contains one of them.

The operations are PyTorch's own kernels for the most part (by element, reductions, indexing,
sorting, normalisations, random numbers), on tensors large enough for the kernel to be what is
timed, then some that go through NVIDIA's libraries (products, convolutions) and two small models.
"""
import copy, sys, time


def build(torch):
    import torch.nn.functional as F
    dev = torch.device('cuda')
    g = torch.Generator(device='cpu').manual_seed(20261117)
    ops = []

    def op(name, fn):
        ops.append((name, fn))

    def randn(*shape, dtype=torch.float32):
        return torch.randn(*shape, generator=g).to(dtype).to(dev)

    def randint(low, high, *shape, dtype=torch.int64):
        return torch.randint(low, high, shape, generator=g).to(dtype).to(dev)

    n = 4096
    x, y = randn(n, n), randn(n, n)
    h, d = x.half(), randn(2048, 2048, dtype=torch.float64)
    i32, j32 = randint(-1000, 1000, n, n, dtype=torch.int32), randint(1, 1000, n, n, dtype=torch.int32)
    u8 = randint(0, 256, n, n, dtype=torch.uint8)

    # ---- by element ----------------------------------------------------------------------------------------
    op('add.f32', lambda: x + y)
    op('mul_add.f32', lambda: x * 1.5 + y)
    op('exp.f32', lambda: torch.exp(x))
    op('sigmoid.f32', lambda: torch.sigmoid(x))
    op('gelu.f32', lambda: F.gelu(x))
    op('relu.f32', lambda: torch.relu(x))
    op('where.f32', lambda: torch.where(x > 0, x, y))
    op('clamp.f32', lambda: torch.clamp(x, -0.5, 0.5))
    op('pow2.f32', lambda: x ** 2)
    op('maximum.f32', lambda: torch.maximum(x, y))
    op('add.f16', lambda: h + h)
    op('exp.f16', lambda: torch.exp(h))
    op('mul_add.f64', lambda: d * 1.5 + d)
    op('sqrt.f64', lambda: torch.sqrt(d.abs()))
    op('add.i32', lambda: i32 + j32)
    op('floor_divide.i32', lambda: i32 // j32)
    op('remainder.i32', lambda: i32 % j32)
    op('bitwise_and.i32', lambda: i32 & j32)
    op('minimum.i32', lambda: torch.minimum(i32, j32))
    op('compare.i32', lambda: i32 < j32)
    op('add.u8', lambda: u8 + u8)

    # ---- copies and conversions -----------------------------------------------------------------------------
    op('to_f16', lambda: x.half())
    op('to_f64', lambda: x.double())
    op('i32_to_f32', lambda: i32.float())
    op('u8_to_f32', lambda: u8.float())
    op('transpose_copy', lambda: x.t().contiguous())
    op('flip', lambda: torch.flip(x, (1,)))
    op('roll', lambda: torch.roll(x, 7, 1))
    op('cat', lambda: torch.cat((x, y), 1))
    op('clone', lambda: x.clone())
    op('fill', lambda: torch.full((n, n), 2.5, device=dev))
    op('arange', lambda: torch.arange(n * n, device=dev))
    op('tril', lambda: torch.tril(x))

    # ---- reductions and scans -------------------------------------------------------------------------------
    op('sum', lambda: x.sum())
    op('sum.dim', lambda: x.sum(1))
    op('mean.f16', lambda: h.float().mean())
    op('max', lambda: x.max())
    op('max.dim', lambda: x.max(1).values)
    op('argmax.dim', lambda: x.argmax(1))
    op('norm', lambda: torch.linalg.vector_norm(x))
    op('std.dim', lambda: x.std(1))
    op('cumsum', lambda: torch.cumsum(x, 1))
    op('any', lambda: (x > 5).any())
    op('count_nonzero', lambda: torch.count_nonzero(i32))
    op('sum.i32', lambda: i32.sum())
    op('prod.dim.f64', lambda: (d * 0.001 + 1).prod(1))

    # ---- indexing --------------------------------------------------------------------------------------------
    idx = randint(0, n, n, n)
    rows = randint(0, n, 2000)
    mask = x > 0.5
    flat = randint(0, n * n, 1 << 20)
    op('gather', lambda: torch.gather(x, 1, idx))
    op('index_select', lambda: torch.index_select(x, 0, rows))
    op('advanced_index', lambda: x[rows])
    op('take', lambda: torch.take(x, flat))
    op('masked_select', lambda: torch.masked_select(x, mask))
    op('masked_fill', lambda: x.masked_fill(mask, 0.0))
    op('nonzero', lambda: torch.nonzero(mask))
    op('scatter_add', lambda: torch.zeros(n, n, device=dev).scatter_add_(1, idx, x))
    op('index_add', lambda: torch.zeros(n, n, device=dev).index_add_(0, rows, x[:2000]))
    table, tokens = randn(50000, 512), randint(0, 50000, 64, 512)
    op('embedding', lambda: F.embedding(tokens, table))
    classes = randint(0, 1000, 256, 256)
    op('one_hot', lambda: F.one_hot(classes, 1000))

    # ---- sorting ---------------------------------------------------------------------------------------------
    v, keys = randn(1 << 22), randint(0, 1000, 1 << 22, dtype=torch.int32)
    op('sort.f32', lambda: torch.sort(v).values)
    op('sort.rows', lambda: torch.sort(x[:1024], 1).values)
    op('argsort.i32', lambda: torch.argsort(keys))
    op('topk', lambda: torch.topk(x, 50, 1).values)
    op('unique.i32', lambda: torch.unique(keys))
    op('bincount', lambda: torch.bincount(keys.long(), minlength=1000))
    op('searchsorted', lambda: torch.searchsorted(torch.sort(v).values, v[:1 << 20]))

    # ---- normalisations, activations, losses ------------------------------------------------------------------
    seq = randn(64, 512, 768)
    logits, target = randn(8192, 50000), randint(0, 50000, 8192)
    img = randn(32, 64, 112, 112)
    op('softmax', lambda: torch.softmax(logits, 1))
    op('log_softmax', lambda: torch.log_softmax(logits, 1))
    op('cross_entropy', lambda: F.cross_entropy(logits, target))
    op('layer_norm', lambda: F.layer_norm(seq, (768,)))
    op('batch_norm', lambda: F.batch_norm(img, None, None, training=True))
    op('group_norm', lambda: F.group_norm(img, 8))
    op('silu', lambda: F.silu(seq))
    op('dropout', lambda: F.dropout(seq, 0.1, True))
    op('max_pool2d', lambda: F.max_pool2d(img, 2))
    op('adaptive_avg_pool2d', lambda: F.adaptive_avg_pool2d(img, 7))
    op('mse_loss', lambda: F.mse_loss(x, y))

    # ---- random numbers --------------------------------------------------------------------------------------
    op('randn', lambda: torch.randn(n, n, device=dev))
    op('rand', lambda: torch.rand(n, n, device=dev))
    op('randint', lambda: torch.randint(0, 100, (n, n), device=dev))
    op('bernoulli', lambda: torch.bernoulli(torch.full((n, n), 0.3, device=dev)))
    op('multinomial', lambda: torch.multinomial(torch.softmax(logits[:1024], 1), 4))

    # ---- through NVIDIA's libraries ----------------------------------------------------------------------------
    a, b = randn(2048, 2048), randn(2048, 2048)
    kernel = randn(64, 64, 3, 3)
    op('lib.matmul', lambda: a @ b)
    op('lib.matmul.f16', lambda: a.half() @ b.half())
    op('lib.bmm', lambda: torch.bmm(seq, seq.transpose(1, 2)))
    op('lib.conv2d', lambda: F.conv2d(img, kernel, padding=1))

    # ---- two small models --------------------------------------------------------------------------------------
    torch.manual_seed(20261117)
    layer = torch.nn.TransformerEncoderLayer(768, 12, 3072, dropout=0.0, batch_first=True).to(dev).eval()
    cnn = torch.nn.Sequential(
        torch.nn.Conv2d(64, 64, 3, padding=1), torch.nn.BatchNorm2d(64), torch.nn.ReLU(),
        torch.nn.Conv2d(64, 128, 3, stride=2, padding=1), torch.nn.BatchNorm2d(128), torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(128, 1000)).to(dev).eval()
    half_layer, half_seq = copy.deepcopy(layer).half(), seq.half()
    heads, half_heads = randn(64, 12, 512, 64), randn(64, 12, 512, 64).half()
    op('attention', lambda: F.scaled_dot_product_attention(heads, heads, heads))
    op('attention.f16', lambda: F.scaled_dot_product_attention(half_heads, half_heads, half_heads))
    op('model.transformer_layer', lambda: layer(seq))
    op('model.transformer_layer.f16', lambda: half_layer(half_seq))
    op('model.cnn', lambda: cnn(img))
    return ops


def main():
    out, patterns = sys.argv[1], sys.argv[2:]
    import torch
    torch.backends.cudnn.benchmark = False
    lines = []
    with torch.no_grad():
        for name, fn in build(torch):
            if patterns and not any(p in name for p in patterns):
                continue
            try:
                result = None
                for _ in range(3):
                    result = fn()
                torch.cuda.synchronize()
                times, total = [], 0.0
                while total < 0.5 and len(times) < 300:
                    start = time.perf_counter()
                    result = fn()
                    torch.cuda.synchronize()
                    times.append(time.perf_counter() - start)
                    total += times[-1]
                times.sort()
                # What depends on the result: the sum of what it holds and the sum of its
                # magnitudes, in doubles. (The sum alone says nothing of a result whose values
                # cancel out, as the output of a normalisation does: it is a residue of rounding.)
                held = result.double()
                value, magnitude = (float(held.sum().item()), float(held.abs().sum().item())) if result.numel() else (0.0, 0.0)
                lines.append('%s\t%.4f\t%d\t%.6g\t%.6g' % (name, 1000 * times[len(times) // 2], len(times), value, magnitude))
            except Exception as e:                               # a kernel refused, a launch that fails
                lines.append('%s\tfailed\t0\t%s' % (name, str(e).splitlines()[0][:120] if str(e) else type(e).__name__))
            print(lines[-1], flush=True)
    open(out, 'w').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()

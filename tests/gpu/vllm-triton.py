#!/usr/bin/env python3
"""Triton kernel test: a simple add kernel compiled for sm_75 through the layer.

Verifies that Triton's cubins (which carry the REQNTID attribute 0x10) are
translated and produce correct results on a V100.
Prints PASS or FAIL.
"""
import sys, torch, triton, triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)

torch.manual_seed(42)
n = 4096
x_cpu = torch.randn(n)
y_cpu = torch.randn(n)
expected = x_cpu + y_cpu

x = x_cpu.cuda()
y = y_cpu.cuda()
out = torch.empty(n, device='cuda')
grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
try:
    add_kernel[grid](x, y, out, n, BLOCK_SIZE=256)
    torch.cuda.synchronize()
    result = out.cpu()
    if torch.equal(result, expected):
        print('PASS: Triton add kernel bitwise identical to CPU')
    elif torch.allclose(result, expected, atol=1e-5):
        print('PASS: Triton add kernel within tolerance')
    else:
        diff_idx = (result != expected).nonzero(as_tuple=True)[0]
        i = diff_idx[0].item()
        print(f'FAIL: first diff at [{i}]: gpu={result[i].item():.8e} cpu={expected[i].item():.8e}')
        sys.exit(1)
except RuntimeError as e:
    msg = str(e)
    if 'attribute 0x10' in msg:
        print('FAIL: attribute 0x10 (REQNTID) not recognised by translator')
    elif 'PTX JIT compilation failed' in msg:
        print(f'FAIL: PTX JIT compilation failed')
    else:
        print(f'FAIL: {e}')
    sys.exit(1)

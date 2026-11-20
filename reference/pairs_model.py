#!/usr/bin/env python3
"""Host check of an arrangement ("Q") of Turing products on Volta's instruction.

One slice of the attention kernel: LDSM.4 for A (two fragments of 16 rows by 8 k), LDSM.4 for B
(four fragments of 8 k by 8 columns), eight m16n8k8 products (each A by each B), f32 accumulators.
Q: for an A fragment a and a pair of B fragments (2p, 2p+1), two m8n8k4 in the row.col form whose
operands are whole rows of the memory the matrix loads read. Claim checked here, with the layouts
of tests/kernels/tcrun.c (ref_ldmatrix, ref_mma16, ref_v884): register i of the eight results in
lane L is register i & 3 of the Turing accumulator of (a, 2p + (i >> 2)) in lane pi(L).
Exact integers; the order of sums is not modelled."""
import random
random.seed(5)
rowA = [[random.randint(-3, 3) for k in range(8)] for r in range(32)]
rowB = [[random.randint(-3, 3) for k in range(8)] for r in range(32)]
C = {(a, b): [[random.randint(-9, 9) for n in range(8)] for R in range(16)] for a in range(2) for b in range(4)}

# Turing: what each thread holds after the eight products.
def turing():
    out = {}
    for a in range(2):
        for b in range(4):
            D = [[C[a, b][R][n] + sum(rowA[16 * a + R][k] * rowB[8 * b + n][k] for k in range(8)) for n in range(8)] for R in range(16)]
            for t in range(32):
                g, c = t >> 2, t & 3
                for i in range(4):
                    out[a, b, i, t] = D[g + 8 * (i // 2)][2 * c + (i & 1)]
    return out

def turing_c():
    out = {}
    for a in range(2):
        for b in range(4):
            for t in range(32):
                g, c = t >> 2, t & 3
                for i in range(4):
                    out[a, b, i, t] = C[a, b][g + 8 * (i // 2)][2 * c + (i & 1)]
    return out

def bits(L):
    h, q, k = L >> 4, (L >> 2) & 3, L & 3
    return h, q >> 1, q & 1, k >> 1, k & 1, k

def pi(L):
    h, sr, sc, k1, k0, k = bits(L)
    return 16 * sr + 8 * h + 4 * k0 + 2 * sc + k1

def rho_a(a, L):
    h, sr, sc, k1, k0, k = bits(L)
    return 16 * a + 8 * k1 + 4 * sr + 2 * h + k0

def rho_b(p, L):
    h, sr, sc, k1, k0, k = bits(L)
    return 8 * (2 * p + h) + 4 * sc + k

def v884_row_col(d, a, b):
    """d: 8 values a lane (in and out), a, b: 4 values a lane. Volta's m8n8k4, A by rows, B by columns."""
    for q in range(4):
        A = [[0] * 4 for _ in range(8)]; B = [[0] * 8 for _ in range(4)]; Cm = [[0] * 8 for _ in range(8)]
        for h in range(2):
            for k in range(4):
                lane = 16 * h + 4 * q + k
                for i in range(4):
                    A[k + 4 * h][i] = a[lane][i]
                    B[i][k + 4 * h] = b[lane][i]
                for i in range(8):
                    Cm[(k & 1) + (i & 2) + 4 * h][(i & 4) + (k & 2) + (i & 1)] = d[lane][i]
        D = [[Cm[r][c] + sum(A[r][i] * B[i][c] for i in range(4)) for c in range(8)] for r in range(8)]
        for h in range(2):
            for k in range(4):
                lane = 16 * h + 4 * q + k
                for i in range(8):
                    d[lane][i] = D[(k & 1) + (i & 2) + 4 * h][(i & 4) + (k & 2) + (i & 1)]

T, TC = turing(), turing_c()
wrong = total = 0
for a in range(2):
    for p in range(2):
        # the accumulators enter in the same arrangement: register i of lane L = Turing register of lane pi(L)
        d = [[TC[a, 2 * p + (i >> 2), i & 3, pi(L)] for i in range(8)] for L in range(32)]
        x = [rowA[rho_a(a, L)] for L in range(32)]
        y = [rowB[rho_b(p, L)] for L in range(32)]
        for half in range(2):
            v884_row_col(d, [r[4 * half:4 * half + 4] for r in x], [r[4 * half:4 * half + 4] for r in y])
        for L in range(32):
            for i in range(8):
                total += 1
                wrong += d[L][i] != T[a, 2 * p + (i >> 2), i & 3, pi(L)]
print('results compared', total, 'wrong', wrong)
inv = [None] * 32
for L in range(32):
    inv[pi(L)] = L
print('pi     ', [pi(L) for L in range(32)])
print('inverse', inv)
print('rows of A a thread loads (a=0)', [rho_a(0, L) for L in range(32)])
print('rows of B a thread loads (p=0)', [rho_b(0, L) for L in range(32)])

# ---- B read by a transposed matrix load: B_b[k][n] is half n of memory row 8b + k -------------------------
def turing_t():
    out = {}
    for a in range(2):
        for b in range(4):
            D = [[C[a, b][R][n] + sum(rowA[16 * a + R][k] * rowB[8 * b + k][n] for k in range(8)) for n in range(8)] for R in range(16)]
            for t in range(32):
                g, c = t >> 2, t & 3
                for i in range(4):
                    out[a, b, i, t] = D[g + 8 * (i // 2)][2 * c + (i & 1)]
    return out

def v884_row_row(d, a, b):
    for q in range(4):
        A = [[0] * 4 for _ in range(8)]; B = [[0] * 8 for _ in range(4)]; Cm = [[0] * 8 for _ in range(8)]
        for h in range(2):
            for k in range(4):
                lane = 16 * h + 4 * q + k
                for i in range(4):
                    A[k + 4 * h][i] = a[lane][i]
                    B[k][i + 4 * h] = b[lane][i]
                for i in range(8):
                    Cm[(k & 1) + (i & 2) + 4 * h][(i & 4) + (k & 2) + (i & 1)] = d[lane][i]
        D = [[Cm[r][c] + sum(A[r][i] * B[i][c] for i in range(4)) for c in range(8)] for r in range(8)]
        for h in range(2):
            for k in range(4):
                lane = 16 * h + 4 * q + k
                for i in range(8):
                    d[lane][i] = D[(k & 1) + (i & 2) + 4 * h][(i & 4) + (k & 2) + (i & 1)]

def rho_bt(p, L, half):
    h, sr, sc, k1, k0, k = bits(L)
    return 8 * (2 * p + h) + 4 * half + k, 4 * sc          # the memory row, and the first of its four halves

T = turing_t()
wrong = total = 0
for a in range(2):
    for p in range(2):
        d = [[TC[a, 2 * p + (i >> 2), i & 3, pi(L)] for i in range(8)] for L in range(32)]
        x = [rowA[rho_a(a, L)] for L in range(32)]
        for half in range(2):
            y = []
            for L in range(32):
                row, first = rho_bt(p, L, half)
                y.append(rowB[row][first:first + 4])
            v884_row_row(d, [r[4 * half:4 * half + 4] for r in x], y)
        for L in range(32):
            for i in range(8):
                total += 1
                wrong += d[L][i] != T[a, 2 * p + (i >> 2), i & 3, pi(L)]
print('B by a transposed load: results compared', total, 'wrong', wrong)
print('rows of B a thread loads (p=0, first half of k)', [rho_bt(0, L, 0)[0] for L in range(32)])
print('first half of the row it takes               ', [rho_bt(0, L, 0)[1] for L in range(32)])

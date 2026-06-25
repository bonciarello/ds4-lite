# Batched prefill — plan (branch: perf/prefill-batched)

## Problem
Prefill processes the prompt one token at a time (`ds4_dense_gpu_forward` per token),
so it **re-reads all 4.7GB of weights for every token** (N separate matvecs). A batched
matmul reads each weight tile **once** and applies it to all M prompt tokens, amortizing
the weight reads — this is the whole speedup.

## Baseline (M1 Max, Qwen2-7B Q4_K_M, this branch)
| metric | ds4 | llama.cpp |
|---|---:|---:|
| prefill (pp, 57-tok prompt) | ~37 t/s | 411 t/s (pp512) |
| decode (tg) | ~37 t/s | 44 t/s |

Prefill ≈ decode → zero amortization. Target: bring prefill toward 100s of t/s.

## Approach: process the prompt as M tokens in one batched forward
Per layer, the matmuls become matrix–matrix (X[M,K] · dequant(W[N,K])):
q/k/v/o and gate/up are q4_K, down and output are q6_K.

## Increments (validate + benchmark each)
1. **Matmul kernels** `kernel_mul_mm_q4_K_f32` / `_q6_K_f32`: instantiate the existing
   templated `kernel_mul_mm` (simdgroup_float8x8) for block_q4_K/q6_K, reusing
   `dequantize_q4_K` (moe.metal). Validate a standalone C[M,N]=X·dequant(W) vs CPU;
   micro-bench vs M× matvec (must show weight-read amortization). LOW risk (proven template).
2. **Batched elementwise**: RMSNorm over M rows, NEOX RoPE over M positions, SwiGLU,
   residual add — all trivially parallel over M×dim. Validate vs the single-token kernels.
3. **Batched causal attention** (prefill): M queries, query m attends keys [0..m].
   One simdgroup per (query, head) with a causal bound. Validate vs sequential attention.
4. **Batched forward driver** `ds4_dense_gpu_prefill(tokens[M])`: embeddings for M tokens
   -> per-layer {rmsnorm, qkv matmul, bias, rope, write KV[0..M), causal attn, o matmul,
   residual, ffn matmuls, residual} -> final norm + output matmul for the LAST token only
   (the only logits prefill needs). Writes the same KV state as the sequential path.
5. **Wire into generate/chat**: replace the per-token prefill loop with one (or a few
   chunked) `ds4_dense_gpu_prefill` calls. Gate behind DS4_DENSE_NOPREFILL for A/B.

## Correctness gate
The batched prefill must leave the KV cache + the last-token logits **identical** (within
fp tolerance) to the sequential path, so greedy decode output stays identical to llama.cpp.
Benchmark prefill t/s before/after at several prompt lengths (57, ~200, ~500 tokens).

## Status
- [x] Baseline recorded.
- [ ] Increment 1 (matmul kernels) — next.
- [ ] Increments 2–5.

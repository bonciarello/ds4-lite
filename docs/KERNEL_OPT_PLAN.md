# Dense K-quant matvec optimization — plan (branch: dense-kernel-opt)

## Target (measured)
Decode reads all weights once/token: 4.7GB / ~400 GB·s (M1 Max) ≈ **12 ms/token floor**.
- llama.cpp: ~22 ms/token (44 t/s) → memory-bound (near-optimal).
- ds4 dense: ~312 ms/token (3.2 t/s) → compute-bound on the **float dequant**.
- Need ~**27x** dequant speedup to reach the floor.

## What did NOT work (all correct, all reverted; verified on this branch)
- Command-buffer batching → not dispatch-bound.
- Simdgroup-per-row (block-strided) → GPU already saturated for high-row matvecs.
- float4 vectorization of the multiply-accumulate → no measurable speedup.
- Coalesced simdgroup (lanes read adjacent qs bytes) → wash (helps attn ~3%, hurts
  ffn gate/up where out_dim=18944 already saturates; +32x threads = overhead).

Conclusion: inner-loop tweaks don't move the needle. The kernels are **structurally**
wrong: they dequantize each weight to float and do float multiplies.

## The real lever: integer dot products (ggml / llama.cpp approach)
Quantize the **activation** x to int8 once per matvec input, then do **int8 × int4
integer MAC** between the q4 weights and the q8 activation, applying scales at the end.
Integer MAC is much faster than float dequant+mul and reads the quants directly.

ds4 already has the CPU reference: `ds4_vec_dot_q4_K_q8_K` (ds4.c ~2786) and
`block_q8_K` (d, int8 qs[256], int16 bsums[16]). The GPU path is to port that.

### Implementation pieces (validate + measure the gap at each)
1. **Quantize-x kernel** `x(f32) -> q8_K`: d = max|x|/127, qs = round(x/d),
   bsums[g] = sum of qs in 16-element groups. Validate: dequant(q8_K) ≈ x.
2. **Int-dot q4_K kernel** (port `ds4_vec_dot_q4_K_q8_K`): per block, unpack the 6-bit
   scales into 8 d-scales + 8 m-scales; for each of 8 groups of 32: int dot of 32 q4
   nibbles × 32 q8 int8 (scaled by d-scale) minus m-scale × q8 bsum; all × d_q4 × d_q8.
   Validate vs the scalar kernel / CPU vec_dot; check greedy still matches llama.cpp.
3. **Wire** into dense_matvec: quantize the matvec input to q8_K once, then dispatch the
   int-dot kernel for q4_K. Measure decode t/s (baseline 3.2 t/s).
4. Repeat for **q6_K** (`ds4_vec_dot` analog) — the ffn_down bottleneck (74 blocks/row).
5. Optional: threadgroup-tiling (share the q8 x across rows) and simdgroup-matrix for
   prefill (batched) speed.

Measurement: `DS4_DENSE_PROFILE=1 ./ds4 --metal-dense-generate ...` for per-phase ms,
`tests/bench_dense.sh` for the two-column ds4-vs-llama.cpp table, and the greedy match
as the correctness gate. Run 3x and take the median (thermal throttling).

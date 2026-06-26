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

## UPDATE: integer-dot (scalar) attempted on this branch — SLOWER, reverted
Implemented `kernel_dense_quantize_q8_K` + `kernel_dense_mul_mv_q4_K_q8_K` (port of
ds4_vec_dot_q4_K_q8_K) and wired q4_K through it. Result: **2.5 t/s vs 3.2 baseline
(slower)** and a small numerical drift (greedy diverged after ~10 tokens). Reverted.

**Why**: the CPU win comes from NEON `vdotq_s32` (a 4-wide int8 dot in one
instruction). Apple GPUs have no equivalent gain for a **scalar** int MAC — integer
and float FMA run at the same rate — so the int path is no faster, and the per-matvec
q8_K quantization adds overhead → net slower.

## ✅✅✅ RESULT (this branch) — 3.14 → ~37 t/s decode (11.8x), gap 14x → ~1.2x

Every step validated (greedy output identical to llama.cpp; `--metal-dense-selftest`
and `--metal-dense-weight-test` PASS) and measured (Qwen2-7B Q4_K_M, M1 Max, prompt
"The capital of France is"):

| step | change | decode t/s | ms/tok | vs llama (44.3) |
|------|--------|-----------:|-------:|----------------:|
| base | scalar 1-thread/row matvec + 28-thread attn + per-op commit/wait + weight copy | 3.14 | 344.6 | 14.1x |
| 1 | ggml simdgroup `mul_mv_q4_K`/`q6_K` (nsg=2,nr0=2, simd_sum) | 3.62 | 291.5 | 12.2x |
| 3 | online-softmax attn, 1 simdgroup/head (`..._sg`) | 4.33 | 250.9 | 10.2x |
| 4 | batch whole token fwd into 1 cmd buffer (barriers, 1 commit/wait) | 15.3 | 65 | 2.9x |
| 5 | parallel RMSNorm (128 threads + tg reduction) vs 1-thread | **~37** | **~27** | **~1.2x** |

The two biggest levers were **not** the matvec kernels:
- **Step 4 (batching):** once 1+3 made per-op compute small, the ~450 per-op
  commit+wait round-trips/token dominated. One command buffer + `MTLBarrierScopeBuffers`
  between dependent dispatches (gated by `g_batch_serialize` → DeepSeek untouched): 3.5x.
- **Step 5 (RMSNorm):** the old `kernel_dense_rms_norm_f32` ran on a *single* GPU
  thread, ×57/token. Parallelizing (simd_sum + threadgroup reduce) gave another 2.4x.

Additional shipped work:
- **Zero-copy weights (default on, opt-out `DS4_DENSE_NO_ZEROCOPY`):** wrap the
  mmap'd weights as shared MTLBuffers (`ds4_gpu_set_model_map_range` +
  `ds4_gpu_tensor_wrap_buffer`) instead of a 4.7GB GPU copy. **TTFT 0.84s → 0.13s
  (6.5x), prefill 6 → 38 t/s**, ~4.5GB RAM saved; decode unchanged.
- **q2/q3/q5_K simdgroup ports:** all five K-quants now have optimized matvec
  kernels (generality for every dense model), validated by selftest Case 14
  (sg-GPU vs CPU dequant on synthetic blocks).
- **Fused gate+up+swiglu (q4_K):** implemented and measured — a **wash** post-batching
  (extra register pressure offsets the saved dispatch), so **reverted**.

Code: `metal/dense.metal` (`kernel_dense_mul_mv_q{2,3,4,5,6}_K_f32_sg`,
`kernel_dense_attn_decode_f32_sg`, `kernel_dense_rms_norm_f32_sg`); `ds4_metal.m`
(`ds4_gpu_run_kquant_sg`, `ds4_gpu_run_grid`, attn-sg dispatch, batched + zero-copy
`ds4_dense_gpu_forward`/`_create`, `ds4_gpu_tensor_wrap_buffer`). A/B toggles:
`DS4_ATTN_SCALAR`, `DS4_DENSE_NOBATCH`, `DS4_RMS_SCALAR`, `DS4_DENSE_NO_ZEROCOPY`,
`DS4_DENSE_SG` (weight-test). Remaining ~5ms gap to llama: many tiny dispatches
(bias/rope/residual) each barriered, plus per-token CPU embedding dequant + logits
readback — diminishing returns.

## Backlog — punti da sviluppare (priorità)

Stato decode: ~37 t/s (gap 1.2x). I limiti restanti, in ordine di impatto:

1. **Prefill batchato (simdgroup-matrix / `kernel_mul_mm`).** IL gap più grande:
   prefill ~38 t/s vs llama 411 t/s (~11x). Oggi il prompt è processato token-per-token
   con matvec; va impacchettato in un matmul (N token → un `mul_mm` su `simdgroup_float8x8`,
   già presente per il path DeepSeek). Impatto: TTFT su prompt lunghi (RAG, codice,
   long-context). È anche il lever che l'analisi iniziale indicava.

2. **Attention per contesti lunghi.** ✅ FATTO (branch perf/longctx-attention).
   Split-KV / flash-decoding: la sequenza KV è divisa in chunk da ~512 chiavi, un
   simdgroup per (testa, split) calcola un softmax online parziale (n_head*S simdgroup
   invece di n_head), poi un kernel di combine fonde i parziali. Kernel
   `kernel_dense_attn_decode_split_f32` + `_combine_f32`; scratch pm/pl/pacc nello
   struct; attivo per n_ctx>512 (S = ceil(n_ctx/512), cap 64). A/B: DS4_ATTN_NOSPLIT.
   Correttezza: greedy identico (md5 match su 540 token). **Decode a ctx ~990 (S=2):
   8.57 -> 14.86 t/s = 1.73x**; il beneficio cresce col contesto (più split). Piccoli
   contesti restano single-simdgroup (nessun overhead di combine).

3. **Gap decode residuo (~5 ms).** Tanti micro-dispatch (bias/rope/residual), ciascuno
   con una barriera, più lavoro CPU per-token (dequant embedding + readback logits 608KB).
   Lavoro: fondere il bias nel matvec, ridurre le barriere dove le op sono indipendenti
   (q/k/v, gate/up), tenere i logits su GPU per l'argmax. Rendimenti decrescenti.

4. **Chat dense — colmare il gap col REPL DeepSeek.** `--metal-dense-chat` oggi è
   minimale (getline + greedy argmax + ctx fisso). Feature da portare, prese dal REPL
   DeepSeek (`repl_chat` in `ds4_cli.c`), in ordine di valore/sforzo:
   - **Reflection mode** ✅ FATTO (sempre attiva, con strip): per i modelli dense la
     reflection è SEMPRE on (niente `/think`/`/nothink`); restano solo `/help` `/exit`.
     Ogni turno inietta una direttiva + prima l'assistant con `<think>\n`; il ragionamento è
     renderizzato in grigio con un filtro a byte che **sopprime i marker `<think>`/`</think>`
     dall'output** (e salta gli spazi iniziali della risposta), poi la risposta.
     Regola lingua: il think può essere in qualsiasi lingua, ma la risposta finale è
     **nella lingua dell'utente** (direttiva "answer in the SAME language as this message").
     La risposta
     finale in chiaro. Implementazione nostra (no token nativi DeepSeek). **Strip della
     history FATTO**: a fine turno `/think` si riavvolge `pos` all'inizio del turno utente
     e si re-immette un turno PULITO (user senza direttiva + risposta senza `<think>`),
     sfruttando il fatto che l'attention legge solo `[0,pos)`. Risultato: `/nothink` resta
     diretto (niente imitazione), il contesto è preservato (risposta tenuta in history) e
     non si gonfia. Validato: ricordo cross-turno OK, selftest/generate non regrediti.
   - **linenoise**: editing della riga + cronologia su file (frecce, modifica). Facile, alto valore.
   - **Temperature / top-k / top-p sampling**: ora è solo greedy/argmax. Banale.
   - **Contesto scorrevole (sliding-window / trim)**: oggi termina con `[context full]`;
     servirebbe far scorrere il KV invece di fermarsi. Il più utile per chat lunghe.
   - **Comandi slash**: `/help`, `/ctx [N]` (resize del contesto a caldo), `/read <file>`
     (carica un file nel prompt), `/exit` (già c'è).
   - **Salvataggio/ripristino sessione** (KV persistente, come `ds4_session`).
   - **Speculative decoding** (draft tokens) per decode più veloce — più complesso, opzionale.
   - NON rilevanti per la dense (specifiche DeepSeek): modalità thinking `/think`,
     `/think-max`, `/nothink`, `/power`, max-effort prefix, routing distribuito.
   Nota: il REPL DeepSeek è legato a `ds4_engine`/`ds4_session`; portarlo sulla dense
   significa o adottare quell'infrastruttura o reimplementare il sottoinsieme generico.

5. **(opzionale) YaRN / context scaling > 32K.** Per superare il contesto nativo Qwen2;
   il path denso oggi non applica scaling RoPE, quindi oltre 32K la qualità crolla.

**Earlier (pre-result) conclusion**: the real lever on Metal is the GPU **simdgroup-matrix units**
(`simdgroup_float8x8` / `simdgroup_multiply_accumulate`), which is how ggml/llama.cpp's
Metal kernels accelerate. ds4 already uses them in `kernel_mul_mm` (dense.metal). The
next attempt should adapt that matrix-multiply structure to the dense decode matvec
(or batch tokens to make it a matmul). This is a substantial, careful rewrite — best
in a fresh session with a cool machine for reliable measurement. Baseline to beat:
**~3.2 t/s decode**, correctness gate = greedy match vs llama-simple.

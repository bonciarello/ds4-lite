# qwen3_next support in ds4 — implementation plan

Status: **✅ WORKING — forward validated greedy-identical to llama.cpp.**
`./ds4 --metal-q3n-generate gguf/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf "The capital of
France is" 12` → " Paris. The capital of Germany is Berlin. The capital of" — byte-identical
to the oracle (`docs/qwen3next_oracle.txt`). The full hybrid forward (36 Gated-DeltaNet +
12 full-attention layers, each + 512-expert MoE) runs on a 32 GB Mac at ~1 tok/s
(correctness-first: quantized matvecs on GPU; DeltaNet recurrence + MoE routing on CPU;
experts copied to GPU on demand). Remaining work is **speed** (move the recurrence + MoE
to the already-validated GPU kernels; SSD-stream experts) and wiring into the /model chat.

The sections below document the architecture + the phased path that got here.

## 1. What qwen3_next is

A **hybrid** decoder, unlike anything ds4 runs today:

- **Per-layer mix**: most layers use **Gated DeltaNet linear attention** (an SSM-style
  recurrence); every `full_attention_interval`-th layer (4) uses **standard full
  attention** (GQA + RoPE — this part ds4 already has).
- **MoE FFN**: `n_expert = 512`, `n_expert_used = 10`, top-k routed, **+ a shared
  expert**. (ds4 has a MoE path, but it is DeepSeek-specific: MLA + indexer + hash
  compression + hyper-connections — a different routing/layout.)
- Reference config (Qwen3-Coder-Next): 48 layers, hidden 2048, 16 attn heads, 2 KV heads,
  `linear_num_value_heads = 32`, `linear_conv_kernel_dim`, `linear_key/value_head_dim`,
  `moe_intermediate_size`, `shared_expert_intermediate_size`.

The Gated DeltaNet layer in llama.cpp is built from three ggml ops:
`ggml_ssm_conv` (short causal 1-D conv over the sequence), `ggml_ssm_scan`
(the selective-scan **recurrence** that carries a per-channel state), and
`ggml_gated_linear_attn` (the gated linear-attention combine). Each has a dedicated
Metal kernel in `ggml/src/ggml-metal/ggml-metal.metal` (`kernel_ssm_conv_*`,
`kernel_ssm_scan*`, the gated-linear-attn kernel) and a CPU reference in `ggml/src/ggml.c`.

## 2. The gap vs ds4 today

| Piece | ds4 today | Needed |
|---|---|---|
| Arch detection / shape | dense + DeepSeek only | + `qwen3next` family + shape reader |
| GGUF tensor loading | dense + DeepSeek tensors | + DeltaNet conv/gate/state, 512 experts, shared expert |
| Full-attention layer (GQA+RoPE) | ✅ (dense path) | reuse |
| **`ssm_conv` Metal kernel** | ❌ | port from llama.cpp |
| **`ssm_scan` Metal kernel** (selective scan recurrence) | ❌ | port — **the hard part** |
| **`gated_linear_attn` Metal kernel** | ❌ | port |
| Gated DeltaNet graph (conv→scan→gate→norm→out) | ❌ | new forward |
| MoE 512-expert routing + shared expert | ⚠️ DeepSeek-only | new/adapted matvec+top-k+combine |
| Hybrid scheduling (linear vs full per layer) | ❌ | new |
| Validation vs reference logits | ds4 standard | reference run + exact-match harness |

## 3. Phased plan

**Phase 0 — skeleton (this change).** Detect `qwen3next`; add `DS4_ARCH_QWEN3NEXT`; read
its hparams into the shape struct (`config_build_qwen3next_shape`); reject it cleanly at
engine open with a "not yet runnable — see this doc" message. No kernels yet. *Goal:
foundation + the model is recognized, not garbage-loaded.*

**Phase 1 — loader + full-attention path.** Map the GGUF tensor names (DeltaNet tensors,
experts, shared expert, full-attn tensors, norms) and bind them. Implement the
full-attention layers (every 4th) by reusing the dense attention kernels, with the
DeltaNet layers stubbed (identity/zeros) so the graph runs end-to-end (wrong output, but
shape-correct and debuggable).

**Phase 2 — MoE FFN.** Implement the 512-expert top-10 router + expert matvec + shared
expert + combine (a generic dense-MoE, distinct from DeepSeek's). Validate the FFN in
isolation against reference activations.

**Phase 3 — Gated DeltaNet (the hard part).** Port the three Metal kernels:
1. `kernel_ssm_conv` — causal depthwise 1-D conv (kernel width `linear_conv_kernel_dim`).
2. `kernel_ssm_scan` — the selective-scan recurrence: per (head, channel) carry a state
   `S`, update `S = a⊙S + k⊗v` style with the delta rule + decay, emit `y = q·S`. Port
   the math from `ggml.c` (`ggml_compute_forward_ssm_scan_f32`) and the Metal kernel.
3. `kernel_gated_linear_attn` — the output gating/combine.
Wire the DeltaNet layer graph: in_proj → conv → scan → gate → RMSNorm → out_proj, with
residuals. **Validate each kernel against ggml's CPU reference first** (tiny inputs),
then the whole layer.

**Phase 4 — integration + validation.** Hybrid per-layer dispatch (linear vs full); KV
cache only for the full-attention layers + the recurrent **state cache** for DeltaNet
layers (the DeltaNet "KV" is the SSM state, sized per layer, not per token — a different
cache shape). End-to-end greedy match vs a reference (llama.cpp) at several positions.

## 4. Notes / risks

- The **state cache** for DeltaNet is the conceptually new memory object: a fixed-size
  recurrent state per layer (not growing with context), updated each token. Decode is
  cheap (O(1) state update); prefill must run the scan over the prompt.
- `ssm_scan` is numerically sensitive (the recurrence); do it in f32. Match ggml's order
  of operations to hit exact logits.
- gpt-oss-20b and Qwen3-30B-A3B are **plain MoE** (no linear attention) — Phase 2's
  generic MoE would also unlock those on the dense path, a useful intermediate win.

## 5. Precise reference map (from llama.cpp @ ~/Documents/GitHub/llama.cpp)

**hparams** (`src/llama-hparams.h:155`, `src/models/qwen3next.cpp:4-62`):
`ssm_d_conv` (conv width, 4), `ssm_d_inner`, `ssm_d_state`, `ssm_dt_rank` (= n_value_heads),
`ssm_n_group` (= n_key_heads), `n_ff_exp`, `n_ff_shexp`, `f_norm_rms_eps`,
`full_attention_interval` (4; layer is full-attn when `i % interval == interval-1`, else
DeltaNet — `hparams.is_recr(i)`). Derived: `head_k_dim = head_v_dim = ssm_d_state`,
`key_dim = head_k_dim*n_k_heads`, `value_dim = head_v_dim*n_v_heads`,
`conv_dim = key_dim*2 + value_dim`, `ba_dim = n_v_heads*2`.

**GGUF tensors per layer** (`src/llama-arch.cpp:411`, `src/models/qwen3next.cpp:38`):
- DeltaNet layer: `attn_norm`, `attn_qkv` `[n_embd, key_dim*2+value_dim]`, `attn_gate`
  `[n_embd, value_dim]`, `ssm_conv1d` `[d_conv, conv_dim]`, `ssm_dt.bias` `[dt_rank]`,
  `ssm_a` `[dt_rank]`, `ssm_ba` `[n_embd, ba_dim]`, `ssm_norm` `[head_v_dim]`,
  `ssm_out` `[value_dim, n_embd]`, `attn_post_norm`.
- Full-attn layer: `attn_norm`, `attn_q` `[n_embd, head_k*n_head*2]` (q **+ gate**),
  `attn_k`, `attn_v`, `attn_q_norm`, `attn_k_norm`, `attn_output`, `attn_post_norm`.
- MoE (all layers): `ffn_gate_inp` `[n_embd,n_expert]`, `ffn_{gate,up,down}_exps`
  `[…,n_expert]` (or fused `ffn_gate_up_exps`), `ffn_gate_inp_shexp`,
  `ffn_{gate,up,down}_shexp` (shared expert).

**DeltaNet layer graph** (`qwen3next.cpp:367-534`): in_proj(attn_qkv)+gate(z) → ba_proj
(`ssm_ba`)→ split β=sigmoid, α; `α' = softplus(α + ssm_dt)·ssm_a` → conv1d(`ssm_conv1d`,
prepend conv_state)+SiLU → split q,k,v → L2-norm(q),L2-norm(k), repeat-interleave to
n_v_heads → `ggml_gated_linear_attn(q,v,k,gate=α',β,state)` → RMSNorm(`ssm_norm`)·SiLU(z)
→ out_proj(`ssm_out`). Two caches: **conv_state** `[d_conv-1, conv_dim, n_seqs]` and
**ssm_state** `[head_v_dim, head_v_dim, n_v_heads, n_seqs]`.

**Full-attn layer** (`qwen3next.cpp:206-284`): q-proj gives [q | gate]; RMSNorm q,k; RoPE
(NeoX); softmax attention; `out *= sigmoid(gate)`; out-proj.

**DeltaNet recurrence — CORRECTED (important).** qwen3_next does **not** use a fused
`ggml_ssm_scan` or `ggml_gated_linear_attn` op. `src/models/qwen3next.cpp` calls only
`ggml_ssm_conv` (line 443); the whole delta-rule lives in `src/models/delta-net-base.cpp`
(606 lines) as the **chunkwise gated delta rule** built from *standard* ops:
`ggml_mul_mat`, `ggml_cumsum` (prefix-sum along the chunk), `ggml_pad`, `ggml_mul`,
`ggml_transpose`, `ggml_cont`, `ggml_view`/`reshape`, plus a triangular mask. So the only
**exotic kernels** ds4 lacks are:
1. **`ssm_conv`** — causal depthwise 1-D conv. ✅ **landed + validated** (see §7).
2. **`cumsum`** — inclusive prefix-sum along a chunk axis. ✅ **landed + validated** (§7).
Everything else (the chunked matmuls, masking, gating) is expressible with ds4's existing
dense matvec/elementwise kernels. This is **more tractable** than the original
"port a selective-scan recurrence" framing — no numerically-delicate sequential scan kernel.

**llama.cpp files**: `src/models/qwen3next.cpp`, `src/models/delta-net-base.cpp`,
`src/llama-arch.cpp`, `src/llama-hparams.{h,cpp}`, `ggml/src/ggml.c`,
`ggml/src/ggml-metal/ggml-metal.metal`.

## 6. Skeleton landed (Phase 0)
- `DS4_ARCH_QWEN3NEXT` family + `ds4_arch_is_qwen3next()`.
- `config_build_qwen3next_shape()` reads block_count / embedding_length / head counts /
  expert_count / context_length under the `qwen3next` namespace and sets the family.
- `config_validate_model` dispatches `general.architecture == "qwen3next"` to it.
- The dense chat/generate entry points reject it cleanly with a pointer to this doc
  (instead of crashing or loading garbage).

## 7. Kernel work landed (Phase 3, partial)
Both exotic kernels qwen3_next needs are now in `metal/dense.metal`, each validated
on-device (M1 Max) against a CPU reference in `./ds4 --metal-dense-selftest` (PASS):
- **`kernel_dense_ssm_conv_f32`** (Case 14) — causal depthwise 1-D conv
  (`out[t,c] = Σ_k sx[t+k,c]·w[k,c]`, history prepended → causal window). max err < 1e-5.
- **`kernel_dense_cumsum_f32`** (Case 15) — inclusive prefix-sum along the chunk axis
  (`out[r,i] = Σ_{j<=i} in[r,j]`). max err < 1e-5.

Both reuse ds4's standard dispatch path (buffers/pipeline/encode), so they slot directly
into the eventual DeltaNet graph. Everything else the delta rule needs (chunked matmuls,
triangular mask, gating) is already expressible with ds4's dense matvec/elementwise kernels.

**Remaining for a runnable model** (all needs the model GGUF to validate end-to-end):
wire the chunked delta-rule graph; 512-expert MoE + shared expert; hybrid per-layer
dispatch (linear vs full attention every 4th); the recurrent conv/state **cache**
(fixed-size per layer, not per-token); greedy match vs llama.cpp.

## 7b. GROUND TRUTH — real model dims (Qwen3-Next-80B-A3B-Instruct Q4_K_M)

Dumped from the actual GGUF (843 tensors, 41 KV). This is the authoritative spec for the
forward; supersedes the earlier derivations.

**hparams**: block_count 48, embedding_length 2048, head_count 16, head_count_kv 2,
key_length = value_length = 256, expert_count 512, expert_used_count 10,
expert_feed_forward_length 512, expert_shared_feed_forward_length 512,
feed_forward_length 5120, rope.dimension_count 64, rope.freq_base **1e7**,
rms_eps 1e-6, context_length 262144. ssm: conv_kernel 4, group_count 16, inner_size 4096,
state_size 128, time_step_rank 32. Layer i is **full-attn iff i % 4 == 3** (12 layers:
3,7,…,47), else **DeltaNet** (36 layers).

Derived DeltaNet dims: value_dim 4096, n_v_heads 32, head_v_dim 128 (= state_size);
key_dim 2048, n_k_heads 16 (= group_count), head_k_dim 128; conv_dim 8192 (= key_dim·2 +
value_dim); ba_dim 64 (= n_v_heads·2); dt_rank 32 (= n_v_heads).

**DeltaNet layer tensors** (blk.0): `attn_norm`[2048] F32; `attn_qkv`[2048,8192] Q5_K
(= q2048|k2048|v4096); `attn_gate`[2048,4096] Q4_K (z gate); `ssm_conv1d`[4,8192] F32
(causal conv over q|k|v); `ssm_ba`[2048,64] Q4_K (β|α); `ssm_dt.bias`[32] F32; `ssm_a`[32]
F32; `ssm_norm`[128] F32 (RMSNorm over head_v_dim); `ssm_out`[4096,2048] Q4_K;
`post_attention_norm`[2048] F32.

**Full-attn layer tensors** (blk.3): `attn_norm`[2048]; `attn_q`[2048,8192] Q4_K
(= q4096|gate4096); `attn_k`[2048,512] Q4_K; `attn_v`[2048,512] Q6_K; `attn_q_norm`[256]
F32; `attn_k_norm`[256] F32 (per-head RMSNorm, head_dim 256); `attn_output`[4096,2048] Q4_K;
`post_attention_norm`[2048]. (q output is [q | gate]; final out *= sigmoid(gate).)

**MoE tensors** (every layer): `ffn_gate_inp`[2048,512] F32 (router); `ffn_gate_exps`
[2048,512,512] Q4_K, `ffn_up_exps`[2048,512,512] Q4_K, `ffn_down_exps`[512,2048,512] Q6_K
(512 experts, hidden 512); shared expert `ffn_gate_inp_shexp`[2048] F16,
`ffn_gate_shexp`[2048,512] Q4_K, `ffn_up_shexp`[2048,512] Q4_K, `ffn_down_shexp`[512,2048]
Q6_K. **Global**: `token_embd`[2048,151936] Q4_K, `output_norm`[2048] F32,
`output`[2048,151936] Q6_K.

Forward order per layer: norm → (DeltaNet | full-attn) → residual →
post_attention_norm → MoE (router top-10 experts + shared expert) → residual.

## 8. Running 80B on 32 GB — SSD streaming (the hardware unblock)

qwen3_next only ships as **80B-A3B** (Q4_K_M ≈ 45 GB), which does not fit 32 GB RAM.
But it is **80B total / ~3B active** — only 10 of 512 experts fire per token. That is
exactly the sparse profile ds4's **DeepSeek SSD weight-streaming** was built for: experts
live on SSD, and each token `pread`s only the routed experts into reusable GPU staging
slabs (LRU, 16 slabs), overlapping I/O with compute.

A subagent mapped the mechanism: it is **agnostic to expert count/distribution** — it just
needs per-expert absolute byte offsets + routing that emits selected expert IDs/weights.
The only DeepSeek coupling is the **router path** (compute routing on GPU → read back IDs →
`pread`), which is adaptable to qwen3_next's plain top-10 + shared expert. So the streaming
path is **reusable**; the work is: produce the qwen3_next expert offset table, swap in its
router, and widen the 3 streaming gates (currently `ds4_arch_is_deepseek()`) to include it.
Target model: single-file `lmstudio-community/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf`
(48.5 GB, loadable by ds4's single-file loader). Decode is I/O-bound (~few tok/s) but it
**runs on this 32 GB machine** — which makes the full implementation worth doing.

## 9. Shape reader extended (Phase 1a)
`config_build_qwen3next_shape` now also reads the Gated-DeltaNet + MoE hparams into the
shape: `ssm.{conv_kernel,inner_size,state_size,time_step_rank,group_count}`,
`full_attention_interval` (default 4), `expert_feed_forward_length` (+ shared-expert FFN),
`attention.layer_norm_rms_epsilon`, `rope.freq_base`, and `n_head_dim = n_embd/n_head`.
New `ds4_shape` fields: `ssm_conv_kernel/ssm_inner_size/ssm_state_size/ssm_dt_rank/
ssm_n_group/full_attn_interval`. This is what the forward + the streaming router will read.

## 10. Tensor binding landed (Phase 1) — VALIDATED on the real model
`weights_bind_qwen3next` + `weights_bind_layer_qwen3next` map every GGUF tensor into the
fixed layer layout, branching per layer type via `ds4_q3n_layer_is_full_attn(il)`
(`il % interval == interval-1`):
- DeltaNet layers → new `ds4_layer_weights` fields `q3n_in_proj` (attn_qkv), `q3n_gate`
  (attn_gate), `q3n_conv1d`, `q3n_ba`, `q3n_dt_bias`, `q3n_a`, `q3n_ssm_norm`, `q3n_out_proj`.
- full-attn layers → reuse `attn_q/k/v/out` (+ new `attn_q_norm`, `attn_k_norm`).
- MoE (all layers) → reuse `ffn_gate_inp` + `ffn_{gate,up,down}_exps` + `*_shexp`, plus new
  `ffn_gate_inp_shexp`; `attn_norm` = input norm, `ffn_norm` = `post_attention_norm`.
`weights_bind` routes `qwen3next` here. **Validated**: binding the real 48-layer/512-expert
Q4_K_M model reports "weights bound OK — 48 layers (12 full-attn, 36 DeltaNet), 512 experts
(top-10) + shared" with no missing tensor (all 843 mapped). The chat entry binds + reports,
then still rejects (forward pending). **Next**: Phase 2/3 forward — full-attn layer (reuse
dense GQA + q/k-norm + sigmoid gate), DeltaNet (conv→delta-rule→norm→gate→out), MoE over
streaming, hybrid dispatch + state cache, validate vs llama.cpp.

## 11. DeltaNet decode = simple recurrence (KEY simplification) + more kernels landed
Studying `delta-net-base.cpp`: the **chunked** path (prefill) uses heavy ops (`ggml_tri`,
`ggml_exp`, **`ggml_solve_tri`** — a triangular solve, hard to port). BUT there is an
**autoregressive path** (`build_delta_net_autoregressive`, n_tokens==1) — the recurrent
form ds4's token-by-token decode needs, which avoids tri/solve_tri/cumsum entirely. Per
head, state S is [S_k × S_v] (here 128×128); for one token with scalar gate g (GDA) + beta:
`S *= exp(g)` → `sk = Sᵀk` → `d = β(v−sk)` → `S += k⊗d` → `o = Sᵀq` (q pre-scaled 1/√S_k).
So DeltaNet decode is **one kernel**, not a chunked machine. Prefill can run the same
recurrence token-by-token (correct, just sequential) — chunked/solve_tri only needed later
for fast prefill.

Validated kernels now in `metal/dense.metal` (all PASS in `--metal-dense-selftest`, <1e-5):
ssm_conv (14), cumsum (15), **head RMSNorm (16)** + **sigmoid gate (17)** (full-attn layer),
**`kernel_dnet_ar_step_f32` (18)** — the gated-delta recurrence (validates output AND the
updated state). One thread per (head, state column) → no cross-thread deps.

**Oracle** (`docs/qwen3next_oracle.txt`): llama.cpp greedy on the real model,
"The capital of France is" → " Paris. The capital of Germany is Berlin. The capital of"
(12 tokens). The ds4 forward must reproduce this.

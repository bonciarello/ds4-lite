# qwen3_next support in ds4 — implementation plan

Status: **planning + skeleton**. This documents how to bring the Qwen3-Next architecture
(`general.architecture = "qwen3next"`, e.g. *Qwen3-Coder-Next*) to the ds4 Metal engine.
A full, validated implementation is a multi-week effort; this plan breaks it into phases
so the hard parts are isolated.

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

**Remaining for a runnable model** (all needs the ~38 GB Qwen3-Coder-Next GGUF to validate
end-to-end, so deferred): wire the chunked delta-rule graph; 512-expert MoE + shared
expert; hybrid per-layer dispatch (linear vs full attention every 4th); the recurrent
conv/state **cache** (fixed-size per layer, not per-token); greedy match vs llama.cpp.

#ifndef DS4_H
#define DS4_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "ds4_ssd.h"

/* Public engine boundary.
 *
 * The CLI and server should treat ds4_engine as the loaded model and
 * ds4_session as one mutable inference timeline.  A session owns the live KV
 * cache and logits; callers provide full token prefixes and let
 * ds4_session_sync() reuse, extend, or rebuild the graph state.  Keep this
 * header narrow so HTTP/CLI code does not depend on tensor internals. */

typedef enum {
    DS4_BACKEND_METAL,
    DS4_BACKEND_CUDA,
    DS4_BACKEND_CPU,
} ds4_backend;

typedef enum {
    DS4_THINK_NONE,
    DS4_THINK_HIGH,
    DS4_THINK_MAX,
} ds4_think_mode;

typedef enum {
    DS4_LOG_DEFAULT,
    DS4_LOG_PREFILL,
    DS4_LOG_GENERATION,
    DS4_LOG_KVCACHE,
    DS4_LOG_TOOL,
    DS4_LOG_WARNING,
    DS4_LOG_TIMING,
    DS4_LOG_OK,
    DS4_LOG_ERROR,
} ds4_log_type;

typedef struct {
    int *v;
    int len;
    int cap;
} ds4_tokens;

typedef struct {
    int id;
    float logit;
    float logprob;
} ds4_token_score;

#define DS4_DEFAULT_TEMPERATURE 1.0f
#define DS4_DEFAULT_TOP_P 1.0f
#define DS4_DEFAULT_MIN_P 0.05f

typedef struct ds4_engine ds4_engine;
typedef struct ds4_session ds4_session;

typedef void (*ds4_session_progress_fn)(void *ud, const char *event, int current, int total);
typedef bool (*ds4_session_cancel_fn)(void *ud);

#define DS4_SESSION_SYNC_INTERRUPTED 2

typedef enum {
    DS4_DISTRIBUTED_NONE = 0,
    DS4_DISTRIBUTED_COORDINATOR,
    DS4_DISTRIBUTED_WORKER,
} ds4_distributed_role;

typedef struct {
    uint32_t start;
    uint32_t end;
    bool has_output;
    bool set;
} ds4_distributed_layers;

typedef struct {
    ds4_distributed_role role;
    ds4_distributed_layers layers;
    const char *listen_host;
    int listen_port;
    const char *coordinator_host;
    int coordinator_port;
    uint32_t prefill_chunk;
    uint32_t prefill_window;
    uint32_t activation_bits;
    bool replay_check;
    bool debug;
} ds4_distributed_options;

typedef struct {
    const char *model_path;
    const char *mtp_path;
    ds4_backend backend;
    int n_threads;
    uint32_t prefill_chunk;
    int mtp_draft_tokens;
    float mtp_margin;
    const char *directional_steering_file;
    const char *expert_profile_path;
    float directional_steering_attn;
    float directional_steering_ffn;
    int power_percent;
    uint32_t ssd_streaming_cache_experts;
    uint64_t ssd_streaming_cache_bytes;
    uint32_t ssd_streaming_preload_experts;
    uint64_t simulate_used_memory_bytes;
    bool warm_weights;
    bool quality;
    bool ssd_streaming;
    bool ssd_streaming_cold;
    bool inspect_only;
    bool load_slice;
    uint32_t load_layer_start;
    uint32_t load_layer_end;
    bool load_output;
    ds4_distributed_options distributed;
} ds4_engine_options;

typedef void (*ds4_token_emit_fn)(void *ud, int token);
typedef void (*ds4_generation_done_fn)(void *ud);

typedef struct {
    uint64_t total_bytes;
    uint64_t raw_bytes;
    uint64_t compressed_bytes;
    uint64_t scratch_bytes;
    uint32_t prefill_cap;
    uint32_t raw_cap;
    uint32_t comp_cap;
} ds4_context_memory;

typedef struct {
    uint8_t *ptr;
    uint64_t len;
    uint64_t cap;
} ds4_session_snapshot;

typedef struct {
    char *path;
    uint64_t bytes;
} ds4_session_payload_file;

int ds4_engine_open(ds4_engine **out, const ds4_engine_options *opt);
void ds4_engine_close(ds4_engine *e);

/* Fase 3.5: GPU dense matvec smoke test. Validates the dense matvec dispatch
 * path on-device with no model. Returns 0 on PASS. Implemented on the GPU
 * backend; a CPU-only build returns an error. */
int ds4_gpu_dense_matvec_selftest(char *err, size_t errlen);

/* Batched-prefill q4_K matmul: correctness (vs CPU) + amortization micro-benchmark.
 * n_tok is the batch size M (default 32). Returns 0 on PASS. */
int ds4_gpu_dense_matmul_test(int n_tok, char *err, size_t errlen);

/* Fase 3.5 step 4: verify the type-dispatch dense matvec on a real weight (copies
 * the quantized weight to GPU, dispatches the kernel for its GGML type, compares
 * to the CPU dequant). type = GGML id; in_dim multiple of 256. Returns 0 on ok. */
int ds4_gpu_dense_matvec_verify(int type, const void *wbytes, uint64_t wnbytes,
                                const float *x, uint32_t in_dim, uint32_t out_dim,
                                float *maxerr_out);

/* Load a dense model and verify the dense GPU matvec on real weight tensors. */
int ds4_dense_weight_test(const char *model_path, char *err, size_t errlen);

/* ---- Dense GPU forward (Fase 3.5 step 4) -----------------------------------
 * Plain-old-data descriptors so the Metal forward (which cannot see the ds4.c
 * model/weights structs) gets the weight bytes (mmap pointers), GGML types and
 * dims it needs. data==NULL means the weight is absent (e.g. optional bias). */
typedef struct {
    int          type;   /* GGML type id (0=f32, 12=q4_k, 14=q6_k, ...) */
    const void  *data;   /* pointer into the mmapped model (tensor_data) */
    unsigned long long bytes;
    unsigned     dim0;   /* in_dim for matvec, or length for 1D */
    unsigned     dim1;   /* out_dim for matvec, 0 for 1D */
} ds4_dense_wdesc;

typedef struct {
    ds4_dense_wdesc attn_norm, attn_q, attn_q_bias, attn_k, attn_k_bias,
                    attn_v, attn_v_bias, attn_out;
    ds4_dense_wdesc ffn_norm, ffn_gate, ffn_up, ffn_down;
    /* gemma3 only (NULL data otherwise): QK-norm [head_dim], post-attn & post-ffn norms.
     * attn_q_norm/attn_k_norm are also used by qwen3 (per-head QK-norm). */
    ds4_dense_wdesc attn_q_norm, attn_k_norm, attn_post_norm, ffn_post_norm;
    /* MoE FFN (NULL data on a dense model): router + per-expert gate/up/down (3D [.., n_expert]). */
    ds4_dense_wdesc ffn_gate_inp, ffn_gate_exps, ffn_up_exps, ffn_down_exps;
    /* LayerNorm bias (beta) for the input/ffn norms — NULL on RMSNorm models. */
    ds4_dense_wdesc attn_norm_bias, ffn_norm_bias;
    /* phi-2: bias on attn_output + the FFN up/down projections (NULL when absent). */
    ds4_dense_wdesc attn_out_bias, ffn_up_bias, ffn_down_bias;
    /* gemma4: heterogeneous per-layer attention dims (0 -> use the model-wide d->head_dim/n_kv/n_rot).
     * SWA layers: head_dim 256 / 16 KV heads / rope base 1e4; FULL: 512 / 4 / 1e6 + rope_freqs. */
    unsigned head_dim, n_kv, n_rot;     /* per-layer; 0 = fall back to the uniform model values */
    float    rope_base_layer;           /* per-layer rope base (gemma4 swa 1e4 / full 1e6); 0 = use desc */
    float    out_scale;                 /* gemma4 layer_output_scale scalar (cur *= out_scale); 0 = none */
    ds4_dense_wdesc rope_freqs;         /* gemma4 full-attn proportional-rope freq factors [n_rot/2]; NULL otherwise */
} ds4_dense_layer_desc;

typedef struct {
    unsigned n_layer, n_embd, n_ff, n_head, n_kv, head_dim, n_vocab, n_rot, n_ctx;
    float    rms_eps, rope_base;
    ds4_dense_wdesc       token_embd, output_norm, output;
    ds4_dense_layer_desc *layers;   /* [n_layer] */
    const void         *model_base; /* mmap base (for zero-copy weight wrapping) */
    unsigned long long  model_size; /* mmapped model size in bytes */
    /* gemma3 extensions (gemma==0 -> plain dense path, all fields below ignored). */
    int      gemma;             /* 1 when the gemma3 forward variant should run */
    int      gemma4;            /* 1 for gemma4: per-layer head dims, V-norm, layer out_scale, rope freq_factors */
    float    embed_scale;       /* token embeddings scaled by this (sqrt(n_embd)) */
    float    attn_scale;        /* attention softmax scale; 0 -> default 1/sqrt(head_dim) */
    float    rope_scale;        /* global-layer RoPE linear freq_scale (gemma: 0.125) */
    float    rope_base_local;   /* sliding-layer RoPE base (gemma: 10000) */
    float    rope_scale_local;  /* sliding-layer RoPE freq_scale (gemma: 1.0) */
    unsigned swa_window;        /* sliding-window size (gemma: 1024); 0 -> full attention */
    unsigned swa_pattern;       /* 1 global every swa_pattern layers (gemma3: 6, gemma2: 2) */
    float    attn_softcap;      /* gemma-2 attn_logit_softcapping (=50); 0 -> off */
    float    final_softcap;     /* gemma-2 final_logit_softcapping (=30); 0 -> off */
    /* MoE FFN (n_expert==0 -> plain dense FFN). qwen3moe etc.: router top-k of n_expert experts. */
    unsigned n_expert;          /* total experts; 0 = dense */
    unsigned n_expert_used;     /* experts routed per token (top-k) */
    unsigned n_ff_exp;          /* per-expert FFN hidden dim */
    float    expert_weights_scale; /* scale applied to the routed weights (qwen3moe: usually 1) */
    int      moe_gating;        /* routing: 0 = softmax-over-all then top-k then renorm (qwen3moe);
                                 *          1 = top-k by logit then softmax over the selected (gpt-oss) */
    /* Capability flags for non-llama-style dense archs (all 0 = plain llama). */
    int      ffn_geglu;         /* FFN activation: 1 = GeGLU (gelu), 0 = SwiGLU (silu) */
    int      norm_layernorm;    /* norm type: 1 = LayerNorm (mean+var+gamma+beta), 0 = RMSNorm */
    int      parallel_residual; /* 1 = attn + ffn both read the input norm, summed (phi-2, GPT-NeoX) */
    int      alibi;             /* 1 = ALiBi positional bias (no RoPE): bloom/mpt/falcon */
    ds4_dense_wdesc output_norm_bias;  /* LayerNorm beta for the final norm (NULL on RMSNorm) */
    ds4_dense_wdesc tok_norm;      /* post-embedding LayerNorm weight (bloom word_embeddings_layernorm); NULL if absent */
    ds4_dense_wdesc tok_norm_bias; /* post-embedding LayerNorm beta */
} ds4_dense_model_desc;

typedef struct ds4_dense_gpu ds4_dense_gpu;
/* Allocate scratch buffers + dense GPU KV cache for the given shape. */
ds4_dense_gpu *ds4_dense_gpu_create(const ds4_dense_model_desc *desc);
void ds4_dense_gpu_free(ds4_dense_gpu *g);
/* One-token forward at position pos; writes desc->n_vocab logits. 0 on success. */
int ds4_dense_gpu_forward(ds4_dense_gpu *g, const ds4_dense_model_desc *desc,
                          int token, unsigned pos, float *logits);

/* Batched prefill: process M tokens at positions [start_pos, start_pos+M) in one
 * forward (matmuls), write the KV cache, and return the last token's logits. If
 * all_logits != NULL it is filled with [M, n_vocab] logits for every position
 * (used by speculative verification); last_logits may be NULL then. */
int ds4_dense_gpu_prefill(ds4_dense_gpu *g, const ds4_dense_model_desc *desc,
                          const int *tokens, unsigned M, unsigned start_pos,
                          float *last_logits, float *all_logits);

/* Load a dense model and greedily generate n_predict tokens (self-contained). */
int ds4_dense_generate(const char *model_path, const char *prompt, int n_predict,
                       char *err, size_t errlen);

/* ---- qwen3_next forward (hybrid Gated-DeltaNet + 512-expert MoE) ----------- *
 * Each layer is full-attention (is_full_attn) or Gated-DeltaNet; MoE FFN on every
 * layer. Weights are mmap pointers (ds4_dense_wdesc). The expert tensors are the full
 * [n_embd, n_ff_exp, n_expert] blocks; the driver slices the active experts per token. */
typedef struct {
    int is_full_attn;
    ds4_dense_wdesc attn_norm, post_attn_norm;
    /* full-attention layer: attn_q packs [query|gate] interleaved per head */
    ds4_dense_wdesc attn_q, attn_k, attn_v, attn_out, attn_q_norm, attn_k_norm;
    /* Gated-DeltaNet layer */
    ds4_dense_wdesc dn_in_proj, dn_gate, dn_conv1d, dn_ba, dn_dt_bias, dn_a, dn_ssm_norm, dn_out_proj;
    /* MoE (every layer): router, routed experts (3D), shared expert */
    ds4_dense_wdesc ffn_gate_inp, ffn_gate_exps, ffn_up_exps, ffn_down_exps;
    ds4_dense_wdesc ffn_gate_inp_shexp, ffn_gate_shexp, ffn_up_shexp, ffn_down_shexp;
} ds4_q3n_layer_desc;

typedef struct {
    unsigned n_layer, n_embd, n_vocab, n_ctx;
    unsigned n_head, n_kv, head_dim, n_rot;        /* full-attention */
    unsigned n_expert, n_expert_used, n_ff_exp;    /* MoE */
    /* DeltaNet dims: key/value heads + per-head dims, conv width + packed conv dim */
    unsigned dn_n_k_heads, dn_n_v_heads, dn_head_k, dn_head_v;
    unsigned dn_conv_kernel, dn_conv_dim, dn_key_dim, dn_value_dim;
    unsigned full_attn_interval;
    float    rms_eps, rope_base;
    ds4_dense_wdesc       token_embd, output_norm, output;
    ds4_q3n_layer_desc   *layers;     /* [n_layer] */
    const void           *model_base; /* mmap base */
    unsigned long long    model_size;
    int                   model_fd;   /* open model fd (for pread-streaming experts; -1 if none) */
} ds4_q3n_model_desc;

typedef struct {
    ds4_dense_wdesc attn_norm, post_attn_norm;                 /* pre-attn + pre-MoE RMSNorms */
    ds4_dense_wdesc attn_q, attn_q_bias, attn_k, attn_k_bias,  /* GQA, all with bias */
                    attn_v, attn_v_bias, attn_out, attn_out_bias;
    ds4_dense_wdesc attn_sinks;                                /* per-head sink logits [n_head] */
    ds4_dense_wdesc ffn_gate_inp, ffn_gate_inp_bias;           /* MoE router (+bias) */
    ds4_dense_wdesc ffn_gate_exps, ffn_gate_exps_bias,         /* top-4/32 experts (3D) + biases */
                    ffn_up_exps, ffn_up_exps_bias,
                    ffn_down_exps, ffn_down_exps_bias;
} ds4_gptoss_layer_desc;

typedef struct {
    unsigned n_layer, n_embd, n_vocab, n_ctx;
    unsigned n_head, n_kv, head_dim, n_rot;
    unsigned n_expert, n_expert_used, n_ff_exp;
    unsigned n_swa;                  /* sliding-window size (128); alternates with full per layer */
    float    rms_eps, rope_base;
    /* YaRN rope (gpt-oss): ext_factor=1, attn_factor=1, beta_fast=32, beta_slow=1. */
    float    yarn_freq_scale;        /* 1/scaling_factor (1/32) */
    float    yarn_mscale;            /* attn_factor*(1+0.1*ln(1/freq_scale)) ~ 1.347 */
    float    yarn_corr0, yarn_corr1; /* rope_yarn ramp correction dims (precomputed) */
    ds4_dense_wdesc        token_embd, output_norm, output;
    ds4_gptoss_layer_desc *layers;   /* [n_layer] */
    const void            *model_base;
    unsigned long long     model_size;
    int                    model_fd;
} ds4_gptoss_model_desc;

typedef struct ds4_gptoss_gpu ds4_gptoss_gpu;
ds4_gptoss_gpu *ds4_gptoss_gpu_create(const ds4_gptoss_model_desc *desc);
void ds4_gptoss_gpu_free(ds4_gptoss_gpu *g);
int ds4_gptoss_gpu_forward(ds4_gptoss_gpu *g, const ds4_gptoss_model_desc *desc,
                           int token, unsigned pos, float *logits);

typedef struct ds4_q3n_gpu ds4_q3n_gpu;
ds4_q3n_gpu *ds4_q3n_gpu_create(const ds4_q3n_model_desc *desc);
void ds4_q3n_gpu_free(ds4_q3n_gpu *g);
/* One-token forward at position pos; writes desc->n_vocab logits. 0 on success. */
int ds4_q3n_gpu_forward(ds4_q3n_gpu *g, const ds4_q3n_model_desc *desc,
                        int token, unsigned pos, float *logits);

/* ---- Mamba (state-space model; no attention) ----------------------------- *
 * Each block: rmsnorm -> in_proj -> [x|z]; causal conv1d + SiLU on x; x_proj ->
 * [dt|B|C]; dt_proj + softplus; selective scan h = exp(dt*A)*h + dt*B*x,
 * y = C·h + D*x; gate y *= silu(z); out_proj. Recurrent state per layer (conv
 * window + ssm state), no KV cache. Projections (Q8_0) run on GPU; the small
 * F32 conv/scan/gate run on CPU. */
typedef struct {
    ds4_dense_wdesc attn_norm;                       /* pre-block RMSNorm (F32) */
    ds4_dense_wdesc ssm_in, ssm_x, ssm_dt, ssm_out;  /* projections (Q8_0) */
    ds4_dense_wdesc ssm_conv1d, ssm_conv1d_bias;     /* causal conv weight [d_conv,d_inner] + bias (F32) */
    ds4_dense_wdesc ssm_a, ssm_d, ssm_dt_bias;       /* A_log [d_state,d_inner], D [d_inner], dt bias (F32) */
} ds4_mamba_layer_desc;

typedef struct {
    unsigned n_layer, n_embd, n_vocab;
    unsigned d_inner, d_conv, d_state, dt_rank;
    float    rms_eps;
    ds4_dense_wdesc       token_embd, output_norm, output;
    ds4_mamba_layer_desc *layers;   /* [n_layer] */
} ds4_mamba_model_desc;

typedef struct ds4_mamba_gpu ds4_mamba_gpu;
ds4_mamba_gpu *ds4_mamba_gpu_create(const ds4_mamba_model_desc *desc);
void ds4_mamba_gpu_free(ds4_mamba_gpu *g);
int ds4_mamba_gpu_forward(ds4_mamba_gpu *g, const ds4_mamba_model_desc *desc,
                          int token, unsigned pos, float *logits);
int ds4_mamba_generate(const char *model_path, const char *prompt, int n_predict,
                       char *err, size_t errlen);

/* ---- RWKV-7 (Goose): linear-attention delta-rule recurrence + channel mix ---- */
typedef struct {
    ds4_dense_wdesc attn_norm, attn_norm_b, attn_norm_2, attn_norm_2_b;   /* LayerNorms (time / channel) */
    ds4_dense_wdesc lerp_fused;                                           /* [n_embd, 6] token-shift lerps */
    ds4_dense_wdesc w0, w1, w2;          /* decay LoRA + bias */
    ds4_dense_wdesc a0, a1, a2;          /* in-context learning rate LoRA + bias */
    ds4_dense_wdesc v0, v1, v2;          /* value-residual mix LoRA + bias */
    ds4_dense_wdesc g1, g2;              /* gate LoRA */
    ds4_dense_wdesc k_k, k_a, r_k;       /* per-channel scalars */
    ds4_dense_wdesc key, value, receptance, output;   /* projections (Q8_0) */
    ds4_dense_wdesc ln, ln_b;            /* group-norm on the WKV output */
    ds4_dense_wdesc cm_lerp_k, cm_key, cm_value;      /* channel mix */
} ds4_rwkv7_layer_desc;

typedef struct {
    unsigned n_layer, n_embd, n_vocab, head_size, n_head, n_ff;
    float    eps;
    ds4_dense_wdesc       token_embd, tok_norm, tok_norm_b, output_norm, output_norm_b, output;
    ds4_rwkv7_layer_desc *layers;   /* [n_layer] */
} ds4_rwkv7_model_desc;

typedef struct ds4_rwkv7_gpu ds4_rwkv7_gpu;
ds4_rwkv7_gpu *ds4_rwkv7_gpu_create(const ds4_rwkv7_model_desc *desc);
void ds4_rwkv7_gpu_free(ds4_rwkv7_gpu *g);
int ds4_rwkv7_gpu_forward(ds4_rwkv7_gpu *g, const ds4_rwkv7_model_desc *desc,
                          int token, unsigned pos, float *logits);
int ds4_rwkv7_generate(const char *model_path, const char *prompt, int n_predict,
                       char *err, size_t errlen);

/* ---- BERT (encoder-only, bidirectional) — embedding model, no LM head ------- */
typedef struct {
    ds4_dense_wdesc attn_q, attn_q_b, attn_k, attn_k_b, attn_v, attn_v_b;
    ds4_dense_wdesc attn_out, attn_out_b, attn_out_norm, attn_out_norm_b;   /* post-attn LayerNorm */
    ds4_dense_wdesc ffn_up, ffn_up_b, ffn_down, ffn_down_b;
    ds4_dense_wdesc layer_out_norm, layer_out_norm_b;                       /* post-FFN LayerNorm */
} ds4_bert_layer_desc;

typedef struct {
    unsigned n_layer, n_embd, n_head, head_dim, n_ff;
    unsigned pooling;   /* 1 = mean, 2 = CLS (gguf bert.pooling_type) */
    float    eps;
    ds4_dense_wdesc       token_embd, pos_embd, token_types, tok_norm, tok_norm_b;
    ds4_bert_layer_desc  *layers;   /* [n_layer] */
} ds4_bert_model_desc;

typedef struct ds4_bert_gpu ds4_bert_gpu;
ds4_bert_gpu *ds4_bert_gpu_create(const ds4_bert_model_desc *desc);
void ds4_bert_gpu_free(ds4_bert_gpu *g);
/* Encode M tokens -> mean-pooled, L2-normalized embedding [n_embd]. 0 on success. */
int ds4_bert_gpu_embed(ds4_bert_gpu *g, const ds4_bert_model_desc *desc,
                       const int *tokens, uint32_t n_tokens, float *out_embd);
int ds4_bert_embed(const char *model_path, const char *prompt, char *err, size_t errlen);
/* Load a qwen3_next model and greedily generate n_predict tokens (EXPERIMENTAL). */
int ds4_q3n_generate(const char *model_path, const char *prompt, int n_predict,
                     char *err, size_t errlen);
int ds4_gptoss_generate(const char *model_path, const char *prompt, int n_predict,
                        char *err, size_t errlen);

/* Interactive multi-turn ChatML chat REPL for dense models. Keeps the KV cache
 * across turns and generates until <|im_end|>/EOS (no token limit). ctx_size<=0
 * defaults to 4096; system may be NULL. Reads stdin until "/exit" or EOF. */
int ds4_dense_chat(const char *model_path, const char *system, int ctx_size,
                   char *err, size_t errlen);

/* 1 if the GGUF uses a supported dense architecture (qwen2/llama/...), else 0. */
int ds4_model_is_dense(const char *path);
void ds4_engine_summary(ds4_engine *e);
int ds4_engine_vocab_size(ds4_engine *e);
int ds4_engine_power(ds4_engine *e);
int ds4_engine_set_power(ds4_engine *e, int power_percent);
const char *ds4_engine_model_name(ds4_engine *e);
int ds4_engine_layer_count(ds4_engine *e);
uint32_t ds4_engine_layer_compress_ratio(ds4_engine *e, uint32_t layer);
uint64_t ds4_engine_hidden_f32_values(ds4_engine *e);
/* Stable id for cache compatibility.  0 is the original Flash shape, so old
 * KV files with the previously-zero reserved byte remain Flash-compatible;
 * Pro and later shapes must use nonzero ids. */
int ds4_engine_model_id(ds4_engine *e);
const char *ds4_backend_name(ds4_backend backend);
bool ds4_think_mode_enabled(ds4_think_mode mode);
const char *ds4_think_mode_name(ds4_think_mode mode);
const char *ds4_think_max_prefix(void);
uint32_t ds4_think_max_min_context(void);
ds4_think_mode ds4_think_mode_for_context(ds4_think_mode mode, int ctx_size);
/* Uses the active model shape selected by ds4_engine_open(); call after opening
 * the GGUF so Flash/Pro dimensions are known. */
ds4_context_memory ds4_context_memory_estimate(ds4_backend backend, int ctx_size);
ds4_context_memory ds4_context_memory_estimate_with_prefill(
        ds4_backend backend,
        int ctx_size,
        uint32_t prefill_chunk);
bool ds4_log_is_tty(FILE *fp);
void ds4_log(FILE *fp, ds4_log_type type, const char *fmt, ...);
int ds4_engine_generate_argmax(ds4_engine *e, const ds4_tokens *prompt,
                               int n_predict, int ctx_size,
                               ds4_token_emit_fn emit,
                               ds4_generation_done_fn done,
                               void *emit_ud,
                               ds4_session_progress_fn progress,
                               void *progress_ud);
int ds4_engine_collect_imatrix(ds4_engine *e,
                               const char *dataset_path,
                               const char *output_path,
                               int ctx_size,
                               int max_prompts,
                               int max_tokens);
void ds4_engine_dump_tokens(ds4_engine *e, const ds4_tokens *tokens);
int ds4_dump_text_tokenization(const char *model_path, const char *text, FILE *fp);
int ds4_engine_head_test(ds4_engine *e, const ds4_tokens *prompt);
int ds4_engine_first_token_test(ds4_engine *e, const ds4_tokens *prompt);
int ds4_engine_metal_graph_test(ds4_engine *e, const ds4_tokens *prompt);
int ds4_engine_metal_graph_full_test(ds4_engine *e, const ds4_tokens *prompt);
int ds4_engine_metal_graph_prompt_test(ds4_engine *e, const ds4_tokens *prompt, int ctx_size);

void ds4_tokens_push(ds4_tokens *tv, int token);
void ds4_tokens_free(ds4_tokens *tv);
void ds4_tokens_copy(ds4_tokens *dst, const ds4_tokens *src);
bool ds4_tokens_starts_with(const ds4_tokens *tokens, const ds4_tokens *prefix);

void ds4_tokenize_text(ds4_engine *e, const char *text, ds4_tokens *out);
void ds4_tokenize_rendered_chat(ds4_engine *e, const char *text, ds4_tokens *out);
void ds4_chat_begin(ds4_engine *e, ds4_tokens *tokens);
void ds4_encode_chat_prompt(
        ds4_engine *e,
        const char *system,
        const char *prompt,
        ds4_think_mode think_mode,
        ds4_tokens *out);
void ds4_chat_append_max_effort_prefix(ds4_engine *e, ds4_tokens *tokens);
void ds4_chat_append_message(ds4_engine *e, ds4_tokens *tokens, const char *role, const char *content);
void ds4_chat_append_assistant_prefix(ds4_engine *e, ds4_tokens *tokens, ds4_think_mode think_mode);

char *ds4_token_text(ds4_engine *e, int token, size_t *len);
int ds4_token_eos(ds4_engine *e);
int ds4_token_user(ds4_engine *e);
int ds4_token_assistant(ds4_engine *e);

int ds4_session_create(ds4_session **out, ds4_engine *e, int ctx_size);
void ds4_session_free(ds4_session *s);
int ds4_session_power(ds4_session *s);
int ds4_session_set_power(ds4_session *s, int power_percent);
bool ds4_session_is_distributed(ds4_session *s);
void ds4_session_set_progress(ds4_session *s, ds4_session_progress_fn fn, void *ud);
/* UI-only progress. It may report fine-grained progress inside a prefill chunk;
 * callers must not treat it as a durable KV checkpoint boundary. */
void ds4_session_set_display_progress(ds4_session *s, ds4_session_progress_fn fn, void *ud);
/* Optional cooperative cancellation.  ds4_session_sync() checks it only at
 * safe boundaries where the live checkpoint is either unchanged or represents a
 * valid token prefix, and returns DS4_SESSION_SYNC_INTERRUPTED when it stops. */
void ds4_session_set_cancel(ds4_session *s, ds4_session_cancel_fn fn, void *ud);
void ds4_session_report_progress(ds4_session *s, const char *event, int current, int total);
/* Distributed coordinator sessions return 1 when the full layer route is
 * available, 0 when it is still incomplete, and -1 for a local API error. */
int ds4_session_distributed_route_ready(ds4_session *s, char *err, size_t errlen);

typedef enum {
    DS4_SESSION_REWRITE_ERROR = -1,
    DS4_SESSION_REWRITE_OK = 0,
    /* The live backend state cannot be rewritten safely in place.  The caller should
     * restore an older checkpoint if it has one, then sync to the prompt. */
    DS4_SESSION_REWRITE_REBUILD_NEEDED = 1,
} ds4_session_rewrite_result;

/* Synchronize the live session to a full prompt token prefix.  If the current
 * checkpoint is a prefix, only the suffix is evaluated; otherwise the backend
 * state is refilled from scratch. */
int ds4_session_sync(ds4_session *s, const ds4_tokens *prompt, char *err, size_t errlen);
bool ds4_session_rewrite_requires_rebuild(int live_len, int canonical_len, int common);
ds4_session_rewrite_result ds4_session_rewrite_from_common(
        ds4_session *s, const ds4_tokens *prompt, int common,
        char *err, size_t errlen);
int ds4_session_common_prefix(ds4_session *s, const ds4_tokens *prompt);
int ds4_session_argmax(ds4_session *s);
int ds4_session_argmax_excluding(ds4_session *s, int excluded_id);
int ds4_sample_logits(const float *logits, int n_vocab, float temperature,
                      int top_k, float top_p, float min_p, uint64_t *rng);
int ds4_session_sample(ds4_session *s, float temperature, int top_k, float top_p, float min_p, uint64_t *rng);
int ds4_session_top_logprobs(ds4_session *s, ds4_token_score *out, int k);
int ds4_session_token_logprob(ds4_session *s, int token, ds4_token_score *out);
int ds4_session_copy_logits(ds4_session *s, float *out, int cap);
int ds4_session_set_logits(ds4_session *s, const float *logits, int n);
int ds4_session_eval(ds4_session *s, int token, char *err, size_t errlen);
int ds4_session_eval_speculative_argmax(ds4_session *s, int first_token,
                                        int max_tokens, int eos_token,
                                        int *accepted, int accepted_cap,
                                        char *err, size_t errlen);
void ds4_session_invalidate(ds4_session *s);
void ds4_session_rewind(ds4_session *s, int pos);
int ds4_session_pos(ds4_session *s);
int ds4_session_ctx(ds4_session *s);
int ds4_session_prefill_cap(ds4_session *s);
int ds4_engine_routed_quant_bits(ds4_engine *e);
bool ds4_engine_has_output_head(ds4_engine *e);
bool ds4_engine_has_mtp(ds4_engine *e);
int ds4_engine_mtp_draft_tokens(ds4_engine *e);
const ds4_tokens *ds4_session_tokens(ds4_session *s);

/* Low-level graph slice entry points used by distributed inference.  The
 * transport/session routing logic lives in ds4_distributed.c. */
int ds4_session_layer_slice_reset(ds4_session *s, char *err, size_t errlen);
int ds4_session_eval_layer_slice(ds4_session *s,
                                 const int *tokens,
                                 uint32_t n_tokens,
                                 uint32_t pos0,
                                 uint32_t layer_start,
                                 uint32_t layer_end,
                                 const float *input_hc,
                                 float *output_hc,
                                 bool output_logits,
                                 float *logits,
                                 char *err,
                                 size_t errlen);
int ds4_session_eval_output_head_from_hc(ds4_session *s,
                                         const float *hidden_hc,
                                         uint32_t n_tokens,
                                         float *logits,
                                         char *err,
                                         size_t errlen);

/* Disk KV payload helpers.  HTTP/agent code owns the outer file header and
 * persistence policy; the engine owns the DS4-specific serialized graph state. */
#define DS4_SESSION_PAYLOAD_MAGIC UINT32_C(0x34565344) /* "DSV4" */
#define DS4_SESSION_PAYLOAD_VERSION UINT32_C(2)
#define DS4_SESSION_PAYLOAD_U32_FIELDS 13u
#define DS4_SESSION_LAYER_PAYLOAD_MAGIC UINT32_C(0x4c565344) /* "DSVL" */
#define DS4_SESSION_LAYER_PAYLOAD_VERSION UINT32_C(1)
#define DS4_SESSION_LAYER_PAYLOAD_U32_FIELDS 14u

uint64_t ds4_session_payload_bytes(ds4_session *s);
int ds4_session_stage_payload(ds4_session *s, ds4_session_payload_file *out,
                              char *err, size_t errlen);
int ds4_session_write_staged_payload(const ds4_session_payload_file *payload,
                                     FILE *fp, char *err, size_t errlen);
void ds4_session_payload_file_free(ds4_session_payload_file *payload);
int ds4_session_save_payload(ds4_session *s, FILE *fp, char *err, size_t errlen);
int ds4_session_load_payload(ds4_session *s, FILE *fp, uint64_t payload_bytes, char *err, size_t errlen);
int ds4_session_save_snapshot(ds4_session *s, ds4_session_snapshot *snap, char *err, size_t errlen);
int ds4_session_load_snapshot(ds4_session *s, const ds4_session_snapshot *snap, char *err, size_t errlen);
void ds4_session_snapshot_free(ds4_session_snapshot *snap);

uint64_t ds4_session_layer_payload_bytes(ds4_session *s,
                                         uint32_t layer_start,
                                         uint32_t layer_end);
int ds4_session_save_layer_payload(ds4_session *s, FILE *fp,
                                   uint32_t layer_start, uint32_t layer_end,
                                   char *err, size_t errlen);
int ds4_session_load_layer_payload(ds4_session *s, FILE *fp,
                                   uint64_t payload_bytes,
                                   const int *tokens, uint32_t n_tokens,
                                   uint32_t layer_start, uint32_t layer_end,
                                   char *err, size_t errlen);

#endif

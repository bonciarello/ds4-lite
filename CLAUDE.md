# ds4-lite

This repository (`github.com/bonciarello/ds4-lite`) is a **fork** of
[`antirez/ds4`](https://github.com/antirez/ds4) — the DwarfStar native
inference engine for DeepSeek V4 Flash / PRO.

Upstream: https://github.com/antirez/ds4

## Obiettivo del fork

Rendere `ds4` **multi-architettura**: far convivere i modelli **DeepSeek V4**
(MoE + MLA, grandi) con **modelli densi piccoli** (Llama / Qwen / Mistral 7B–14B),
selezionati automaticamente dal GGUF al caricamento. **Invariante**: il path
DeepSeek non deve regredire (verifica contro `tests/test-vectors/`).

Design e roadmap: [docs/DENSE_SUPPORT_DESIGN.md](docs/DENSE_SUPPORT_DESIGN.md).

Stato:
- **Fase 0** ✅ studio + design.
- **Fase 1** ✅ astrazione `arch_family` landata in `ds4.c` (enum
  `ds4_arch_family`, campi shape `arch`/`meta_ns`/`n_ff`, helper di capacità
  `ds4_has_moe/mla/indexer/hc`/`ds4_arch_is_deepseek`). I 3 gate SSD-streaming
  `(PRO||FLASH)` convertiti a `ds4_arch_is_deepseek()` (default sicuro=disabilitato
  per i densi). Gli altri branch `DS4_MODEL_VARIANT ==` sono distinzioni interne
  PRO-vs-FLASH e restano. Additivo, `make all` verde, zero warning, binari OK.
  Gate test-vectors NON ancora eseguito (richiede pesi GGUF non scaricati) →
  non-regressione garantita per costruzione, non per esecuzione.
- **Fase 2** ✅ loader dense (target **Qwen2**, famiglie: qwen2/qwen3/llama/mistral).
  In `ds4.c`: dispatch su `general.architecture` in `config_validate_model`,
  `config_build_dense_shape()` che costruisce `g_ds4_shape` dai metadati `<ns>.*`
  (helper `*_ns`), con MoE/MLA/indexer/hc a 0. Per ora si ferma con messaggio
  onesto "graph Fase 3 non disponibile".
  **Validato su GGUF Qwen2-7B-Instruct Q4_K_M reale** (4.68 GB in `gguf/`):
  tutti i campi letti corretti vs parse indipendente; provati i fallback
  head_dim=n_embd/n_head (key_length assente → 128) e vocab dal tokenizer
  (vocab_size assente → 152064). Arch sconosciuta → path DeepSeek invariato.
  `make all` verde. Le validazioni tensor-layout condizionali (§2.2/2.3) sono
  legate al grafo → accorpate alla Fase 3.
- **Fase 3** 🟡 in corso (grafo dense su GPU), decomposta in 8 sotto-task
  (`docs/DENSE_SUPPORT_DESIGN.md` §4c). Oracolo: llama.cpp (`brew install llama.cpp`).
  - **3.1** ✅ **pre-tokenizer Qwen2**. In `ds4.c`: enum `ds4_pretok` + campo
    `pre_type` in `ds4_vocab` (letto da `tokenizer.ggml.pre`), `vocab_lookup_optional`
    per token speciali assenti nei vocab non-DeepSeek (bos/eos da metadata),
    `qwen2_tokenize_text()` (split GPT-2/Qwen2) con dispatch in `bpe_tokenize_text`.
    **Validato 9/9** vs `llama-tokenize` (`tests/validate_qwen2_tokenizer.sh`):
    contrazioni, indentazione, newline, UTF-8, punteggiatura. DeepSeek (JOYAI)
    invariato. Hook di test: `./ds4 --dump-tokens -p` (bypassa graph/weights).
  - **3.2** ✅ **bind pesi densi**. In `ds4.c`: campi densi in `ds4_layer_weights`
    (attn_q/k/v+bias, attn_out, ffn_gate/up/down), `weights_bind_layer_dense` /
    `weights_bind_output_dense` / `weights_bind_dense`, dispatch in `weights_bind`
    su `ds4_arch_is_deepseek()`. `config_validate_model` denso non esce più
    (costruisce shape e ritorna); guard "graph 3.4+" spostato dopo `weights_bind`
    in engine open. **Validato** su Qwen2 reale: `--inspect` lega 28 layer senza
    `ds4_die`; run normale → "weights bound OK". Bias QKV opzionali (Qwen2 sì,
    Llama no). `make all` verde.
  - **3.3** ✅ **layout validation dense**. `weights_validate_layout_dense` +
    `dense_expect_dims` (type-agnostic, valida solo dim via `tensor_expect_layout`),
    chiamata in `weights_bind_dense`. **Validato** su Qwen2: ~300 controlli dim
    (28 layer) passano → q_dim=3584, kv_dim=512 (GQA), n_ff=18944 combaciano.
  - **3.4** 🟡 **CPU reference forward** scritto in `ds4.c`: `forward_token_dense_cpu`
    (+ `dense_layer_forward_cpu`, `dense_matvec`, `dense_dequant_row`,
    `dense_rope_neox` NEOX, `dense_kv_cache`). Transformer pre-norm standard:
    embedding → GQA attn + NEOX RoPE → SwiGLU FFN → output. Dequant F32/F16/Q8_0
    (K-quant NON decodificati di proposito: il GGUF Q4_K_M ha Q6_K → serve un GGUF
    F16/Q8_0 per validare). **Compila pulito**, funzioni `DS4_MAYBE_UNUSED`.
    ⚠️ NON ancora wired nella eval dispatch e NON runtime-validato: il path CPU
    crasha il kernel macOS → validare su **Linux** (`make cpu`) vs llama.cpp.
    Non eseguire il path CPU su macOS.
  - **3.5** 🟡 port Metal avviato. **1° step validato su questo Mac (M1 Max)**:
    `ds4_gpu_dense_matvec_selftest` (in `ds4_metal.m`, flag `./ds4 --metal-dense-selftest`)
    esegue `kernel_mul_mv_f32_f32` su GPU e confronta col dot product CPU → **PASS**
    (<1e-4). Valida il path di dispatch denso (buffer/args/pipeline/encode/readback)
    end-to-end, senza modello. Kernel densi confermati in `metal/dense.metal`.
    Step validati (in `--metal-dense-selftest`, su M1 Max, tutti vs CPU ref):
    - step 0/1 ✅ matvec **F32 + Q8_0** (<1e-4)
    - step 2 ✅ kernel **RoPE NEOX** (`kernel_dense_rope_neox_f32`) (<1e-5)
    - step 3a ✅ kernel **SwiGLU + RMSNorm** densi (<1e-5)
    - step 3b ✅ **blocco FFN completo** (rmsnorm→gate/up→swiglu→down→residuo) chainato (<1e-4)
    - step 3c ✅ **blocco ATTENTION completo** (rmsnorm→q/k/v matvec+bias→RoPE→append KV→
      GQA attention→out matvec→residuo) chainato, KV pre-riempita 3 pos (<1e-4).
      Kernel: `kernel_dense_attn_decode_f32` (un thread/query head, online softmax).
    **Entrambi i sub-blocchi transformer (attn + FFN) girano end-to-end su GPU.**
    Kernel densi in `metal/dense.metal`; helper host riusabili `ds4_gpu_run_simple`,
    `ds4_gpu_dense_matvec_f32`, `ds4_gpu_matvec_run_once`.
  - **step 4 sotto-step 1+2** ✅ matvec type-dispatch sui **pesi reali** validato:
    `./ds4 --metal-dense-weight-test MODEL.gguf` (`ds4_dense_weight_test` /
    `ds4_gpu_dense_matvec_verify`). Su Qwen2 reale: attn_q (Q4_K), attn_v/ffn_down/
    output (Q6_K) tutti PASS, rel_err ~1e-7 vs CPU. Copia il peso su GPU; lo zero-copy
    `ds4_gpu_wrap_model_range` è rinviato al driver (serve per non duplicare GB di pesi).
  - **step 4 (3-6)** ✅✅ **FORWARD DENSO FUNZIONANTE** — `./ds4 --metal-dense-generate
    MODEL PROMPT [N]`. Driver GPU completo in `ds4_metal.m` (`ds4_dense_gpu_create/
    forward/free`): embedding (Q4_K dequant) → 28× [rmsnorm → attn GQA (q/k/v matvec+
    bias, RoPE NEOX, KV cache via tensor-view, attention) → residuo → rmsnorm → FFN
    SwiGLU → residuo] → output_norm → output matvec → logits. Pesi reali via cache GPU
    per-tensore + matvec type-dispatch. Orchestrazione + generate greedy in `ds4.c`
    (`ds4_dense_generate`).
    **VALIDATO**: continuazione greedy **identica a llama-simple** su Qwen2-7B Q4_K_M
    ("The capital of France is" → " Paris. It is the most populous city in the European
    Union and the second most"). Bug risolto: `ds4_gpu_tensor_copy` (blit async su
    g_batch_cb non committato) per l'append KV → sostituito con scrittura diretta nello
    slot via `ds4_gpu_tensor_view` (operazioni ordinate).
    **Polish**:
    - ✅ **Wiring `./ds4 -p`**: i modelli densi col prompt one-shot sono instradati a
      `ds4_dense_generate` (helper `ds4_model_is_dense` in `ds4.c`, routing in `main()`
      di `ds4_cli.c`). `./ds4 -m qwen2.gguf -p "..."` funziona. (Chat/server dense = follow-up.)
    - ⏸️ **Velocità** (~4 tok/s, ~10x più lento di llama.cpp): **compute-bound** sui kernel
      reference 1-thread/riga. Tentativi (entrambi corretti ma NON più veloci, revertiti):
      (a) **batching** command buffer (`begin/end_commands`) — l'overhead dispatch non è il
      collo di bottiglia; (b) **simdgroup-per-riga** (32 thread/riga, blocchi strided +
      `simd_sum`) — **più lento**: con `n_embd=3584` ci sono solo 14 blocchi/riga (<32 lane)
      → sotto-utilizzo + overhead simd_sum, e l'output matvec lancia 4.8M thread per lavoro
      minimo. Il vero lever resta un matvec a **parallelismo per-elemento** (32 thread sui
      ~3584 elementi della riga, non sui 14 blocchi) con dequant per-elemento — più complesso,
      payoff incerto, follow-up dedicato.
    - ⏸️ **Zero-copy weight wrap**: `ds4_gpu_wrap_model_range` restituisce un `MTLBuffer`
      grezzo, i miei helper usano `ds4_gpu_tensor` (mismatch astrazione) → più invasivo.
      Ora copia ~4.7GB su GPU (ok su 32GB unified; `ds4_gpu_set_model_map_range` mappa le view).
    **Supporto quant per il dense — COMPLETO e validato su GPU** (`--metal-dense-selftest`,
    12 casi PASS <1e-2): F32, **Q8_0**, **Q3_K**, **Q4_K**, **Q5_K**, **Q6_K** matvec densi
    (kernel `kernel_dense_mul_mv_{q3,q4,q5,q6}_K_f32` in `metal/dense.metal`, dequant
    canonica GGML inline). Aggiunti in `ds4.c`: `DS4_TENSOR_Q3_K=11/Q5_K=13/Q6_K=14` +
    `block_q3_K/q5_K/q6_K` (110/176/210B). Caricamento già ok via `gguf_types`.
    Il Qwen2 Q4_K_M (169 Q4_K + 29 Q6_K + 141 F32) è interamente calcolabile su GPU; e
    la copertura K-quant comune è completa (Q3/Q4/Q5/Q6_K). NB: kernel "reference"
    (1 thread/riga), correttezza-first; ottimizzazione dopo. Validazione indipendente
    finale = step 4 greedy vs llama.cpp.
    Nota: i kernel `.metal` sono caricati da disco a runtime (modifiche ai soli `.metal`
    non richiedono rebuild di ds4_metal.o); eseguire `./ds4` dalla root del repo.
  - 3.7 wiring eval dispatch + validazione greedy vs llama.cpp (su questo Mac).
  - Nota quant: ds4 supporta F32/F16/Q8_0/Q4_K/Q2_K/IQ2_XXS ma **NON Q6_K**
    (assente dall'enum). RoPE DeepSeek è GPT-J (coppie i,i+1); densi usano NEOX.
  - Baseline Qwen2 (llama.cpp, M-series): pp512≈413 t/s, tg128≈44 t/s.
    Riferimento greedy + harness: `tests/bench_dense.sh`.
  - GGUF reale in `gguf/qwen2-7b-instruct-q4_k_m.gguf` (4.68 GB, non committare).

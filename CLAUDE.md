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
    - ⏸️ **Velocità** (~3-4 tok/s decode, ~14x più lento di llama.cpp). **Profilato**
      (`DS4_DENSE_PROFILE=1`): ffn 51%, attn 41%, output 8% per token. 3 tentativi, tutti
      corretti ma NON più veloci, revertiti: (a) batching command buffer — non dispatch-bound;
      (b) simdgroup-per-riga uniforme — più lento sui matvec a basso nblk; (c) simdgroup solo
      su alto nblk (ffn_down, 74 blocchi) — **anche più lento**. **Finding raffinato**: NON è
      parallelism-bound — 3584 righe saturano già le ALU dell'M1 Max, quindi aggiungere
      thread/simd_sum aggiunge solo overhead. **Analisi memory-bound (target preciso)**:
      decode legge tutti i pesi 1 volta/token → 4.7GB/400GB·s = ~12 ms/token (floor memoria).
      llama.cpp ≈22 ms/token (44 t/s) = **memory-bound** (ottimale, dequant nascosto dietro le
      letture). ds4 ≈312 ms/token (3.2 t/s) = **compute-bound** sul dequant scalare → serve
      **~27x** di speedup compute per toccare il floor. float4 sul kernel q4_K provato: corretto
      ma **0 speedup misurabile** (la GPU Apple gestisce già bene lo scalare), revertito. Via B
      (K-quant generale) richiede l'INTERO kernel ottimizzato llama.cpp-style (dequant vettoriale
      + tiling con x condiviso in threadgroup memory + simdgroup-matrix), non una singola tecnica
      → lavoro pluri-sessione dedicato. Profila con `DS4_DENSE_PROFILE=1`, valida con
      `tests/bench_dense.sh`. Dettagli in [[ds4-lite-dense-optimizations]].
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
  - Baseline Qwen2 (llama.cpp, M1 Max): pp512≈413 t/s, tg128≈44 t/s.
  - **Benchmark denso** `tests/bench_dense.sh` (metodologia standard prefill/decode):
    misura ds4 dense prefill/decode t/s (warmup + measured) vs `llama-bench` pp512/tg128
    e verifica la correttezza greedy vs `llama-simple`. `--sweep` produce CSV
    (formato speed-bench del progetto) + SVG via `speed-bench/plot_speed.py`.
    `ds4_dense_generate` stampa "dense metrics: prefill X t/s (TTFT Ys), gen Z t/s";
    con `DS4_BENCH_CSV=path` appende una riga CSV. Misura M1 Max: ds4 ~3.2 t/s decode
    vs llama.cpp ~44 (~14x), greedy identico. (Le 2 ottimizzazioni in [[ds4-lite-dense-optimizations]].)
  - GGUF reale in `gguf/qwen2-7b-instruct-q4_k_m.gguf` (4.68 GB, non committare).

## gemma-3 (general.architecture == "gemma3") ✅

Supporto **gemma-3** (validato su `gemma-3-27b-it.Q4_K_M`, 62 layer) sul **path denso**:
`./ds4 --metal-dense-generate gguf/gemma-3-27b-it.Q4_K_M.gguf "..."` e
`./ds4 --metal-dense-chat gguf/gemma-3-27b-it.Q4_K_M.gguf`. **Validato byte-identico a
llama-simple** ("1, 2, 3, 4, 5, 6, 7," → " 8, 9, 10, 11, 12, 13"); chat risponde "Paris."
~10.7 tok/s su M1 Max. Non regressione: Qwen2 denso + `--metal-dense-selftest` invariati.

- **Arch**: `DS4_ARCH_GEMMA3` (`ds4_arch_is_gemma3()`). `config_build_gemma3_shape` riusa
  `config_build_dense_shape` (head_dim=key_length=128 ≠ n_embd/n_head; vocab dal tokenizer;
  rope.freq_base 1e6 = base globale) e marca arch + `n_swa`=1024. Gira sul **driver denso**
  (`ds4_dense_model_desc.gemma` + branch `ds4_gemma_gpu_forward`), NON un driver separato.
- **Quirk gemma** (in `ds4_gemma_gpu_forward`): embedding ×√n_embd (Q6_K tied output);
  **QK-norm** per-head; **4 norm/layer** (input/post-attn/pre-ffn/post-ffn); **GeGLU** (gelu
  tanh); **sliding-window 1024** su 5/6 layer (globale a `il%6==5`) con RoPE locale base 10000
  freq_scale 1.0 vs globale 1e6 freq_scale 0.125; attn scale = 1/√(n_embd/n_head) = 1/√168 per
  il 27B (NON 1/√head_dim). 4 kernel Metal isolati in `dense.metal`
  (`kernel_gemma_{rms_norm_f32_sg,qk_norm_f32,geglu_f32,rope_neox_f32}`).
- **Gotcha pesi norm**: i pesi RMSNorm/QK-norm gemma **includono già il +1** (llama.cpp lo
  bake in conversione) → moltiplicare per `w` diretto, NON `(1+w)`.
- **Gotcha GeGLU**: il `tanh` di Metal (fast-math, default ON) **overflowa** per argomenti
  grandi (exp→inf, inf/inf=NaN) → clamp dell'argomento a ±15 nel kernel geglu.
- **Tokenizer SentencePiece** (`DS4_PRETOK_SPM`): gemma usa `tokenizer.ggml.model=llama`
  (SPM con `scores`, niente `merges`). `vocab_load` rende i merges opzionali + carica gli
  scores; `spm_tokenize_text` (spazio→▁ U+2581, merge guidato dagli score sui simboli
  adiacenti, byte-fallback `<0xXX>`); detokenizzazione SPM in `dense_token_bytes`/
  `dense_print_token` (▁→spazio). Chat template: `<bos><start_of_turn>user\n{q}<end_of_turn>\n
  <start_of_turn>model\n`, stop su `<end_of_turn>`(106)/eos; reflection+tools OFF per gemma.
- **Bug latente risolto**: `DS4_MAX_LAYER` era 61 (DeepSeek Pro) ma gemma-3-27b ha 62 →
  off-by-one OOB su `ds4_weights.layer[]` che corrompeva lo stack adiacente (il vocab in chat;
  il generate usa `desc->layers` su heap quindi sembrava ok). Alzato a 64 + guardia runtime in
  `config_build_dense_shape`. Riguardava ogni modello denso ≥62 layer.
- **Ottimizzazioni RAM**:
  - **KV cache sliding-window (ring buffer)**: i layer locali (52/62) cappano la cache a
    `swa_window`=1024 slot e la riusano come ring (scrivi a `pos%cap`, attendi `[0, min(pos+1,cap))`;
    la softmax è order-invariant e ogni K è già ropato alla sua pos assoluta). Solo i ~10 layer
    globali tengono la cache piena. `dense_kv_cap()` + `g->swa_window/swa_pattern` in
    `ds4_dense_gpu`. Riduzione KV ~3× a 4K ctx, ~5–6× a 32K–128K (es. 32K: 15.4→2.9 GiB).
    Validato byte-corretto a 1942 token (ring in wrap). L'auto-context gemma conta solo i layer
    globali (f16) → contesti molto più ampi (es. ~16K→128K+ su 32GB).
  - **SSD streaming dei pesi**: se `model_size > 0.60×RAM`, `ds4_dense_gpu_create` salta il pin di
    residency (`setenv DS4_METAL_NO_RESIDENCY`) → le pagine dei pesi mmap si caricano on-demand
    dall'SSD (RAM wired minore, decode più lento) invece di andare in OOM. Override con
    `DS4_DENSE_STREAM` / `DS4_DENSE_RESIDENT`. Il wrapping zero-copy è invariato; si gata solo la
    residency. (gemma 16GB su 32GB → 16<19.2 → resident, comportamento invariato.)
- **Limite noto**: il prefill gemma è token-by-token (no matmul batchato) → O(ctx²), lento su prompt
  lunghi (~10 t/s, TTFT 191s su 1942 token). Follow-up: aggiungere il prefill batchato gemma.
- Dettagli + status in [[gemma3-support]].

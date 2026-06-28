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
  - **Auto-context budget GPU-resident (fix OOM)**: con modello residente, pesi+KV+command-buffer
    sono tutti allocazioni GPU. Su Mac unified-memory `recommendedMaxWorkingSetSize` sovra-riporta
    (≈ RAM fisica) e NON è un bound OOM affidabile; il command buffer va in OOM ben prima di
    0.85×RAM. Quindi `ds4_auto_context` per il caso residente usa budget = `0.62×RAM` (+ cap col
    working set se più basso) e 2.5 GiB di headroom sopra model+KV. Su gemma-27b/32GB → auto 32K
    (~18 GiB di picco) invece di 128–256K che OOMmava. (Prima il budget 0.85×RAM sceglieva 256K →
    KV 10.7 GiB + modello 16 GiB → `kIOGPUCommandBufferCallbackErrorOutOfMemory`.)
  - **SSD streaming dei pesi**: se `model_size > 0.60×RAM`, `ds4_dense_gpu_create` salta il pin di
    residency (`setenv DS4_METAL_NO_RESIDENCY`) → le pagine dei pesi mmap si caricano on-demand
    dall'SSD (RAM wired minore, decode più lento) invece di andare in OOM. Override con
    `DS4_DENSE_STREAM` / `DS4_DENSE_RESIDENT`. Il wrapping zero-copy è invariato; si gata solo la
    residency. (gemma 16GB su 32GB → 16<19.2 → resident, comportamento invariato.)
- **Limite noto**: il prefill gemma è token-by-token (no matmul batchato) → O(ctx²), lento su prompt
  lunghi (~10 t/s, TTFT 191s su 1942 token). Follow-up: aggiungere il prefill batchato gemma.
- **Benchmark vs llama.cpp** (`tests/bench_vs_llama.sh [N_PREDICT] [GLOB]`): solo sui GGUF già
  scaricati in `gguf/` (non scarica nulla), per ciascuno misura prefill/decode t/s + peak RSS in 3
  config: **ds4 resident**, **ds4 streaming** (`DS4_DENSE_STREAM`), **llama.cpp** (`llama-bench`).
  Salta i modelli non-dense (q3n/DeepSeek). Lo streaming è validato dal **calo del prefill t/s**
  (pesi faultati dall'SSD, non pinnati); il decode è invariato (pagine in cache). Misure M1 Max 32GB:
  decode ds4 ≈ **89% di llama.cpp** (gemma 10.6 vs 11.9; qwen2 39 vs 44), prefill più lento (gemma
  10.8 vs 70 per il token-by-token; qwen2 201 vs 260), RSS leggermente inferiore a llama. Output: tabella + CSV.
- Dettagli + status in [[gemma3-support]].

## qwen3_next (MoE) — streaming esperti e carico RAM

Il modello `Qwen3-Next-80B-A3B` (~48 GiB su disco, ~3B attivi: 6/512 esperti per layer per
token) gira su 32 GB via `./ds4 --metal-q3n-generate <gguf> "..." N`. Driver `ds4_q3n_gpu_*`
(non l'engine DeepSeek): copia/streamma gli esperti on-demand.

- **Cap della cache esperti (lever RAM)** — `DS4_Q3N_EXPERT_CACHE_GIB=N`: gli esperti routati
  passano da `q3n_cached_expert` (LRU O(1) su `NSMutableOrderedSet`); `q3n_prune_experts` libera
  i più freddi dopo ogni layer MoE (esperto condiviso + pesi densi restano pinnati). Con il cap
  attivo gli esperti vengono letti con **pread direttamente nel buffer GPU** (non memcpy dal mmap)
  + prefetch **F_RDADVISE**. Default (env non settata) = illimitato = comportamento originale,
  byte-identico.
- **Metrica RAM corretta = "peak memory footprint" di `/usr/bin/time -l`** (= `phys_footprint`,
  memoria privata NON riciclabile), **NON `ps -o rss`**. La ps-rss conta anche le pagine *clean*
  file-backed (mmap/UBC, ~26.7 GiB qui, costanti) che il kernel ricicla per prime sotto pressione:
  sovrastima enormemente il carico q3n ed è insensibile al cap. Il footprint **scala col cap**
  (M1 Max, N=8): illimitato 7.37 / cap4 5.56 / cap2 3.57 / cap1 2.57 GiB ≈ base(~1.6)+cap. pread
  è anche **più veloce** del baseline mmap (cap4 4.27 vs 3.25 t/s).
- **Perché serve pread**: copiare dal mmap faulta le pagine dell'esperto nella RSS e su macOS quelle
  pagine file-backed NON sono evictabili (`madvise DONTNEED` è un no-op per loro — stesso muro del
  fallito "Via 2" denso). pread le tiene fuori dal footprint del processo (vanno nella buffer cache
  riciclabile). Questo riquadra anche il denso: i pesi densi sono copie GPU private → lì
  footprint≈rss≈dimensione modello (irriducibile); gli esperti MoE inattivi sono cache riciclabile
  → il footprint reale del MoE è una frazione del modello.
- **Quant come SECONDO lever (Q4 → Q2_K → IQ2_XXS)**: esperti più piccoli = meno byte da streammare
  per token → attacca direttamente il collo di banda SSD (il limite reale; il prefetch parallelo non
  può, siamo già al picco ~5 GB/s) + footprint più basso a parità di cap + hit-rate migliore. Il
  matvec q3n copre **Q2_K–Q6_K + IQ2_XXS** (tipo 16). Confronto su `Qwen3-Next-80B-A3B` (M1 Max,
  footprint reale / decode t/s):

  | quant | disk | cap2 | cap1 |
  |-------|-----:|------|------|
  | Q4_K_M  | 48.5 GB | 3.57 / 4.11 | 2.57 / 3.74 |
  | Q2_K    | 29.3 GB | 3.25 / 5.31 | 2.25 / 4.93 |
  | IQ2_XXS | 21.2 GB | 3.17 / 5.88 | **2.14 / 5.59** |

  **IQ2_XXS vince su ogni asse** (l'80B a 2.14 GiB di footprint + 5.6 t/s); IQ2@cap1 batte Q4@cap2 su
  footprint *e* velocità. Tutti **byte-identici a llama-simple**.
- **IQ2_XXS — kernel + workflow** (`metal/moe.metal`): `kernel_dense_mul_mv_iq2_xxs_f32` (ref, 1
  thread/riga) + `_sg` (simdgroup NSG=2/nr0=2/simd_sum, grid letto da `constant`); il `_sg` è il path
  veloce (naive 2.45 → sg 6.18 t/s), riusa le tabelle grid/segni IQ2 già in moe.metal. Cablati in
  `dense_kquant_kernel`/`dense_kquant_sg_kernel` (case 16). `ds4_q3n_gpu_forward` usa
  `ds4_dense_embed_row` (qualsiasi K-quant) per `token_embd`, non più hardcoded a Q4_K.
  **L'IQ2_XXS RICHIEDE un imatrix** (llama-quantize fallisce senza): `llama-imatrix -m <Q4> -f
  calib.txt -o imat -ngl 0` (CPU, calibrazione >1024 token) → `llama-quantize --imatrix imat
  --allow-requantize <Q4> <out> IQ2_XXS`. NB: IQ2_S/XS/M hanno layout diverso (nessun kernel). La
  qualità IQ2 qui è limitata (imatrix minimale + doppia quant da Q4, niente sorgente BF16 offline) ma
  resta coerente.
- **Benchmark**: `tests/bench_q3n_cache.sh [N] [MODEL]` fa lo sweep del cap e riporta footprint vs
  maxRSS (+ t/s); passa il modello come `$2` per confrontare i quant. Dettagli/stato in [[qwen3next-impl]].

## Chat CLI (stile Claude Code) — `--metal-dense-chat`

La chat interattiva (`ds4_dense_chat`, condivisa da dense + gemma + q3n) replica le
funzionalità osservabili del CLI di Claude Code, **tenendo il banner box iniziale**. Il repo
`anthropics/claude-code` è solo il repo pubblico meta (niente sorgente CLI, pacchetto npm
minificato); le feature sono reimplementate da zero in C.

- **Rendering markdown** (`md_state`/`md_feed`/`md_emit_line` prima di `dense_chat_gen_response`):
  renderer line-buffered della risposta — code block ``` come box a barra laterale (`╭─ lang` /
  `│ ` cyan / `╰─`), `#` header → bold, bullet `-`/`*`/`N.`, inline **bold**/*italic*/`code`.
  Solo su tty; `DS4_NO_MARKDOWN` off, `DS4_FORCE_MARKDOWN` on (pipe).
- **Esc per interrompere**: durante la generazione il tty va in raw non-bloccante e un Esc ferma
  lo stream (cooked mode ripristinato per il prompt). Solo su tty.
- **Status footer + scorciatoie**: sotto l'input box, riga dim con `<modello> · ⏎ send · esc stop ·
  / commands · ^D exit` (il modello resta visibile dopo lo scroll del banner). In `dense_ctx_status`.
- **Menu slash autocomplete**: digitando `/` la status footer diventa un menu comandi filtrato live
  (match più vicino evidenziato) via il meccanismo status multi-riga di linenoise (`dense_slash_menu`
  in `dense_layout_cb`). `DENSE_CMDS[]` è la lista comandi.
- **Tools/function-calling** (`read_file`/`bash`/`web_fetch`/`web_search`/`write_file`/`edit_file`/
  `glob`/`grep`, formato Hermes `<tool_call>`): **ON per OGNI architettura** — dense, gemma e
  **qwen3_next** (`tools_on = getenv("DS4_DENSE_NO_TOOLS")==NULL`, nessuna esclusione per arch). Per
  gemma il prompt-tool è iniettato nel **primo turno user** (`gemma_sys_pending`, gemma non ha system
  role); gli altri usano il system role. Il loop di esecuzione tool è condiviso da dense + q3n.
  Validato end-to-end su gemma-3-27b e **Qwen3-Next-80B IQ2_XXS** (emette `<tool_call>` ben formato,
  esegue write_file + bash, risponde dal risultato reale). NB: il prompt-tool è ~624 token → su q3n
  lento aggiunge un prefill iniziale una tantum (più rapido con IQ2_XXS). Il tool loop rende chiamata
  (🔧) + box risultato (`dense_tool_render`). Disabilita con `DS4_DENSE_NO_TOOLS`.

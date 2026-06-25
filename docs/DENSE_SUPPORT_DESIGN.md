# Design — Supporto multi-architettura (modelli densi piccoli + DeepSeek V4)

**Stato:** Fase 0 (studio + design). Nessuna modifica al comportamento attuale.
**Obiettivo:** estendere DwarfStar/`ds4` da motore mono-modello (DeepSeek V4 Flash/PRO)
a motore **multi-architettura**, in cui modelli **densi piccoli** (Llama / Qwen /
Mistral 7B–14B) e i modelli **DeepSeek V4** (MoE + MLA) **convivono** e vengono
selezionati automaticamente al caricamento dal GGUF.

**Invariante di progetto:** il path DeepSeek V4 non deve mai regredire. Ogni fase è
verificabile contro i vettori ufficiali in `tests/test-vectors/`.

---

## 1. Architettura attuale (come ds4 assume "è DeepSeek")

### 1.1 Lo scheletro di dispatch *esiste già*

Il codice ha già l'ossatura per "più modelli che convivono", oggi usata solo per
distinguere due **taglie** dello stesso modello (Flash vs PRO):

| Elemento | Posizione | Ruolo |
| --- | --- | --- |
| `enum ds4_variant {FLASH, PRO}` | `ds4.c:135-138` | Identifica la variante |
| `struct ds4_shape` | `ds4.c:140-175` | Tutti i parametri del modello |
| `DS4_SHAPE_FLASH` / `DS4_SHAPE_PRO` | `ds4.c:177-249` | Costanti per le due varianti |
| `g_ds4_shape` (globale) | `ds4.c:251` | Shape attiva, scelta al load |
| `ds4_select_shape_from_metadata()` | `ds4.c:3736-3794` | Sceglie la shape dai metadati GGUF |
| `ds4_shape_matches_metadata()` | (chiamata da sopra) | Match esatto dei parametri |
| `DS4_MODEL_VARIANT == …` | 17 occorrenze | Branch comportamentali |

**Conseguenza per il design:** non si crea un dispatch nuovo, si **generalizza**
quello esistente da `variant` (taglia) a `arch_family` (famiglia architetturale).

### 1.2 Il problema: Flash e PRO sono la *stessa* architettura

`DS4_SHAPE_FLASH` e `DS4_SHAPE_PRO` differiscono solo in dimensioni (layer, embd,
heads, n_expert…). Entrambi presuppongono i **tre meccanismi DeepSeek V4**:

1. **MLA — Multi-head Latent Attention** (attention compressa, non standard):
   campi `n_head_dim=512`, `n_rot=64`, `n_lora_q`, `n_lora_o`, `n_value_dim`.
2. **MoE — Mixture of Experts**: `n_expert` (256/384), `n_expert_used=6`,
   `n_expert_shared=1`, `n_ff_exp`, più gli `expert_weight_scale`.
3. **Indexer + Hash-Compression (Sinkhorn)**: `n_indexer_head`, `n_indexer_top_k`,
   `n_hc`, `n_hc_sinkhorn_iter` — sparse attention specifica DeepSeek.

Un modello **denso** (Llama/Qwen) non ha **nessuno** di questi: attention standard
(GQA), un solo FFN SwiGLU per layer, niente routing/indexer/Sinkhorn.

### 1.3 Quanto è pervasiva l'assunzione DeepSeek (conteggio in `ds4.c`)

| Concetto | Occorrenze | Implica |
| --- | ---: | --- |
| `n_hc` (hash compression) | 433 | grafo + KV cache |
| `compress` | 432 | grafo + KV cache su disco |
| `indexer` | 418 | sparse attention |
| `routed` (experts) | 343 | MoE |
| `DS4_N_EXPERT` | 257 | MoE |
| `DS4_N_EXPERT_USED` | 154 | MoE routing |
| `n_lora` | 48 | MLA |
| `sinkhorn` | 32 | hash compression |

Lettura: l'assunzione non è in un punto, è **diffusa**. La strategia corretta è
**isolamento dietro `arch_family`**, non rimozione.

### 1.4 Metadati GGUF: namespace `deepseek4.*` hard-coded

Il loader legge esclusivamente chiavi `deepseek4.*` (`ds4.c:2018-2031`):
`deepseek4.block_count`, `deepseek4.attention.head_count`,
`deepseek4.attention.indexer.*`, `deepseek4.expert_count`,
`deepseek4.attention.compress_ratios` (richiesta, `ds4.c:3797`), ecc.
I GGUF densi usano namespace `llama.*` / `qwen2.* ` → serve un **mapping di chiavi**
per famiglia.

### 1.5 Dove vive il grafo di calcolo (Fase 3, il grosso)

| File | LOC | Contenuto |
| --- | ---: | --- |
| `ds4_metal.m` | 26 819 | Grafo + kernel Metal (target primario) |
| `ds4_cuda.cu` | 13 256 | Grafo + kernel CUDA |
| `ds4_gpu.h` | 1 024 | Interfaccia backend |
| `ds4_rocm.cu` | 131 | Shim ROCm |

I kernel **base e riusabili** ci sono già (`kernel_swiglu_f32`,
`kernel_rms_norm_f32_4`, `kernel_rms_norm_mul_f32_4`, RoPE, matmul/attention).
Il path dense **riusa** questi e **salta** MLA/indexer/MoE/Sinkhorn.

---

## 2. Mappa dei punti che bloccano un modello denso

Catalogo dei `ds4_die()` / `exit()` / validazioni da rendere condizionali alla
famiglia (`ARCH_DEEPSEEK`). Nel ramo `ARCH_DENSE` vanno saltati o sostituiti.

### 2.1 Selezione shape / metadati
- `ds4.c:3736-3794` — `ds4_select_shape_from_metadata()`: fa `exit(1)` se la shape
  non combacia con Flash/PRO. **Punto di aggancio principale del dispatch.**
- `ds4.c:2018-2031` — lettura chiavi `deepseek4.*`. Serve mapping per `llama.*`.
- `ds4.c:3796-3842` — `validate_compress_ratio_metadata()`: richiede
  `deepseek4.attention.compress_ratios`. Inesistente nei densi → saltare.
- `ds4.c:3843` — richiede `deepseek4.swiglu_clamp_exp` array → saltare/def. dense.

### 2.2 Validazione layout dei tensori
- `ds4.c:626,631` — indice layer entro `DS4_N_LAYER` (ok, neutro).
- `ds4.c:641` — `"unsupported DeepSeek4 model variant"` (lo `switch` su variant).
- `ds4.c:2571` — `"DSV4 indexer QAT expects 128-wide indexer rows"` → solo indexer.
- `ds4.c:3143,3193,3215` — `ds4_validate_layout`: tensori attesi (tipo/dim).
- `ds4.c:3238,3244` — tipi tensori **routed expert** + allineamento QK_K → MoE.
- `ds4.c:3421-3445` — presenza/forma dei **routed expert** → MoE.
- `ds4.c:3557-3561` — validazione range layer dei pesi.
- `ds4.c:3681` — MTP routed gate/up (speculative) → MoE.

### 2.3 Caricamento pesi degli esperti (tutto MoE, da saltare nel dense)
- `ds4.c:5536-5543` — atteso tensore esperti 3D.
- `ds4.c:5581-5588` — `IQ2_XXS` expert tensors (gate/up).
- `ds4.c:5653-5683` — coppie `IQ2_XXS` routed.
- `ds4.c:5713-5785` — `Q2_K` expert (down).
- `ds4.c:6004-6030` — `Q4_K` expert.

### 2.4 Branch comportamentali su variante (17 punti)
`ds4.c:633, 8347, 11575-11595, 13686-13715, 19507-19533, 20334, 25473, 26018`.
Da convertire in branch su `arch_family`/capacità, con default sicuro per il dense.

---

## 3. Design proposto: livello `arch_family`

### 3.1 Nuovo enum e campo capacità

```c
typedef enum {
    DS4_ARCH_DEEPSEEK = 0,   /* MLA + MoE + indexer (Flash, PRO)        */
    DS4_ARCH_DENSE    = 1,   /* attention standard + FFN denso          */
} ds4_arch_family;
```

`ds4_shape` guadagna:
```c
ds4_arch_family arch;     /* famiglia */
uint32_t        n_ff;     /* FFN denso (0 se MoE) */
const char     *meta_ns;  /* namespace metadati GGUF: "deepseek4" | "llama" | ... */
```

**Convenzione di capacità** (evita di sparpagliare `if arch==…`):
- `n_expert == 0`     → niente MoE (FFN denso via `n_ff`)
- `n_indexer_head==0` → niente sparse-attention/indexer
- `n_hc == 0`         → niente hash-compression/Sinkhorn
- `n_lora_q == 0`     → attention standard (non MLA)

Così la maggior parte dei branch diventa `if (DS4_N_EXPERT)` / `if (DS4_N_HC)`
invece di `if (variant == …)`, e i punti §2.4 si semplificano.

### 3.2 Esempio di shape densa (Qwen2-7B, indicativo)

```c
static const ds4_shape DS4_SHAPE_QWEN2_7B = {
    .name = "Qwen2 7B (dense)", .arch = DS4_ARCH_DENSE, .meta_ns = "qwen2",
    .n_layer = 28, .n_embd = 3584, .n_vocab = 152064,
    .n_head = 28, .n_head_kv = 4, .n_head_dim = 128,
    .n_ff = 18944, .rms_eps = 1e-6f, .rope_freq_base = 1000000.0f,
    /* MoE/MLA/indexer/hc tutti a 0 → disattivati per convenzione */
    .n_expert = 0, .n_lora_q = 0, .n_indexer_head = 0, .n_hc = 0,
};
```

### 3.3 Dispatch generalizzato

`ds4_select_shape_from_metadata()` (`ds4.c:3736`) diventa:
1. legge `general.architecture`;
2. se `"deepseek4"` → logica attuale (Flash/PRO), **invariata**;
3. se `"llama"/"qwen2"/"mistral"` → costruisce una shape densa dai metadati
   `<ns>.block_count`, `<ns>.embedding_length`, `<ns>.attention.head_count`,
   `<ns>.feed_forward_length`, ecc.;
4. altrimenti → errore con messaggio chiaro (come oggi).

---

## 4. Fasi (aggiornate per la convivenza)

| Fase | Contenuto | Difficoltà | Rischio regressione DeepSeek |
| --- | --- | --- | --- |
| **0** | Studio + questo documento | Bassa | Nessuno |
| **1** | Astrazione `arch_family`, shape densa, dispatch | Bassa-Media | Basso (rami separati, invariante test-vectors) |
| **2** | Loader dense: mapping metadati `<ns>.*`, validazioni layout condizionali (§2.1–2.3) | Media | Basso |
| **3** | **Grafo dense su GPU** (`ds4_metal.m`, poi `ds4_cuda.cu`): attention standard + FFN SwiGLU, riuso kernel base, skip MLA/indexer/MoE/Sinkhorn | **Alta** | Medio |
| **4** | KV cache normale per dense; SSD-streaming/KV-on-disk gestiti o disabilitati con messaggio; CLI/server uniformi | Media | Basso |
| **5** | Altre famiglie dense (Mistral, Gemma, Phi): solo nuova shape + quirk | Bassa | Basso |

**Collo di bottiglia:** Fase 3 (grafo GPU). Tutto il resto è refactoring
verificabile.

### Criteri di accettazione per fase
- **F1:** GGUF DeepSeek caricano e generano logit **identici** a `main` (regressione
  bit-per-bit sui `tests/test-vectors/`). Una shape densa fittizia viene
  riconosciuta dal dispatch.
- **F2:** un GGUF Qwen2/Llama 7B arriva a "caricato" senza `ds4_die`.
- **F3:** parità token-by-token vs `llama.cpp` su prompt corto, poi long-context.
- **F4:** `./ds4 -m piccolo.gguf` e `./ds4 -m ds4flash.gguf` funzionano senza flag
  speciali.

---

## 4b. Ricognizione Fase 3 — punti d'innesto del grafo (preliminare)

- L'orchestrazione del forward per-layer è **host-driven in `ds4.c`**, non un
  singolo "build graph" in `ds4_metal.m`. `ds4_metal.m` (~27k LOC) è gestione
  buffer/pipeline + dispatch dei kernel.
- Builder per-layer DeepSeek in `ds4.c` (da rendere condizionali / affiancare con
  un path denso): `layer_attn_pre_one` (`:6660`), `layer_attn_norm_one` (`:6899`),
  `layer_q_projection_with_lora_one` (`:6940`, **MLA**), `layer_kv_projection_normed_one`
  (`:6959`), `rope_tail_layer_inplace` (`:7084`).
- Bind dei pesi per-layer: `weights_bind_layer` (`ds4.c:4199`) — qui vanno mappati
  i tensori densi (un solo `ffn_gate/up/down`, `attn_q/k/v/output`, niente expert).
- Validazione layout: `weights_validate_layout` (`ds4.c:3578`) — da affiancare con
  una versione densa.
- Kernel Metal **riusabili** già presenti: `kernel_swiglu_f32`, `kernel_rms_norm_f32_4`,
  `kernel_rms_norm_mul_f32_4`, più RoPE/matmul/flash-attention. Il path denso
  riusa attention standard (GQA) + FFN SwiGLU e **non** tocca MoE/indexer/Sinkhorn.
- Validazione: confronto token-by-token vs `llama.cpp` su un GGUF Qwen2 reale.

## 4c. Fase 3 — decomposizione in sotto-task validabili

La Fase 3 (esecuzione dense) è il blocco grosso e va spezzata in passi piccoli,
ognuno **validabile in isolamento contro llama.cpp** (oracolo). Ordine consigliato:

| # | Sotto-task | Cosa | Validazione (oracolo: llama.cpp) | Serve grafo? |
| --- | --- | --- | --- | --- |
| 3.1 ✅ | **Tokenizer Qwen2** | `ds4_pretok`+`pre_type` (da `tokenizer.ggml.pre`), `qwen2_tokenize_text()`, dispatch in `bpe_tokenize_text`, special token opzionali | **9/9 PASS** vs `llama-tokenize` (`tests/validate_qwen2_tokenizer.sh`) | No |
| 3.2 ✅ | **Bind pesi densi** | `weights_bind_{layer,output,}_dense`, campi densi in `ds4_layer_weights`, dispatch in `weights_bind`; biases QKV opzionali | **Validato** su Qwen2: `--inspect` lega 28 layer senza `ds4_die`; run → "weights bound OK" | No |
| 3.3 ✅ | **Layout validation dense** | `weights_validate_layout_dense` + `dense_expect_dims` (type-agnostic) | **Validato**: ~300 controlli dim su Qwen2 (28 layer) passano | No |
| 3.4 | **Embedding + output head** | `token_embd` lookup + `output`/`lm_head` + RMSNorm finale | logit del primo token vs llama.cpp (`--logits`) | Sì (minimo) |
| 3.5 | **Attention GQA** | attention standard (no MLA): Q/K/V, RoPE, softmax, no compressor/indexer | hidden state dopo layer 0 vs riferimento | Sì |
| 3.6 | **FFN SwiGLU dense** | `gate/up` → SiLU·mul → `down` (riuso `kernel_swiglu_f32`) | hidden state dopo FFN layer 0 | Sì |
| 3.7 | **Forward completo N layer** | Loop su tutti i layer + sampling greedy | continuazione greedy token-by-token identica a llama.cpp (`bench_dense.sh`) | Sì |
| 3.8 | **Speed bench** | `ds4-bench` sul path dense | t/s comparabili con `llama-bench` | Sì |

Principio: 3.1–3.3 non richiedono il grafo e si validano subito. 3.4 in poi
richiedono il grafo dense su Metal, costruito incrementalmente confrontando lo
stato nascosto layer-per-layer con llama.cpp. Lo strumento di validazione è
`tests/bench_dense.sh` (gate finale: 3.7 PASS = output identico all'oracolo).

## 5. Prossimo passo concreto (inizio Fase 1)

1. Aggiungere `ds4_arch_family` e i campi `arch/n_ff/meta_ns` a `ds4_shape`
   (default `DS4_ARCH_DEEPSEEK`, così Flash/PRO restano invariati).
2. Estendere `DS4_MODEL_*` macro con `DS4_ARCH` e helper di capacità
   (`ds4_has_moe()`, `ds4_has_mla()`, `ds4_has_indexer()`, `ds4_has_hc()`).
3. Convertire i 17 branch `DS4_MODEL_VARIANT ==` (§2.4) in branch di capacità
   dove possibile, lasciando intatto il comportamento DeepSeek.
4. Build `make` (Metal) + suite `tests/test-vectors/` per dimostrare zero
   regressione: questo è il gate della Fase 1.

> Nota build: `make cpu` è solo per diagnostica. Su macOS recenti il path CPU può
> crashare il kernel (vedi README): lo studio si fa **leggendo** il codice, non
> eseguendo CPU su Mac.

#!/usr/bin/env bash
# ===========================================================================
# bench_vs_llama.sh — compare ds4 vs llama.cpp on the OFFLINE (already-downloaded)
# GGUF models in gguf/, measuring prefill/decode tokens-per-second and actual RAM.
#
# For each model it runs three configs and samples peak RSS (resident set = physical
# memory the engine holds) while each runs:
#
#   1. ds4 resident   — weights pinned on the GPU (default; fastest prefill).
#   2. ds4 streaming  — DS4_DENSE_STREAM=1, no residency pin; weights faulted from
#                       SSD on demand. The SSD-streaming signal is the PREFILL t/s
#                       drop vs resident (the weights are read from SSD, not pinned);
#                       on a model that fits, RSS is similar (pages still page-cached)
#                       — the benefit is not pinning, so larger-than-resident models
#                       become runnable instead of OOMing.
#   3. llama.cpp      — llama-bench (the reference engine).
#
# t/s come from each engine's own timing. NEVER downloads: only benchmarks gguf/*.
#
# Usage: tests/bench_vs_llama.sh [N_PREDICT] [GLOB]
# ===========================================================================
set -u
cd "$(dirname "$0")/.." || exit 1

NPRED="${1:-64}"
GLOB="${2:-gguf/*.gguf}"
PROMPT="Explain in detail how a modern CPU cache hierarchy works, including L1/L2/L3 levels, set associativity, cache lines, write-back versus write-through, and the most common replacement policies used in practice."
PP_TOKENS=48     # approx prompt tokens; seeds llama-bench -p

DS4=./ds4
[ -x "$DS4" ] || { echo "ds4 binary not found (run: make ds4)"; exit 1; }
have_llama=1; command -v llama-bench >/dev/null 2>&1 || have_llama=0
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
CSV="${DS4_BENCH_OUT:-bench_vs_llama.csv}"
echo "model,engine,config,prefill_tok_s,decode_tok_s,peak_rss_gib" > "$CSV"

# Run a command in the background; sample peak RSS (sum of $1-named procs, KiB). RSS.
run_sample() {
  local pname="$1"; shift
  local peak_rss=0
  "$@" >"$TMP/out" 2>"$TMP/err" &
  local pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    local sum=0 r
    for p in $(pgrep -x "$pname" 2>/dev/null); do
      r=$(ps -o rss= -p "$p" 2>/dev/null | tr -d ' '); [ -n "$r" ] && sum=$((sum + r))
    done
    [ "$sum" -gt "$peak_rss" ] && peak_rss=$sum
    sleep 0.3
  done
  wait "$pid" 2>/dev/null
  RSS_GIB=$(LC_ALL=C awk -v k="$peak_rss" 'BEGIN{printf "%.1f", k/1048576}')   # C locale -> '.' decimal
}

row() { printf "%-26s %-15s %-12s %-12s %-9s\n" "$1" "$2" "$3" "$4" "$5"; }
row "model" "config" "prefill t/s" "decode t/s" "RSS GiB"
printf '%s\n' "--------------------------------------------------------------------------------"

shopt -s nullglob
for MODEL in $GLOB; do
  base=$(basename "$MODEL"); short=$(echo "$base" | sed 's/\.gguf$//' | cut -c1-26)

  # classify the model: dense (gemma/qwen2/llama/mistral), qwen3_next, or unsupported
  kind="dense"
  if ! "$DS4" --metal-dense-generate "$MODEL" "hi" 1 >/dev/null 2>"$TMP/probe"; then
    if grep -qi "qwen3_next" "$TMP/probe"; then kind="q3n"
    else row "$short" "(skip: unsupported)" "-" "-" "-"; continue; fi
  fi

  RSS_MODEL=$(LC_ALL=C awk -v b="$(stat -f%z "$MODEL" 2>/dev/null || echo 0)" 'BEGIN{printf "%.0f", b/1073741824}')

  if [ "$kind" = "q3n" ]; then
    # 80B MoE, ~${RSS_MODEL} GiB on disk — larger than RAM, so ALWAYS SSD-streamed (cannot be
    # pinned resident). The point: RSS stays a FRACTION of the model size (the rest is on SSD).
    run_sample ds4 "$DS4" --metal-q3n-generate "$MODEL" "$PROMPT" "$NPRED"
    pf=$(grep -aoE "prefill [0-9.]+ t/s" "$TMP/err" | grep -oE "[0-9.]+" | head -1)
    de=$(grep -aoE "gen [0-9.]+ t/s"     "$TMP/err" | grep -oE "[0-9.]+" | head -1)
    row "$short" "ds4 SSD-stream" "${pf:-?}" "${de:-?}" "$RSS_GIB / ${RSS_MODEL}"
    echo "$base,ds4,q3n SSD-stream,${pf:-},${de:-},$RSS_GIB" >> "$CSV"
    # llama.cpp can't full-offload a >RAM model; note it rather than OOM the box.
    row "$short" "llama.cpp" "-" "(needs > RAM)" "-"
    printf '%s\n' "--------------------------------------------------------------------------------"
    continue
  fi

  for cfg in "ds4 resident:DS4_DENSE_RESIDENT=1" "ds4 streaming:DS4_DENSE_STREAM=1"; do
    label=${cfg%%:*}; envv=${cfg#*:}
    run_sample ds4 env "$envv" "$DS4" --metal-dense-generate "$MODEL" "$PROMPT" "$NPRED"
    pf=$(grep -aoE "prefill [0-9.]+ t/s" "$TMP/err" | grep -oE "[0-9.]+" | head -1)
    de=$(grep -aoE "gen [0-9.]+ t/s"     "$TMP/err" | grep -oE "[0-9.]+" | head -1)
    row "$short" "$label" "${pf:-?}" "${de:-?}" "$RSS_GIB"
    echo "$base,ds4,$label,${pf:-},${de:-},$RSS_GIB" >> "$CSV"
  done

  if [ "$have_llama" = 1 ]; then
    run_sample llama-bench llama-bench -m "$MODEL" -p "$PP_TOKENS" -n "$NPRED" -ngl 99 -r 1
    # t/s is the number BEFORE the '±' in the pp/tg rows
    pf=$(grep -aE "pp${PP_TOKENS}\b" "$TMP/out" | sed 's/±.*//' | grep -aoE "[0-9]+\.[0-9]+" | tail -1)
    de=$(grep -aE "tg${NPRED}\b"     "$TMP/out" | sed 's/±.*//' | grep -aoE "[0-9]+\.[0-9]+" | tail -1)
    row "$short" "llama.cpp" "${pf:-?}" "${de:-?}" "$RSS_GIB"
    echo "$base,llama.cpp,llama-bench,${pf:-},${de:-},$RSS_GIB" >> "$CSV"
  fi
  printf '%s\n' "--------------------------------------------------------------------------------"
done

echo; echo "CSV: $CSV"
echo "Reading it:"
echo " - RSS = peak RESIDENT SET = physical RAM only (mmap'd weight pages still on SSD are NOT"
echo "   counted). So RSS is exactly the RAM load."
echo " - qwen3_next (~45 GiB) runs on this box with RSS far below its size — the rest stays on"
echo "   SSD ('RSS / model' column). That is the RAM reduction: a model bigger than RAM runs."
echo " - For models that FIT in RAM (gemma/qwen2), resident and streaming have similar RSS (no"
echo "   memory pressure to evict); the SSD-streaming signal there is the lower prefill t/s."
echo " - llama.cpp pp/tg = prefill/decode t/s; it cannot full-offload a model larger than RAM."

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

  if ! "$DS4" --metal-dense-generate "$MODEL" "hi" 1 >/dev/null 2>"$TMP/probe"; then
    if grep -qiE "not a dense|not yet|not supported|qwen3_next" "$TMP/probe"; then
      row "$short" "(skip: non-dense)" "-" "-" "-"; continue
    fi
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
echo " - SSD streaming (DS4_DENSE_STREAM) is working when 'ds4 streaming' has a much LOWER"
echo "   prefill t/s than 'ds4 resident' — the weights are faulted from SSD, not GPU-pinned."
echo "   On a model that fits, RSS stays similar; the win is not pinning, so a model too big"
echo "   to pin resident becomes runnable instead of OOMing."
echo " - RSS = peak resident set (physical memory held). llama.cpp pp/tg = prefill/decode t/s."

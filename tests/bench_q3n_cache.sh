#!/usr/bin/env bash
# ===========================================================================
# bench_q3n_cache.sh — qwen3_next (MoE) routed-expert cache: the RAM lever.
#
# The 80B-A3B model is ~48 GiB on disk but only 6/512 experts per layer are
# active per token. ds4 streams the routed experts from SSD; this benchmark
# sweeps DS4_Q3N_EXPERT_CACHE_GIB (the resident-expert cap) and reports the
# TRUE RAM load for each.
#
# Two memory numbers (both from /usr/bin/time -l):
#   * peak memory footprint  = phys_footprint = the process's NON-reclaimable
#     physical memory (dirty + wired + GPU copies). THIS is the real RAM load
#     (what jetsam / a memory limit counts). It scales with the cap.
#   * maximum resident set    = also counts CLEAN file-backed pages (the mmap'd
#     / pread'd model pages still in the unified buffer cache). Those are
#     reclaimed first under pressure, so this OVERSTATES the burden — it stays
#     ~constant regardless of the cap. (ps -o rss measures this one; do not use
#     it as the q3n RAM number.)
#
# With the cap set, experts are streamed via pread (their bytes land in the
# reclaimable buffer cache, never in the process footprint) and the cold ones
# are freed after each MoE layer, so footprint ≈ base + cap.
#
# Usage: tests/bench_q3n_cache.sh [N_PREDICT] [MODEL]
# ===========================================================================
set -u
cd "$(dirname "$0")/.." || exit 1

N="${1:-8}"
MODEL="${2:-gguf/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf}"
PROMPT="Explain how a CPU cache works."
DS4=./ds4
[ -x "$DS4" ] || { echo "ds4 binary not found (run: make ds4)"; exit 1; }
[ -f "$MODEL" ] || { echo "model not found: $MODEL"; exit 1; }
command -v /usr/bin/time >/dev/null || { echo "/usr/bin/time required"; exit 1; }
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

run() {  # $1 = label, $2 = env assignment ("" for baseline)
  env $2 /usr/bin/time -l "$DS4" --metal-q3n-generate "$MODEL" "$PROMPT" "$N" \
      >"$TMP/out" 2>"$TMP/err"
  # numbers (bytes) — parse with awk in C locale so the decimal is a dot
  local fp rss gen
  fp=$(grep -a "peak memory footprint" "$TMP/err"     | grep -aoE "[0-9]+" | head -1)
  rss=$(grep -a "maximum resident set size" "$TMP/err" | grep -aoE "[0-9]+" | head -1)
  gen=$(grep -aoE "gen [0-9.]+ t/s" "$TMP/err" | grep -oE "[0-9.]+" | head -1)
  LC_ALL=C awk -v l="$1" -v fp="${fp:-0}" -v rss="${rss:-0}" -v gen="${gen:-?}" \
    'BEGIN{printf "%-22s %8.2f %12.2f %10s\n", l, fp/1073741824, rss/1073741824, gen}'
}

printf "%-22s %8s %12s %10s\n" "config" "footGiB" "maxRSS_GiB" "gen_t/s"
printf '%s\n' "------------------------------------------------------------"
run "unbounded"        ""
run "cap 4 GiB"        "DS4_Q3N_EXPERT_CACHE_GIB=4"
run "cap 2 GiB"        "DS4_Q3N_EXPERT_CACHE_GIB=2"
run "cap 1 GiB"        "DS4_Q3N_EXPERT_CACHE_GIB=1"
echo
echo "footGiB = peak memory footprint = the real RAM load (scales with the cap)."
echo "maxRSS  = incl. reclaimable file cache; ~constant, NOT the burden (don't use ps rss here)."

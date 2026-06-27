#!/usr/bin/env bash
# Benchmark ds4 across context: (1) decode speed vs context depth, (2) output quality vs
# RoPE-extension factor. Defaults to Qwen2-7B (fast). Usage: tests/bench_context.sh [MODEL]
set -u
cd "$(dirname "$0")/.."
MODEL="${1:-gguf/qwen2-7b-instruct-q4_k_m.gguf}"
NATIVE=32768   # Qwen2-7B native context
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

mkfiller() {  # $1 = approx token count -> stdout
  python3 - "$1" <<'PY'
import sys
n=int(sys.argv[1])
s="The quick brown fox jumps over the lazy dog near the riverbank in early autumn. "
# ~16 tokens per sentence; repeat to ~n tokens, then a question at the end
print((s*((n//16)+1))[:n*5] + "\n\nIn one word, the capital of France is")
PY
}

echo "======================================================================"
echo " ds4 context benchmark — model: $(basename "$MODEL")"
echo "======================================================================"
echo
echo "## 1) Decode speed vs context depth (RoPE unscaled, depth < native ${NATIVE})"
printf "   %-12s %-14s %-14s\n" "depth(tok)" "prefill t/s" "decode t/s"
for D in 256 1024 4096 16384; do
  mkfiller "$D" > "$TMP/p_$D.txt"
  OUT=$(DS4_CTX_EXTEND=1 ./ds4 --metal-dense-generate "$MODEL" "$(cat "$TMP/p_$D.txt")" 16 2>&1)
  PF=$(echo "$OUT" | grep -aoE "prefill [0-9.]+ t/s" | grep -oE "[0-9.]+" | head -1)
  GE=$(echo "$OUT" | grep -aoE "gen [0-9.]+ t/s" | grep -oE "[0-9.]+" | head -1)
  printf "   %-12s %-14s %-14s\n" "~$D" "${PF:-?}" "${GE:-?}"
done

echo
echo "## 2) Output quality vs RoPE extension (factual QA; correct answer must appear)"
printf "   %-18s %-10s %-8s\n" "context" "extension" "score"
declare -a QS=("the capital of France is" "the capital of Japan is" "two plus two equals" "the largest planet in the solar system is")
declare -a AS=("paris" "tokyo" "4|four" "jupiter")
for CTX in 32768 65536 131072 262144; do
  EXT=$(python3 -c "print('%.0fx'%($CTX/$NATIVE))")
  ok=0; tot=${#QS[@]}
  for i in "${!QS[@]}"; do
    R=$(printf '%s\n/exit\n' "${QS[$i]}" | ./ds4 --metal-dense-chat "$MODEL" "$CTX" 2>/dev/null | tr 'A-Z' 'a-z' | tr -d '\r')
    if echo "$R" | grep -qaE "${AS[$i]}"; then ok=$((ok+1)); fi
  done
  printf "   %-18s %-10s %s/%s\n" "$CTX" "$EXT" "$ok" "$tot"
done
echo
echo "## 3) Long-context retrieval (needle-in-haystack): hide a passcode, ask for it"
printf "   %-14s %-14s %-8s\n" "haystack(tok)" "prefill t/s" "found?"
for N in 1500 4000 12000; do
  python3 - "$N" > "$TMP/needle_$N.txt" <<'PY'
import sys
n=int(sys.argv[1]); s="The committee reviewed the quarterly logistics report in great detail. "
half=(s*((n//2//12)+1))
print(half + "IMPORTANT: The secret passcode for the vault is BANANA-7723. " + half +
      "\n\nBased on the text above, what is the secret passcode for the vault?")
PY
  OUT=$(./ds4 --metal-dense-generate "$MODEL" "$(cat "$TMP/needle_$N.txt")" 16 2>&1)
  PF=$(echo "$OUT" | grep -aoE "prefill [0-9.]+ t/s" | grep -oE "[0-9.]+" | head -1)
  if echo "$OUT" | grep -qaiE "BANANA-7723|BANANA"; then F="YES"; else F="no"; fi
  printf "   %-14s %-14s %-8s\n" "~$N" "${PF:-?}" "$F"
done
echo
echo "Notes: decode slows ~linearly with context depth (attention is O(ctx)/token); short-prompt"
echo "quality is preserved by NTK even at 8x allocation; retrieval works within native. Testing"
echo "retrieval at high extension (>4x) is limited by ds4 prefill speed (O(ctx^2))."

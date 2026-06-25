#!/bin/sh
# validate_qwen2_tokenizer.sh — Fase 3.1 gate.
# Compares ds4's Qwen2 pre-tokenizer (./ds4 --dump-tokens) against llama.cpp's
# reference (llama-tokenize) on a battery of strings. PASS = identical token ids.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
MODEL="${1:-$ROOT/gguf/qwen2-7b-instruct-q4_k_m.gguf}"
DS4="$ROOT/ds4"
LTOK=$(command -v llama-tokenize || echo /opt/homebrew/bin/llama-tokenize)
TMP="$ROOT/tests/bench-out"
mkdir -p "$TMP"

# Diverse cases: words, leading spaces, digits, contractions, code indentation,
# punctuation runs, newlines, accented UTF-8.
set -- \
    "Hello world" \
    "The year 2024 was great, wasn't it?" \
    "don't I'll we've they're it's I'd we'll" \
    "    if (x == 10) { return; }" \
    "line one
line two

line four" \
    "numbers: 1234567 and 89" \
    "Città, perché? Così è." \
    "a,b;c:d!e?f...g" \
    "tab	separated	values"

ds4_ids() {
    "$DS4" --dump-tokens -m "$MODEL" -p "$1" 2>/dev/null \
        | sed -n '1p' | tr -d '[]' | tr ',' ' ' | tr -s ' '
}
llama_ids() {
    printf '%s' "$1" > "$TMP/_in.txt"
    "$LTOK" -m "$MODEL" -f "$TMP/_in.txt" --no-bos 2>/dev/null \
        | sed -n 's/^[[:space:]]*\([0-9][0-9]*\) ->.*/\1/p' | tr '\n' ' ' | tr -s ' '
}

pass=0; fail=0; i=0
for s in "$@"; do
    i=$((i+1))
    a=$(ds4_ids "$s" | xargs echo)
    b=$(llama_ids "$s" | xargs echo)
    if [ "$a" = "$b" ]; then
        pass=$((pass+1))
        printf "case %d: PASS (%s tokens)\n" "$i" "$(echo "$a" | wc -w | xargs)"
    else
        fail=$((fail+1))
        printf "case %d: FAIL\n  input : %s\n  ds4   : %s\n  llama : %s\n" \
            "$i" "$(printf '%s' "$s" | head -1)" "$a" "$b"
    fi
done
echo "----"
echo "PASS=$pass FAIL=$fail"
[ "$fail" -eq 0 ]

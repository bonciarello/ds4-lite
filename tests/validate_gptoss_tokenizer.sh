#!/bin/sh
# validate_gptoss_tokenizer.sh — gpt-oss Phase 1 gate.
# Compares ds4's o200k/gpt-4o pre-tokenizer (./ds4 --dump-tokens) against llama.cpp's
# reference (llama-tokenize) on a battery of strings. PASS = identical token ids.
# o200k specifics covered: case-aware letters (camelCase splits), \p{N}{1,3} number
# grouping, '/'-trailing symbols, attached contractions, unicode/emoji/CJK.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
MODEL="${1:-$ROOT/gguf/gpt-oss-20b-Q8_0.gguf}"
DS4="$ROOT/ds4"
LTOK=$(command -v llama-tokenize || echo /opt/homebrew/bin/llama-tokenize)
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

ds4_ids() {
    "$DS4" --dump-tokens -m "$MODEL" -p "$1" 2>/dev/null \
        | sed -n '1p' | tr -d '[]' | tr ',' ' ' | tr -s ' ' | xargs echo
}
# C-locale awk extraction (sed chokes on UTF-8 bytes under some locales).
llama_ids() {
    printf '%s' "$1" > "$TMP/in.txt"
    "$LTOK" -m "$MODEL" -f "$TMP/in.txt" --no-bos 2>/dev/null \
        | LC_ALL=C awk -F'->' '/->/{gsub(/[^0-9]/,"",$1); if($1!="")printf "%s ",$1}' | xargs echo
}

set -- \
    "Hello world" \
    "HelloWorld getUserName XMLHttpRequest" \
    "The CPUCache works; iOS and macOS too" \
    "The year 2024 had 1234567 and 89 items" \
    "don't I'll we've they're it's I'd we'll" \
    "if (x == 10) { return x**2; }  # note" \
    "path/to/file and a/b/c/d" \
    "snake_case kebab-case Mixed123Inside" \
    "a,b;c:d!e?f...g" \
    "Città, perché? Così è. café Über naïve" \
    "日本語のテスト 漢字 emoji 😀🎉" \
    'JSON {"key": "value", "n": 42}'

pass=0; fail=0; i=0
for s in "$@"; do
    i=$((i+1))
    a=$(ds4_ids "$s"); b=$(llama_ids "$s")
    if [ "$a" = "$b" ]; then
        pass=$((pass+1)); printf "case %2d: PASS (%s tokens)\n" "$i" "$(echo "$a" | wc -w | xargs)"
    else
        fail=$((fail+1)); printf "case %2d: FAIL\n  input: %s\n  ds4  : %s\n  llama: %s\n" "$i" "$s" "$a" "$b"
    fi
done
echo "----"; echo "PASS=$pass FAIL=$fail"
[ "$fail" -eq 0 ]

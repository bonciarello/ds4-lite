#!/bin/sh
# bench_dense.sh — Benchmark + correctness harness for DENSE models in ds4.
#
# Compares ds4 (dense path, available from Fase 3 onward) against llama.cpp on the
# SAME GGUF, prompt and decoding settings. llama.cpp is the reference oracle: a
# correct ds4 dense graph must reproduce its greedy continuation token-for-token,
# and its t/s give a speed baseline.
#
# Methodology mirrors the project's speed-bench: greedy decoding, fixed context,
# fixed -n. See README.md "Speed" and docs/DENSE_SUPPORT_DESIGN.md (Fase 3).
#
# Usage:
#   tests/bench_dense.sh [-m MODEL.gguf] [-p "prompt"] [-n NPREDICT] [-c CTX]
#
# Requires llama.cpp (`brew install llama.cpp`) for the reference side.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
MODEL="$ROOT/gguf/qwen2-7b-instruct-q4_k_m.gguf"
PROMPT="Once upon a time, in a small village in the mountains,"
NPREDICT=128
CTX=32768
OUTDIR="$ROOT/tests/bench-out"

while [ $# -gt 0 ]; do
    case "$1" in
        -m) MODEL=$2; shift 2 ;;
        -p) PROMPT=$2; shift 2 ;;
        -n) NPREDICT=$2; shift 2 ;;
        -c) CTX=$2; shift 2 ;;
        -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "$OUTDIR"
DS4="$ROOT/ds4"
# Newer llama.cpp splits one-shot completion into `llama-completion` (llama-cli
# is chat-only). Prefer completion, fall back to cli for older installs.
LCLI=$(command -v llama-completion || command -v llama-cli || echo /opt/homebrew/bin/llama-completion)
LBENCH=$(command -v llama-bench || echo /opt/homebrew/bin/llama-bench)

echo "== bench_dense =="
echo "model:    $MODEL"
echo "prompt:   $PROMPT"
echo "n/ctx:    $NPREDICT / $CTX (greedy)"
echo

if [ ! -f "$MODEL" ]; then
    echo "ERROR: model not found: $MODEL" >&2
    exit 1
fi

# ---- Reference side: llama.cpp ---------------------------------------------
if [ ! -x "$LCLI" ] && ! command -v llama-cli >/dev/null 2>&1; then
    echo "llama.cpp not found. Install it with: brew install llama.cpp" >&2
    echo "(reference + speed baseline are skipped)" >&2
else
    echo "--- [reference] llama.cpp greedy continuation ---"
    # --temp 0 + top-k 1 = deterministic greedy; -ngl 99 = full GPU offload.
    # llama-completion is one-shot (exits after -n tokens); no chat mode to hang.
    "$LCLI" -m "$MODEL" -p "$PROMPT" -n "$NPREDICT" -c "$CTX" \
        --temp 0 --top-k 1 -ngl 99 2>/dev/null \
        | tee "$OUTDIR/llama_out.txt"
    echo
    echo "--- [reference] llama.cpp speed (llama-bench) ---"
    "$LBENCH" -m "$MODEL" -p 512 -n 128 2>/dev/null | tee "$OUTDIR/llama_bench.txt" || true
    echo
fi

# ---- Candidate side: ds4 dense ---------------------------------------------
echo "--- [candidate] ds4 dense continuation ---"
if [ ! -x "$DS4" ]; then
    echo "ds4 binary not built. Run: make" >&2
else
    # ds4 currently stops at shape construction for dense models (pre-Fase 3).
    # Once the dense graph lands, this produces a real continuation to diff.
    if "$DS4" -m "$MODEL" -p "$PROMPT" -n "$NPREDICT" --ctx "$CTX" --nothink \
            > "$OUTDIR/ds4_out.txt" 2>"$OUTDIR/ds4_err.txt"; then
        cat "$OUTDIR/ds4_out.txt"
        echo
        echo "--- [diff] ds4 vs llama.cpp greedy continuation ---"
        if [ -f "$OUTDIR/llama_out.txt" ]; then
            if diff -u "$OUTDIR/llama_out.txt" "$OUTDIR/ds4_out.txt" >/dev/null; then
                echo "PASS: ds4 dense matches llama.cpp greedy output exactly."
            else
                echo "DIFF: outputs differ (expected during Fase 3 bring-up):"
                diff -u "$OUTDIR/llama_out.txt" "$OUTDIR/ds4_out.txt" | head -40 || true
            fi
        fi
    else
        echo "ds4 cannot run this dense model yet:"
        grep -i "dense\|recognized\|graph" "$OUTDIR/ds4_err.txt" | head -4 || \
            tail -3 "$OUTDIR/ds4_err.txt"
        echo
        echo "=> Fase 3 (dense execution graph) not yet implemented."
        echo "   The reference baseline above is ready to validate against."
    fi
fi

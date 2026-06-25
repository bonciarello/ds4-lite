#!/bin/sh
# bench_dense.sh — Speed + correctness benchmark for the DENSE path vs llama.cpp.
#
# Metrics follow the standard prefill/decode split (see llama-bench): prefill t/s
# (prompt processing) and decode t/s (token generation), plus TTFT. ds4's dense
# numbers come from `./ds4 --metal-dense-generate` (instrumented); the reference
# comes from `llama-bench` (pp/tg) and `llama-simple` (greedy correctness).
#
# NOTE: ds4's dense forward processes the prompt one token at a time and copies
# weights to the GPU on first use, so the *first* run's prefill/TTFT includes a
# one-time ~weights-size warmup. We do a warmup run first and report the second.
#
# Usage: tests/bench_dense.sh [-m MODEL.gguf] [-p "prompt"] [-n NPREDICT]
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
MODEL="$ROOT/gguf/qwen2-7b-instruct-q4_k_m.gguf"
PROMPT="The capital of France is"
NPREDICT=32
OUTDIR="$ROOT/tests/bench-out"; mkdir -p "$OUTDIR"
DS4="$ROOT/ds4"
LBENCH=$(command -v llama-bench   || echo /opt/homebrew/bin/llama-bench)
LSIMPLE=$(command -v llama-simple  || echo /opt/homebrew/bin/llama-simple)

SWEEP=0
while [ $# -gt 0 ]; do
    case "$1" in
        -m) MODEL=$2; shift 2 ;;
        -p) PROMPT=$2; shift 2 ;;
        -n) NPREDICT=$2; shift 2 ;;
        --sweep) SWEEP=1; shift ;;
        -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done
[ -f "$MODEL" ] || { echo "ERROR: model not found: $MODEL" >&2; exit 1; }
[ -x "$DS4" ]   || { echo "ERROR: build ds4 first (make)" >&2; exit 1; }

echo "== bench_dense =="
echo "model:  $MODEL"
echo "prompt: \"$PROMPT\"  n_predict: $NPREDICT"
echo

# ---- ds4 dense speed (warmup, then measured) -------------------------------
echo "--- ds4 dense speed ---"
"$DS4" --metal-dense-generate "$MODEL" "$PROMPT" "$NPREDICT" >/dev/null 2>"$OUTDIR/ds4_warm.txt" || true
"$DS4" --metal-dense-generate "$MODEL" "$PROMPT" "$NPREDICT" >"$OUTDIR/ds4_out.txt" 2>"$OUTDIR/ds4_metrics.txt" || true
DS4_METRICS=$(grep "dense metrics" "$OUTDIR/ds4_metrics.txt" | tail -1)
echo "  ${DS4_METRICS:-(no metrics — run failed?)}"
echo

# ---- llama.cpp speed reference (llama-bench) -------------------------------
echo "--- llama.cpp speed (llama-bench pp512/tg128) ---"
if command -v "$(basename "$LBENCH")" >/dev/null 2>&1 || [ -x "$LBENCH" ]; then
    "$LBENCH" -m "$MODEL" -p 512 -n 128 2>/dev/null | grep -E "qwen|llama|model|t/s|pp512|tg128" | tail -4 || true
else
    echo "  llama-bench not found (brew install llama.cpp)"
fi
echo

# ---- correctness: greedy continuation must match llama-simple --------------
echo "--- correctness: ds4 greedy vs llama-simple (raw) ---"
DS4_CONT=$(sed -n '2p' "$OUTDIR/ds4_out.txt")
if [ -x "$LSIMPLE" ] || command -v "$(basename "$LSIMPLE")" >/dev/null 2>&1; then
    LL_FULL=$("$LSIMPLE" -m "$MODEL" -n "$NPREDICT" "$PROMPT" 2>/dev/null | tr -d '\n')
    # strip the echoed prompt from llama-simple's "prompt+continuation" output
    LL_CONT=$(printf '%s' "$LL_FULL" | sed "s|^.*${PROMPT}||")
    printf "  ds4   :%s\n" "$DS4_CONT"
    printf "  llama :%s\n" "$LL_CONT"
    # compare ignoring leading space differences
    a=$(printf '%s' "$DS4_CONT" | sed 's/^ *//'); b=$(printf '%s' "$LL_CONT" | sed 's/^ *//')
    case "$b" in
        "$a"*|*"$a"*) echo "  => MATCH (ds4 is a prefix of / equals llama-simple greedy)" ;;
        *) echo "  => DIFFER (inspect above)" ;;
    esac
else
    echo "  llama-simple not found"
fi

# ---- optional sweep: t/s vs context, CSV + SVG (project speed-bench format) -
if [ "$SWEEP" -eq 1 ]; then
    echo
    echo "--- sweep: decode t/s vs context (CSV + SVG) ---"
    CSV="$OUTDIR/dense_sweep.csv"; rm -f "$CSV"
    # warmup so the first measured row is not inflated by the GPU weight upload
    "$DS4" --metal-dense-generate "$MODEL" "$PROMPT" 8 >/dev/null 2>&1 || true
    for N in 16 48 96 160 256; do
        DS4_BENCH_CSV="$CSV" "$DS4" --metal-dense-generate "$MODEL" "$PROMPT" "$N" >/dev/null 2>&1 || true
        echo "  n_predict=$N done"
    done
    echo "  CSV: $CSV"
    cat "$CSV"
    if python3 "$ROOT/speed-bench/plot_speed.py" "$CSV" --title "Qwen2 dense t/s (ds4)" -o "$OUTDIR/dense_sweep_ts.svg" 2>/dev/null; then
        echo "  SVG: $OUTDIR/dense_sweep_ts.svg"
    fi
fi

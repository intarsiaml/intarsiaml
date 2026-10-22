#!/usr/bin/env bash
# M2 (EXL3) end to end speed + accuracy of the CUDA prefill GEMM on spark1, inside a fleetq slot (non-exclusive, vLLM up):
#   fleetq acquire --host spark1 --mem 25 --label "M2 gemm e2e" --expect 60 --wait; FLEETQ_JOB=<id> scripts/m2-exl3/e2e.sh <steps>
# STEPS (comma list, default bench,acc): build (scripts/cuda-decode/build.sh of THIS tree into ~/intarsia-decfuse/$TREE; the model needs
# M1's dense FFN, so run it from the m2/exl3-on-m1 worktree), bench (intarsia serve + `intarsia bench speed` at 128 / 1024 / 4096, 128
# forced output tokens), acc (intarsia accuracy against the bf16 golden at --prefill-chunk 30, so prompts cross chunk boundaries, and at
# the default chunk).
# Env: TREE (default m2-exl3-e2e), LABEL (result label / log suffix, default after), OUT (log dir), BENCH_EXTRA (extra env for the
# server, e.g. "IK_X=1"), CHUNKS (accuracy chunk sizes, default "30 default").
# The model: the 4.0 bpw EXL3 Qwen3-8B with its plain F16 tensors cast to bf16 (see scripts/m2-exl3-e2e/run.sh convert).
set -euo pipefail
: "${FLEETQ_JOB:?run under a fleetq slot (FLEETQ_JOB)}"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$S/../.." && pwd)"
HOST=${HOST:-spark1}
TREE=${TREE:-m2-exl3-e2e}
LABEL=${LABEL:-after}
STEPS=${1:-${STEPS:-bench,acc}}
OUT=${OUT:-$REPO/docs/results/2026-10-21-m2-exl3/raw}
CHUNKS=${CHUNKS:-30 default}
R='~/intarsia-decfuse'
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
CONV='$HOME/models-m2/qwen3-8b-exl3-4.0bpw-bf16plain'
GOLD_BF16='$HOME/intarsiaml-golden/qwen3-8b-fp32.safetensors'
BIN="$R/$TREE/target/acc/release/intarsia"
PORT=${PORT:-18471}
mkdir -p "$OUT"
rsh() { ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
has() { [[ ",$STEPS," == *",$1,"* ]]; }

if has build; then
  "$REPO/scripts/cuda-decode/build.sh" "$TREE" 2>&1 | tee "$OUT/cuda-gemm-e2e-build.log"
fi
if has bench; then
  echo "== serve + bench speed ($LABEL)"
  # one ssh: start the server, wait for /health, run the sweep, stop the server (also on error)
  rsh "cd $R/$TREE && mkdir -p reports-gemm && ( $BIN serve --model $CONV --backend cuda --host 127.0.0.1 --port $PORT --served-model-name m2e2e --max-seq 5000 --profile-every 0 --no-supervisor --no-ui > reports-gemm/serve-$LABEL.log 2>&1 & echo \$! > reports-gemm/serve.pid );
       trap 'kill \$(cat reports-gemm/serve.pid) 2>/dev/null; true' EXIT;
       for i in \$(seq 1 240); do curl -fsS http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; sleep 2; done;
       $BIN bench speed --url http://127.0.0.1:$PORT --model m2e2e --sizes 128,1024,4096 --gen 128 --reps 3 --warmup 1 --thinking off --label gemm-$LABEL --out-dir reports-gemm" 2>&1 | tee "$OUT/cuda-gemm-e2e-bench-$LABEL.log" | tail -30
fi
if has acc; then
  for c in $CHUNKS; do
    arg=""; [[ $c != default ]] && arg="--prefill-chunk $c"
    echo "== accuracy vs the bf16 golden ($LABEL, prefill chunk $c)"
    rsh "$BIN accuracy --model $CONV --backend cuda --golden $GOLD_BF16 $arg || true" 2>&1 | tee "$OUT/cuda-gemm-e2e-accuracy-$LABEL-chunk-$c.log" | tail -25
  done
fi
echo "steps done: $STEPS"

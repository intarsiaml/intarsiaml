#!/usr/bin/env bash
# W3 (spark1, vLLM down, inside scripts/cuda-decode/window.sh): 35B `bench speed` A/B of the grouped MoE accumulation fix
# (docs/results/2026-10-17-moe-mma-small-t.md): base = origin/main (tree moe-mma-base), dev = the fix (tree moe-mma-dev),
# three rounds, the arm order alternates per round, prefill 1024 / 4096 / 16384 and decode 128. One round is
# scripts/cuda-decode/w-ab.sh (a fresh `intarsia serve` per arm, the bench is the only client, --warmup 1 --thinking off
# --require-graphs).
#   Env: ROUNDS (3) REPS (3) SIZES (1024,4096,16384) GEN (128) BUDGET_MIN (15)
set -u
P=$HOME/intarsia-decfuse
OUTD=$P/out/moe-mma-ab
DEADLINE=$(( $(date +%s) + ${BUDGET_MIN:-15} * 60 ))
left() { echo $(( (DEADLINE - $(date +%s)) / 60 )); }
for r in $(seq 1 ${ROUNDS:-3}); do
  if (( $(left) < 4 )); then echo "SKIP round $r: $(left) min left"; continue; fi
  if (( r % 2 == 1 )); then arms="base:moe-mma-base dev:moe-mma-dev"; else arms="dev:moe-mma-dev base:moe-mma-base"; fi
  echo "== round $r ($(left) min left) $(date +%T): $arms"
  SIZES=${SIZES:-1024,4096,16384} GEN=${GEN:-128} REPS=${REPS:-3} OUT=$OUTD bash $P/moe-mma-dev/scripts/cuda-decode/w-ab.sh 35b $r $arms
done
echo "== w3 done $(date +%T), $(left) min left"

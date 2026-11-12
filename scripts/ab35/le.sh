#!/usr/bin/env bash
# ab35: the identical localeval suite for one engine arm. usage: le.sh <label> <url> <model> <recipe> <speed|all>
# Runs inside a fleetq slot (FLEETQ_JOB set), so localeval does not queue again. Raw output in docs/results/.../raw/.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
LABEL=$1; URL=$2; MODEL=$3; RECIPE=$4; MODE=$5
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
OUT=$REPO/docs/results/2026-11-12-ab35-intarsia-vs-vllm/raw; mkdir -p "$OUT"
cd /opt/code/deepeval
COMMON=(--url "$URL" --model "$MODEL" --thinking off --structured prompt --wait-idle 600 --recipe "$RECIPE")
keep() { # keep <out-log> <label>: copy the run's summary/result next to the log
  local rid; rid=$(grep -aoE 'results/[0-9]{8}-[^ ]*' "$1" | head -1 | sed 's#results/##; s#/.*##')
  echo "$2 run_id=$rid" | tee -a "$OUT/index.txt"
  [[ -n $rid && -d results/$rid ]] && { mkdir -p "$OUT/$rid"; cp results/$rid/summary.md results/$rid/result.json "$OUT/$rid/" 2>/dev/null; }
}
echo "== $LABEL speed $(date +%T)"
uv run localeval speed "${COMMON[@]}" --sizes 128,1024,4096,16384 --gen 256 --reps 3 --concurrency 2 --label "ab35-$LABEL-speed" > "$OUT/$LABEL-speed.out" 2>&1
tail -3 "$OUT/$LABEL-speed.out"; keep "$OUT/$LABEL-speed.out" "$LABEL speed"
if [[ $MODE == all ]]; then
  echo "== $LABEL gsm8k $(date +%T)"
  uv run localeval quality "${COMMON[@]}" --bench gsm8k --limit 100 --max-tokens 1024 --nonce --pin-gsm8k --label "ab35-$LABEL-gsm8k" > "$OUT/$LABEL-gsm8k.out" 2>&1
  tail -3 "$OUT/$LABEL-gsm8k.out"; keep "$OUT/$LABEL-gsm8k.out" "$LABEL gsm8k"
  echo "== $LABEL ifeval $(date +%T)"
  uv run localeval quality "${COMMON[@]}" --bench ifeval --limit 25 --max-tokens 1024 --nonce --label "ab35-$LABEL-ifeval" > "$OUT/$LABEL-ifeval.out" 2>&1
  tail -3 "$OUT/$LABEL-ifeval.out"; keep "$OUT/$LABEL-ifeval.out" "$LABEL ifeval"
fi
echo "== $LABEL done $(date +%T)"

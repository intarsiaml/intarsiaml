#!/usr/bin/env bash
# Lane dfh: radiance (the production engine on tatooine, its own DFlash2 drafter for the same 27B, tp=2) on the lane's category prompts,
# read-only (radiance stays up), from the Mac, for reference beside the intarsia arms. One fleetq job, both cards quiet:
#   fleetq run --host tatooine-r9700,tatooine-r9700-2 --quiet gpu,cpu --expect 25 --label "dfh radiance ref" -- env W=rad1 scripts/dfh/radiance-ref.sh
# Same files as cat-window.sh (scripts/dfh/cats/*.jsonl), localeval quality --judge none, 512 forced tokens (ignore_eos + min_tokens),
# temperature 0 (localeval sends 0). radiance's /metrics counters (draft tokens / accepted, decode steps and tokens, per-position
# acceptance) are read before and after every file, so acceptance and tokens per decode step come out per category.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}" "${W:?W=<run name>}"
GEN=${GEN:-512}; URL=http://tatooine:8080
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
OUT=$REPO/docs/results/2026-10-26-dfh-hip-dflash/raw/$W; mkdir -p "$OUT"
say() { echo "$(date +%T) $*" | tee -a "$OUT/steps.log"; }
metric() { curl -s -m 5 "$URL/metrics" | grep -v '^#' | grep -E 'radiance:(draft_tokens|draft_accepted|engine_steps_decode|decode_tokens)_total|draft_position_acceptance_ratio|vllm:generation_tokens_total' | sed 's/{model_name="[^"]*",engine="0"/{/' | tr '\n' ' '; }
model=$(curl -s -m 5 "$URL/v1/models" | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
say "radiance answers, model $model"
printf '[targets.dfh-radiance]\nurl = "%s/v1"\nmodel = "%s"\nextra_body = { ignore_eos = true, min_tokens = %s }\n' "$URL" "$model" "$GEN" > "$OUT/targets-radiance.toml"
RECIPE="radiance (tatooine, ~/.local/opt/radiance/radiance-native.sh, qwen3.8-27b-mxfp4.rad, --tp 2, its DFlash2 drafter, --num-speculative-tokens 7), read-only reference, lane dfh"
for r in $(seq 1 "${ROUNDS:-1}"); do
  for f in "$HERE"/cats/*.jsonl; do
    c=$(basename "$f" .jsonl); m0=$(metric)
    ( cd /opt/code/deepeval && LOCALEVAL_TARGETS="$OUT/targets-radiance.toml" uv run localeval quality dfh-radiance --thinking off --label "dfh-$W-radiance-$c-r$r" \
        --recipe "$RECIPE" --dataset "$f" --judge none --max-tokens "$GEN" --wait-idle 60 ) > "$OUT/q-radiance-$c-r$r.log" 2>&1 < /dev/null
    rc=$?; m1=$(metric)
    echo "radiance $c r$r before: $m0" >> "$OUT/spec-metrics.txt"; echo "radiance $c r$r after:  $m1" >> "$OUT/spec-metrics.txt"
    say "round $r arm radiance $c: localeval exit $rc $(grep -aoE 'results/[^ ]+' "$OUT/q-radiance-$c-r$r.log" | tail -1)"
  done
done
say "done"

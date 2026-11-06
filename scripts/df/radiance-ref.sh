#!/usr/bin/env bash
# Lane df (27B DFlash gap): Radiance 1.17.0 (production service on tatooine, tp2, its DFlash2 drafter k7) as it runs, read-only,
# from the Mac. Copy of scripts/dfh/radiance-ref.sh plus the localeval filler sweep (scoreboard's 1k/4k) and the natural-v1
# speed-spec cell. /metrics counters before and after each step give tokens per decode step and seconds per decode step.
#   fleetq run --host tatooine-r9700,tatooine-r9700-2 --quiet gpu,cpu --expect 60 --label "df radiance ref" -- env W=rad1 scripts/df/radiance-ref.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}" "${W:?W=<run name>}"
GEN=${GEN:-512}; URL=http://tatooine:8080
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
OUT=$REPO/docs/results/2026-11-06-27b-dflash-gap/raw/$W; mkdir -p "$OUT"
say() { echo "$(date +%T) $*" | tee -a "$OUT/steps.log"; }
metric() { curl -s -m 5 "$URL/metrics" | grep -v '^#' | grep -E 'radiance:(draft_tokens|draft_accepted|engine_steps_decode|engine_step_seconds_decode|engine_steps_ahead|decode_tokens)_total' | sed 's/{model_name="[^"]*",engine="0"}//' | tr '\n' ' '; }
mark() { echo "$1 before: $2" >> "$OUT/spec-metrics.txt"; echo "$1 after:  $(metric)" >> "$OUT/spec-metrics.txt"; }
model=$(curl -s -m 5 "$URL/v1/models" | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
say "radiance answers, model $model"
printf '[targets.df-radiance]\nurl = "%s/v1"\nmodel = "%s"\nextra_body = { ignore_eos = true, min_tokens = %s }\n' "$URL" "$model" "$GEN" > "$OUT/targets-radiance.toml"
RECIPE="radiance 1.17.0 (tatooine, ~/.local/opt/radiance/radiance-native.sh, qwen3.8-27b-mxfp4.rad, --tp 2, its DFlash2 drafter, --num-speculative-tokens 7, --kv-cache-dtype fp8, --tp-wire wht6, extra --prefill-interleave 2 --checkpoint-slots 160 --draft-short-from 8), read-only reference, lane df"
LE() { ( cd /opt/code/deepeval && LOCALEVAL_TARGETS="$OUT/targets-radiance.toml" uv run localeval "$@" ) < /dev/null; }
if [[ ${SKIP_SPEED:-0} != 1 ]]; then
  TOK="--tokenizer-repo Vontra/Qwen3.8-27B-MLX-4bit --tokenizer-revision 70ae7fac63274ff2eac54152031433374cb80f2f"
  # speed-plain = the scoreboard's filler prompts (decode_1k 200.2), speed-spec = natural-v1 text; same suite flags as MB run 20261104-160305
  for cell in speed-plain speed-spec; do
    m0=$(metric)
    LE suite --url "$URL/v1" --model "$model" --thinking off $TOK --lane spec:dflash --drafter dflash2 --kv-dtype fp8 --group-size 32 --max-num-seqs 16 --power-mode default \
      --tier adhoc --cells $cell --label "df-$W-radiance-$cell" --recipe "$RECIPE" --wait-idle 120 > "$OUT/suite-$cell.log" 2>&1
    say "suite $cell exit $?: $(grep -aoE 'results/[^`]+' "$OUT/suite-$cell.log" | tail -1)"; mark "$cell" "$m0"
  done
fi
for r in $(seq 1 "${ROUNDS:-1}"); do
  for f in "$REPO"/scripts/dfh/cats/*.jsonl; do
    c=$(basename "$f" .jsonl); m0=$(metric)
    LE quality df-radiance --thinking off --label "df-$W-radiance-$c-r$r" --recipe "$RECIPE" --dataset "$f" --judge none --max-tokens "$GEN" --wait-idle 60 > "$OUT/q-radiance-$c-r$r.log" 2>&1
    rc=$?; mark "radiance $c r$r" "$m0"
    say "round $r radiance $c: localeval exit $rc $(grep -aoE 'results/[^ ]+' "$OUT/q-radiance-$c-r$r.log" | tail -1)"
  done
done
say "done"

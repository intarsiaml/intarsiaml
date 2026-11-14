#!/usr/bin/env bash
# soak2 lane: soak of the shipping configuration on spark1 (vLLM up). usage: soak.sh smoke | soak.sh full [soak-min=36]
#   fleetq run --host spark1 --quiet gpu,cpu --mem 55 --expect 50 --label "soak2: soak" -- bash scripts/soak2/soak.sh full
# ONE server for the whole window: `serve --max-batch 2 --prefix-store on` (default 8 GiB host budget, bf16 KV, MTP default on CUDA),
# no env variables. Phases: (0) smoke: startup lines, one lone greedy and one lone T=1.0 request must each raise the spec counters;
# (1) probe at K=1: one greedy and one T=1.0 replay (2 conversations x 6 turns), spec counter diffs per class (acceptance by class);
# (2) soak: replay2 K=2 interleaved, mixed greedy / T=1.0 per conversation, fresh seeds 61, 62, ... for soak-min minutes.
set -uo pipefail
MODE=${1:-full}; SOAKMIN=${2:-36}
WLOG=window-$MODE
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
[[ $MODE == smoke || -f $OUT/replay-plan.json ]] || { echo "missing $OUT/replay-plan.json"; exit 1; }
prep_box
spec() { awk '/^intarsia_spec_(steps|proposed_tokens|accepted_tokens)_total/ {printf "%s=%s ", $1, $2}' "$1"; }
chat1() { # chat1 <temperature> <tag>: one lone request (no prefix reuse: random filler first), 120 tokens
  curl -s --max-time 120 "$URL/chat/completions" -H 'Content-Type: application/json' -d "{\"model\":\"$MODEL\",\"temperature\":$1,\"max_tokens\":120,\"chat_template_kwargs\":{\"enable_thinking\":false},\"messages\":[{\"role\":\"user\",\"content\":\"[$2 $RANDOM$RANDOM] Write a short story about a lighthouse keeper who finds a message in a bottle, in about 120 words.\"}]}" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print('finish', d['choices'][0]['finish_reason'], 'completion_tokens', d['usage']['completion_tokens'])"
}
start_serve S --max-batch 2 --prefix-store on || exit 1
# ---- (0) smoke
log "smoke: startup lines: $(rsh "grep -E 'prefix store|MTP|spec' $W/serve-S.log | head -8 | tr '\n' '|'")"
scrape "$OUT/smoke.m0"; log "spec before: $(spec "$OUT/smoke.m0")"
log "smoke greedy: $(chat1 0 g)"; scrape "$OUT/smoke.m1"; log "spec after lone greedy: $(spec "$OUT/smoke.m1")"
log "smoke T=1.0: $(chat1 1.0 s)"; scrape "$OUT/smoke.m2"; log "spec after lone T=1.0: $(spec "$OUT/smoke.m2")"
p0=$(awk '/^intarsia_spec_proposed_tokens_total/ {print $2}' "$OUT/smoke.m1"); p1=$(awk '/^intarsia_spec_proposed_tokens_total/ {print $2}' "$OUT/smoke.m2")
if [[ -z $p0 || -z $p1 ]] || ! python3 -c "import sys; sys.exit(0 if float('$p1') > float('$p0') else 1)"; then
  log "SMOKE FAILED: a lone T=1.0 request did not raise intarsia_spec_proposed_tokens_total ($p0 -> $p1)"; finish_serve S; exit 2
fi
log "smoke OK: lone T=1.0 request drafted ($p0 -> $p1 proposed)"
[[ $MODE == smoke ]] && { finish_serve S; exit 0; }
# ---- (1) probe: acceptance by class at one stream
for cls in greedy sampled; do
  seed=$([[ $cls == greedy ]] && echo 5 || echo 6)
  scrape "$OUT/serve-S.metrics.before-P-$cls"
  replay2 "P-$cls" interleaved $seed 1 "$OUT/probe-trace-$cls-seed$seed.json" --convs 2 --turns 6 --temp-mode $cls
  scrape "$OUT/serve-S.metrics.after-P-$cls"
  log "probe $cls: $(python3 "$SC/metric_diff.py" "$OUT/serve-S.metrics.before-P-$cls" "$OUT/serve-S.metrics.after-P-$cls")"
done
[[ -s $OUT/P-greedy-interleaved-seed5-k1.summary.json && -s $OUT/P-sampled-interleaved-seed6-k1.summary.json ]] || { log "ABORT: a probe replay wrote no summary"; finish_serve S; exit 3; }
# ---- (2) soak
SOAK_START=$(date +%s); SOAK_END=$(( SOAK_START + SOAKMIN*60 ))
log "soak: no new replay after $(date -r $SOAK_END +%T) (target ${SOAKMIN} min); vLLM counters $(vcount)"
seed=61; fails=0
while (( $(date +%s) < SOAK_END )); do
  replay2 "S$seed" interleaved $seed 2 "$OUT/soak-trace-seed$seed.json" --temp-mode mixed
  if [[ ! -s $OUT/S$seed-interleaved-seed$seed-k2.summary.json ]]; then  # a replay that wrote no summary failed (client error, hang cut by the alarm)
    fails=$((fails+1)); log "replay S$seed produced no summary (consecutive failures $fails)"; (( fails >= 2 )) && { log "ABORT: two failed replays in a row"; break; }
  else fails=0; fi
  scrape "$OUT/serve-S.metrics.after-seed$seed"
  seed=$((seed+1))
  rsh "kill -0 $SPID" 2>/dev/null || { log "SERVER GONE"; break; }
done
log "soak done: $((seed-61)) replays in $(( ($(date +%s)-SOAK_START)/60 )) min; vLLM counters $(vcount)"
finish_serve S

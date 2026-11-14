#!/usr/bin/env bash
# restore-acc lane, window D (spark2, vLLM down): soak at the store's DEFAULT host budget. usage: window_d.sh [cap-min=50] [soak-min=30]
# Runs INSIDE: FLEETQ_OWNER=claude-restore-acc fleetq run --host spark2 --takes-down retro --quiet gpu,cpu --mem 58 --expect 55 ...
# One serve (production shape: --max-batch 2 --prefix-store on, no host_mb, bf16 KV, defaults otherwise, prefill pause default), then
# replay2 K=2 interleaved over fresh seeds 31, 32, ... each recording its own trace from the server's replies; 60 s sampler on the box.
set -uo pipefail
CAPMIN=${1:-50}; SOAKMIN=${2:-30}
WLOG=window-d
SOAK_OUT=${SOAK_OUT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)/docs/results/2026-11-13-restore-acc/raw}
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
[[ -f $OUT/replay-plan.json ]] || { echo "missing $OUT/replay-plan.json (copy it from docs/results/2026-11-13-cutover-soak/raw)"; exit 1; }
take_vllm_down "$CAPMIN"
[[ $(rsh 'systemctl is-active earlyoom') == active ]] || log "WARNING: earlyoom NOT active (memwatch kill line stays 10 GB; brief says keep 15)"
HARD_END=$(( T0 + CAPMIN*60 - 6*60 ))
if start_serve S --max-batch 2 --prefix-store on; then
  SOAK_START=$(date +%s); SOAK_END=$(( SOAK_START + SOAKMIN*60 )); (( SOAK_END > HARD_END )) && SOAK_END=$HARD_END
  log "soak: no new replay after $(date -r $SOAK_END +%T) (target ${SOAKMIN} min, hard end by cap)"
  seed=31
  while (( $(date +%s) < SOAK_END )); do
    replay2 "S$seed" "$URL" interleaved $seed 2 "$OUT/soak-trace-seed$seed.json"
    curl -sf "http://$HN:$PORT/metrics" > "$OUT/serve-S.metrics.after-seed$seed"
    seed=$((seed+1))
  done
  log "soak done: $((seed-31)) replays in $(( ($(date +%s)-SOAK_START)/60 )) min"
  # extra (only if it fits before the cap): the GSM8K shared-prefix arm at concurrency 2 on this long-running, full store (arm name S-W2)
  if [[ -f $OUT/gsm8k-inputs.json ]] && (( CAPMIN*60 - ($(date +%s)-T0) > 9*60 )); then
    curl -sf "http://$HN:$PORT/metrics" > "$OUT/serve-S.metrics.before-SW2"
    ( cd /opt/code/deepeval && uv run python "$SC/gsm8k_prefix.py" --url "$URL" --model guidance --inputs "$OUT/gsm8k-inputs.json" --arm SW2 \
        --out "$OUT" --concurrency 2 --shots 32 --limit 200 --max-tokens 512 ) > "$OUT/SW2-gp.out" 2>&1
    tail -1 "$OUT/SW2-gp.out" | cut -c1-900 | tee -a "$OUT/${WLOG}.log"
    curl -sf "http://$HN:$PORT/metrics" > "$OUT/serve-S.metrics.after-SW2"
    python3 "$SC/metric_diff.py" "$OUT/serve-S.metrics.before-SW2" "$OUT/serve-S.metrics.after-SW2" | tee "$OUT/SW2.storediff.json" | tee -a "$OUT/${WLOG}.log"
  fi
  finish_serve S
fi

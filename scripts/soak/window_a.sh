#!/usr/bin/env bash
# soak lane, window A (spark2, vLLM down): load at K=3 and K=4, then the soak. usage: window_a.sh [cap-min=50] [soak-min=35]
# Runs INSIDE: FLEETQ_OWNER=claude-soak fleetq run --host spark2 --takes-down retro --quiet gpu,cpu --mem 58 --expect 60 ...
# Stops vLLM, restores it on ANY exit (trap) and waits for /v1/models 200. vLLM downtime = cap + about 3.5 min restart, so the
# default cap 50 keeps the window under 55 min. No new soak replay starts after cap-6 min.
#   serve 1 (production shape, defaults): replay2 interleaved K=3 seed 1, K=4 seed 2 (traces recorded by the vLLM arms: replay the same files)
#   serve 2 (--prefix-store host_mb=4096): soak = replay2 K=2 over fresh seeds 11, 12, ... (each records its own trace from the
#   server's own replies); 60 s sampler (MemAvailable, RSS, store tier bytes, hits, evictions, errors, gap histogram) on the box
set -uo pipefail
CAPMIN=${1:-50}; SOAKMIN=${2:-35}
WLOG=window-a
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
for f in replay-plan.json replay-trace-seed1.json replay-trace-seed2.json; do [[ -f $OUT/$f ]] || { echo "missing $OUT/$f (run vllm_arms.sh k3k4 first)"; exit 1; }; done
take_vllm_down "$CAPMIN"
[[ $(rsh 'systemctl is-active earlyoom') == active ]] || log "WARNING: earlyoom NOT active (memwatch kill line stays 10 GB; brief says keep 15)"

if start_serve L1 --max-batch 2 --prefix-store on; then
  replay2 L-k3 "$URL" interleaved 1 3 "$OUT/replay-trace-seed1.json"
  curl -sf "http://$HN:$PORT/metrics" > "$OUT/serve-L1.metrics.after-k3"
  replay2 L-k4 "$URL" interleaved 2 4 "$OUT/replay-trace-seed2.json"
  finish_serve L1
fi

SOAK_START=$(date +%s)
SOAK_END=$(( SOAK_START + SOAKMIN*60 )); HARD_END=$(( T0 + CAPMIN*60 - 6*60 )); (( SOAK_END > HARD_END )) && SOAK_END=$HARD_END
if start_serve S --max-batch 2 --prefix-store host_mb=4096; then
  SOAK_START=$(date +%s); SOAK_END=$(( SOAK_START + SOAKMIN*60 )); (( SOAK_END > HARD_END )) && SOAK_END=$HARD_END
  log "soak: no new replay after $(date -r $SOAK_END +%T) (target ${SOAKMIN} min, hard end by cap)"
  seed=11
  while (( $(date +%s) < SOAK_END )); do
    replay2 "S$seed" "$URL" interleaved $seed 2 "$OUT/soak-trace-seed$seed.json"
    curl -sf "http://$HN:$PORT/metrics" > "$OUT/serve-S.metrics.after-seed$seed"
    seed=$((seed+1))
  done
  log "soak done: $((seed-11)) replays in $(( ($(date +%s)-SOAK_START)/60 )) min"
  finish_serve S
fi

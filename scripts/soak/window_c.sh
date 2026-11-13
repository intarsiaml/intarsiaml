#!/usr/bin/env bash
# restore-acc lane, window C (spark2, vLLM down): GSM8K with a shared long prefix, accuracy after a store restore. usage: window_c.sh [cap-min=50]
# Runs INSIDE: FLEETQ_OWNER=claude-restore-acc fleetq run --host spark2 --takes-down retro --quiet gpu,cpu --mem 58 --expect 55 ...
#   serve 1 "W" (production shape: --max-batch 2 --prefix-store on, default host budget, bf16 KV, defaults otherwise, prefill pause default):
#      W  = gsm8k_prefix.py at concurrency 1 (first request cold and inserts, the rest restore the shared prefix)
#   serve 2 "C" (--prefix-store host_mb=1: the store is ON, so the same boundary-cut chunk schedule and captures, but the host tier
#      is 1 MB, so every insert is rejected and every lookup misses: a cold prefill of the whole prompt for every request): C at concurrency 1
#   then, only if the time left covers each (checked against C's measured wall):
#      serve 3 "O" (store OFF, --max-batch 2), concurrency 2 (optional arm)
#      serve 4 "W2" (as W; warm only from its own first request), concurrency 2 (the production stream count; a separate observation, not the gate)
set -uo pipefail
CAPMIN=${1:-50}
WLOG=window-c
SOAK_OUT=${SOAK_OUT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)/docs/results/2026-11-13-restore-acc/raw}
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
[[ -f $OUT/gsm8k-inputs.json ]] || { echo "missing $OUT/gsm8k-inputs.json"; exit 1; }
left() { echo $(( CAPMIN*60 - ($(date +%s)-T0) )); }
# gp <arm> <serve-name> <concurrency>
gp() {
  echo "== $1 gsm8k-prefix conc=$3 $(date +%T)"
  ( cd /opt/code/deepeval && uv run python "$SC/gsm8k_prefix.py" --url "$URL" --model guidance --inputs "$OUT/gsm8k-inputs.json" --arm "$1" \
      --out "$OUT" --concurrency "$3" --shots 32 --limit 200 --max-tokens 512 ) > "$OUT/$1-gp.out" 2>&1
  tail -2 "$OUT/$1-gp.out" | cut -c1-900 | tee -a "$OUT/${WLOG}.log"
  curl -sf "http://$HN:$PORT/metrics" > "$OUT/serve-$2.metrics.after-$1"
}
take_vllm_down "$CAPMIN"
[[ $(rsh 'systemctl is-active earlyoom') == active ]] || log "WARNING: earlyoom NOT active (memwatch kill line stays 10 GB; brief says keep 15)"
rsh "$BIN --version | head -1" | tee -a "$OUT/${WLOG}.log"
wall_of() { python3 -c "import json;print(int(json.load(open('$OUT/$1.summary.json'))['wall_s']))" 2>/dev/null || echo 0; }
diff_of() { python3 "$SC/metric_diff.py" "$OUT/serve-$1.metrics.before" "$OUT/serve-$1.metrics.after-$2" | tee "$OUT/$2.storediff.json" | tee -a "$OUT/${WLOG}.log"; }

# W and C are the gate pair: both at concurrency 1, run first, no skip guard
if start_serve W --max-batch 2 --prefix-store on; then
  gp W W 1; diff_of W W
  finish_serve W
fi
if start_serve C --max-batch 2 --prefix-store host_mb=1; then
  gp C C 1; diff_of C C
  finish_serve C
fi
C_WALL=$(wall_of C)
O_EST=$(( C_WALL*6/10 + 150 ))
if (( C_WALL > 0 )) && (( $(left) > O_EST + 330 )); then
  log "O: $(left) s left, estimate $O_EST s"
  if start_serve O --max-batch 2; then
    gp O O 2
    finish_serve O
  fi
else
  log "skipping O: $(left) s left, estimate $O_EST s"
fi
W2_EST=$(( $(wall_of W)*6/10 + 150 ))
if (( $(wall_of W) > 0 )) && (( $(left) > W2_EST + 330 )); then
  log "W2: $(left) s left, estimate $W2_EST s"
  if start_serve W2 --max-batch 2 --prefix-store on; then
    gp W2 W2 2; diff_of W2 W2
    finish_serve W2
  fi
else
  log "skipping W2: $(left) s left, estimate $W2_EST s"
fi

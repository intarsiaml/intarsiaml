#!/usr/bin/env bash
# flipprep gate (coordinator, lane dv27 finding): every row count a batched verify pass can have (2..=6 for N = 2, k = 2) runs on the 35B under ADMITTED KERNELS ONLY
# (serve's default; a missing admission is an engine failure: `no kernel for ... [admitted kernels only]`). Servers: F defaults (k = 2, adaptive depth), F with --spec-k 1
# (rows 2 and 4), F with --spec-k 2 --spec-k-min 1 ... . Each runs with INTARSIA_CB_TRACE=1 INTARSIA_SPEC_TRACE=1; rows of a pass = sum of (drafted + 1) of the `spec: token` lines
# between two `DecodePhase` lines (rows_hist.py). ids.py (B = 1, 2, greedy and T = 1.0) is the load. The watchdog line `engine failed` must be absent.
# Run under: fleetq run --host spark2 --quiet gpu,cpu --mem 50 --expect 25 --label "flipprep: batched-verify row counts 2..6, admitted kernels" -- scripts/flipprep/rows.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
OUT=${OUT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)/docs/results/2026-11-15-flipprep/raw/rows}; mkdir -p "$OUT"
LOGNAME_=rows; source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"; source "$S/arms.sh"
for v in ${VISITS:-k2 k1 k2min1}; do
  arm_setup F
  case $v in k2) extra="" ;; k1) extra="--spec mtp --spec-k 1" ;; k2min1) extra="--spec mtp --spec-k 2" ;; esac
  envs="INTARSIA_CB_TRACE=1 INTARSIA_SPEC_TRACE=1"; [[ $v == k2min1 ]] && envs="$envs INTARSIA_SPEC_K_MIN=1"
  if start_serve "rows-$v" "$envs" --max-batch 2 $extra; then
    python3 "$S/ids.py" "$URL" guidance "$OUT/ids-$v.json" "http://$HN:$PORT/metrics" 2>&1 | tee -a "$LOG"
    ( cd /opt/code/deepeval && uv run python "$S/speed.py" --url "$URL" --model guidance --arm "rows-$v" --streams 2 --gen 192 --temperature 0 --set code --out "$OUT/speed" --metrics "http://$HN:$PORT/metrics" ) 2>&1 | tee -a "$LOG"
    log "engine failed lines in rows-$v (must be 0): $(rsh "grep -c 'engine failed\|no kernel' ~/intarsia-flipprep/out/serve-rows-$v.log")"
    finish_serve "rows-$v"
    python3 "$S/rows_hist.py" "$OUT/serve-rows-$v.log" 2>&1 | tee -a "$LOG" "$OUT/rows-hist-$v.txt"
  fi
done

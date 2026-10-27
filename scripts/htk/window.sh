#!/usr/bin/env bash
# Lane htk, THE GPU window (<= 30 min), run on endor inside one fleetq job on tatooine that takes radiance down:
#   part A (window1.sh, on tatooine): HIP top-k tests, mutants, admission sweep, 35B device-vs-host draws, sampled step timing;
#   part B (window2.sh): served A/B of INTARSIA_DEVICE_SAMPLING=0 against =1 (token ids of seeded requests, localeval speed sampled at
#   1/4/16 streams, greedy 1 stream), visits off, on. Radiance is restarted at the end by window2.sh's exit trap (and here if B never starts).
#   fleetq run --host tatooine --takes-down radiance --mem 30 --quiet gpu,cpu --expect 30 --label "HTK GPU window" -- bash scripts/htk/window.sh
set -uo pipefail
H=$(cd "$(dirname "$0")" && pwd); OUT=${OUT:-$H/../../docs/results/2026-10-27-htk-hip-topk/raw}; mkdir -p "$OUT"
trap 'ssh tatooine "bash ~/intarsiaml-htk/src/scripts/htk/win.sh restore" < /dev/null' EXIT
echo "[win] stopping radiance $(date +%T)"
ssh tatooine "systemctl --user stop radiance; sleep 5" < /dev/null
ssh tatooine 'HELD=1 bash -s' < "$H/window1.sh" 2>&1 | tee "$OUT/part-a.txt"
ARMS="${ARMS:-off on}" LATE=99999 bash "$H/window2.sh" 2>&1 | tee "$OUT/part-b.txt"
echo "[win] done $(date +%T)"

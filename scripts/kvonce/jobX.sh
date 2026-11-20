#!/usr/bin/env bash
# kvonce follow-up: is samp64's context schedule (k=1 above 28k, plain steps above 56k) still needed with the knob on? The schedule's two regions
# measured directly next to the knob-on k=2 default: x1 = `--spec-k 1` with the knob on, x2 = knob on k=2 (drift check), sampled T=0.7 32k / 64k.
#   FLEETQ_OWNER=claude-kvonce fleetq run --host spark2 --quiet gpu,cpu --mem 55 --expect 35 --label "kvonce k1 check" -- bash scripts/kvonce/jobX.sh
D="$(dirname "${BASH_SOURCE[0]}")"
ON="INTARSIA_CUDA_ATTN_KV_ONCE=1"
bash "$D/window.sh" x1 "--spec-k 1" "$ON" 0.7:32768,65536:6
bash "$D/window.sh" x2 "" "$ON" 0.7:32768,65536:6

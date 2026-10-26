#!/usr/bin/env bash
# Lane dfh window 3 (ON tatooine, radiance down, called by window.sh): the stop condition with the probe unprofiled
# (w1 and w2 profiled every pass: LoadOptions::default has profile_every 1, so their verify timings are profiled eager passes).
#   card 0: verify_rows_real, 27B tp=1, captured verify (default): bitwise vs single-row steps at 17..16k; cost of q = 2..8 vs decode1
#   card 1: the same at 1024 with INTARSIA_VERIFY_GRAPH=0 (eager, unprofiled), for the like-for-like table
set -uo pipefail
export O=${1:?out dir}
source "$HOME/dfh/lib.sh"
D=${D:-dev}
log "w3 start, VRAM in use $(( $(vram) / 1048576 )) MiB"
clear "w3" || exit 2
E="INTARSIA_VERIFY_DIR=$M27 INTARSIA_VERIFY_ROWS=8 INTARSIA_VERIFY_LENS=17,130,1024,4000,16000" \
  mtest $D probe-27b-tp1-captured 0 1200 --test verify_rows_real -- --nocapture &
E="INTARSIA_VERIFY_DIR=$M27 INTARSIA_VERIFY_ROWS=8 INTARSIA_VERIFY_LENS=1024 INTARSIA_VERIFY_GRAPH=0" \
  mtest $D probe-27b-tp1-eager 1 900 --test verify_rows_real -- --nocapture &
wait
log "w3 done"

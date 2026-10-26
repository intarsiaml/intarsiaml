#!/usr/bin/env bash
# Lane dfh window 3 (ON tatooine, radiance down, called by window.sh): the stop condition with the probe unprofiled (w2 profiled every pass: LoadOptions::default has profile_every 1).
#   card 0: verify_rows_real on the 27B at tp=1 (captured verify; bitwise vs single-row steps at 17..16k; cost of q = 2..8 vs decode1),
#           then the same cost loop with INTARSIA_VERIFY_GRAPH=0 (eager) at 1024 only, for the like-for-like table
#   card 1: 27B tp=1 --spec off vs --spec mtp (k 2, 6) 256 ids, short and 4k (forward_verify changed: MTP's eager fallback takes the capture)
set -uo pipefail
export O=${1:?out dir}
source "$HOME/dfh/lib.sh"
D=${D:-dev}
log "w3 start, VRAM in use $(( $(vram) / 1048576 )) MiB"
clear "w3" || exit 2
(
  E="INTARSIA_VERIFY_DIR=$M27 INTARSIA_VERIFY_ROWS=8 INTARSIA_VERIFY_LENS=17,130,1024,4000,16000" \
    mtest $D probe-27b-tp1-captured 0 1200 --test verify_rows_real -- --nocapture
  E="INTARSIA_VERIFY_DIR=$M27 INTARSIA_VERIFY_ROWS=8 INTARSIA_VERIFY_LENS=1024 INTARSIA_VERIFY_GRAPH=0" \
    mtest $D probe-27b-tp1-eager 0 900 --test verify_rows_real -- --nocapture
) &
(
  for kind in short 4k; do
    gen $D 1 "27b-off-$kind" $M27 $kind
    gen $D 1 "27b-mtp2-$kind" $M27 $kind --spec mtp --spec-k 2
    gen $D 1 "27b-mtp6-$kind" $M27 $kind --spec mtp --spec-k 6
    same "gen-27b-off-$kind.ids" "gen-27b-mtp2-$kind.ids" "27B tp1 $kind ids, mtp k2 vs off"
    same "gen-27b-off-$kind.ids" "gen-27b-mtp6-$kind.ids" "27B tp1 $kind ids, mtp k6 vs off"
  done
) &
wait
log "w3 done"

#!/usr/bin/env bash
# Lane dfh window 1 (ON tatooine, radiance down, called by window.sh). The phase 1 stop condition:
#   card 0: verify_rows_real on the 27B at tp=1 (8-row verify bitwise vs the single-row steps; verify8 / decode1 cost)
#   card 1: the 27B at tp=1, 256 greedy ids, --spec off vs --spec mtp --spec-k 2 and 6 (first HIP run of --spec mtp; plain decode tok/s)
set -uo pipefail
export O=${1:?out dir}
source "$HOME/dfh/lib.sh"
D=${D:-dev}
log "w1 start, VRAM in use $(( $(vram) / 1048576 )) MiB, tree $(cat "$(T $D)/dfh-intarsia.stamp" 2>/dev/null)"
clear "w1" || exit 2
E="INTARSIA_VERIFY_DIR=$M27 INTARSIA_VERIFY_ROWS=8 INTARSIA_VERIFY_LENS=17,130,1024,4000,16000" \
  mtest $D probe-27b-tp1 0 1500 --test verify_rows_real -- --nocapture &
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
log "w1 done"

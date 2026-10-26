#!/usr/bin/env bash
# Lane dfh window 4 (ON tatooine, radiance down, called by window.sh): DFlash2 bring-up at tp=1 and the HIP no-op gate.
#   card 0 (tree D): GPU admission of every HIP kernel (the two new ones included); the 27B DFlash2 probe (taps with fusion on
#                    and INTARSIA_NO_FUSION=1, compared; draft / decode / verify8 ms); G0 plan show + 256 ids for the 8B and 35B
#   card 1 (tree B = origin/main, then D): G0 the same on main; GPU admission on main (is any failure pre-existing?);
#                    the 27B at tp=1, 256 greedy ids, --spec off vs --spec dflash --spec-k 7, short and 4k
set -uo pipefail
export O=${1:?out dir}
source "$HOME/dfh/lib.sh"
D=${D:-dev}; B=${B:-base}
log "w4 start, VRAM in use $(( $(vram) / 1048576 )) MiB"
clear "w4" || exit 2
noop() { # noop <tree> <gpu> <tag>
  local t=$1 g=$2 tag=$3 m mm kind
  for m in 35 8; do
    mm=$M35; [[ $m == 8 ]] && mm=$M8
    for kind in short 4k; do gen $t $g "$tag-$m-$kind" $mm $kind; done
    plan $t $g "$tag-$m" $mm
  done
}
(
  htest $D admission-d 0 1200 --test admission -- --nocapture --test-threads=1
  E="INTARSIA_DFLASH2_TARGET=$M27 INTARSIA_DFLASH2_DRAFT=$DRAFT INTARSIA_DFLASH2_LEN=3000 INTARSIA_DFLASH2_TAPS_OUT=/work/dfh-taps-fused.bin" \
    mtest $D probe27-fused 0 900 --test dflash2_real -- --nocapture
  E="INTARSIA_DFLASH2_TARGET=$M27 INTARSIA_DFLASH2_DRAFT=$DRAFT INTARSIA_DFLASH2_LEN=3000 INTARSIA_DFLASH2_TAPS_OUT=/work/dfh-taps-nofusion.bin INTARSIA_NO_FUSION=1" \
    mtest $D probe27-nofusion 0 900 --test dflash2_real -- --nocapture
  if [[ ${DRY:-0} != 1 ]]; then
    python3 "$HOME/dfh/cmp_taps.py" "$(T $D)/dfh-taps-fused.bin" "$(T $D)/dfh-taps-nofusion.bin" > "$O/cmp-taps.txt" 2>&1
    log "taps fusion on vs off exit $?: $(tr '\n' '|' < "$O/cmp-taps.txt")"
  fi
  noop $D 0 d
) &
(
  noop $B 1 b
  for kind in short 4k; do
    gen $D 1 "27b-off-$kind" $M27 $kind
    gen $D 1 "27b-dfl7-$kind" $M27 $kind --spec dflash --spec-draft $DRAFT --spec-k 7
    same "gen-27b-off-$kind.ids" "gen-27b-dfl7-$kind.ids" "27B tp1 $kind ids, dflash2 k7 vs off"
  done
  htest $B admission-b 1 1200 --test admission -- --nocapture --test-threads=1
) &
wait
for m in 35 8; do
  for kind in short 4k; do same "gen-d-$m-$kind.ids" "gen-b-$m-$kind.ids" "G0 tp=1 $m $kind 256 ids, dev vs main"; done
  for f in plan plantext; do
    if diff <(grep -av 'stamp\|head=\|code-tree' "$O/$f-d-$m.out") <(grep -av 'stamp\|head=\|code-tree' "$O/$f-b-$m.out") > "$O/$f-$m.diff"; then
      log "SAME G0 $f show $m"; else log "DIFF G0 $f show $m: $(wc -l < "$O/$f-$m.diff") lines in $f-$m.diff"; fi
  done
done
log "w4 done"

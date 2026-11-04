#!/bin/zsh
# ABAB-style speed chain on endor, ONE binary for every arm: A = default (HSB unset = unsplit, the HS=1 baseline), B = INTARSIA_METAL_ATTN_HSB=3,
# C = INTARSIA_METAL_ATTN_HS=3 (reference, not bitwise). Order A B C B A C, then the same with MTP k2 (the serve default).
export FLEETQ_OWNER=claude-m27-hsb SIZES=4096,16384 GEN=256 REPS=3
W=/opt/code/intarsiaml/.worktrees/m27-hs-bitwise; BIN=$W/.scratch/intarsia.hsb2; OUT=$W/docs/results/2026-11-03-m27-hs-bitwise/raw/speed
for mode in off k2; do
  if [ $mode = off ]; then SA=(--spec off); else SA=(); fi
  for arm in A1:A B1:B C1:C B2:B A2:A C2:C; do
    tag=${arm%%:*}; kind=${arm#*:}
    case $kind in A) EV=(X=1);; B) EV=(INTARSIA_METAL_ATTN_HSB=3);; C) EV=(INTARSIA_METAL_ATTN_HS=3);; esac
    fleetq run --host endor --expect 14 --mem 28 --quiet gpu,cpu --label "m27-hsb: speed $mode $tag ($kind) 4k/16k" -- env $EV zsh $W/docs/results/2026-11-03-m27-hs-bitwise/raw/scripts/speed-arm.sh $mode-$tag $BIN $OUT $SA
  done
done
echo CHAIN-DONE

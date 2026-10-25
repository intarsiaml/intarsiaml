#!/usr/bin/env bash
# Lane sp window 7, ON endor inside a fleetq job (--takes-down radiance): plain (spec off) serving, main vs branch, --max-batch 16:
# speed-plain (filler 128/1k/4k/16k) and conc-plain-1k (1/4/16 streams, filler text). No speculation (MTP is single-sequence).
S=/opt/code/wt-sp/scripts/sp
trap 'ssh tatooine "bash ~/sp-win.sh restore" 2>&1 | tail -2' EXIT
bash $S/arm.sh sp-base main-plain plain "" "" 16
bash $S/arm.sh sp br-plain plain "" "" 16

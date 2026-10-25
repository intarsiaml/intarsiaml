#!/usr/bin/env bash
# Lane sp window 3, ON tatooine inside a fleetq job that takes radiance down (the stop is below, as in baseline.sh).
#   card 0 (27B): eager MTP k=2 (graph off) vs spec off at 4k/16k (the per-row eager verify fix), host-gap trace of a captured k=2
#                 run, per-op table of an eager k=2 run
#   card 1 (35B): eager k=2 ids at all four sizes (md5 vs window 1), host-gap trace, per-op table
set -uo pipefail
trap 'bash $HOME/sp-win.sh restore' EXIT
systemctl --user stop radiance; sleep 3
S=$HOME/intarsiaml-sp/scripts/sp
H=/data/models/radiance/hf-cache/hub
M27=/models/models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/5233554c5fa56afda40150556b95573c2d7d29c0
mkdir -p $HOME/sp-ids
( HIP_VISIBLE_DEVICES=0 MODELS=$H MODEL=$M27 INTARSIA_SPEC_GRAPH=0 bash $S/ids.sh sp w3-t27-mtp2e 128,1024,4096,16384 --spec mtp --spec-k 2
  HIP_VISIBLE_DEVICES=0 MODELS=$H MODEL=$M27 INTARSIA_SPEC_TRACE=1 bash $S/ids.sh sp w3-t27-trace 1024 --spec mtp --spec-k 2
  HIP_VISIBLE_DEVICES=0 MODELS=$H MODEL=$M27 INTARSIA_SPEC_GRAPH=0 PROFILE_EVERY=1 bash $S/ids.sh sp w3-t27-prof 1024 --spec mtp --spec-k 2 --op-table
) > $HOME/sp-ids/w3-27.log 2>&1 &
( HIP_VISIBLE_DEVICES=1 INTARSIA_SPEC_GRAPH=0 bash $S/ids.sh sp w3-35-mtp2e 128,1024,4096,16384 --spec mtp --spec-k 2
  HIP_VISIBLE_DEVICES=1 INTARSIA_SPEC_TRACE=1 bash $S/ids.sh sp w3-35-trace 1024 --spec mtp --spec-k 2
  HIP_VISIBLE_DEVICES=1 INTARSIA_SPEC_GRAPH=0 PROFILE_EVERY=1 bash $S/ids.sh sp w3-35-prof 1024 --spec mtp --spec-k 2 --op-table
) > $HOME/sp-ids/w3-35.log 2>&1 &
wait
cat $HOME/sp-ids/w3-27.log $HOME/sp-ids/w3-35.log

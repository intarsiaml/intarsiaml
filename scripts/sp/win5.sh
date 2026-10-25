#!/usr/bin/env bash
# Lane sp window 5, ON tatooine inside a fleetq job that takes radiance down (the stop is below, as in baseline.sh).
#   --mtp-draft-vocab N sweep, k=2 captured, natural-v1 128/1k/4k/16k: ids md5 vs spec off (equal by construction), acceptance, tok/s
#   card 0: 27B N=16384, 32768, 65536; card 1: 35B N=16384, 32768, 65536
set -uo pipefail
trap 'bash $HOME/sp-win.sh restore' EXIT
systemctl --user stop radiance; sleep 3
S=$HOME/intarsiaml-sp/scripts/sp
H=/data/models/radiance/hf-cache/hub
M27=/models/models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/5233554c5fa56afda40150556b95573c2d7d29c0
mkdir -p $HOME/sp-ids
( for n in 32768 16384 65536; do
    HIP_VISIBLE_DEVICES=0 MODELS=$H MODEL=$M27 bash $S/ids.sh sp w5-t27-v$n 128,1024,4096,16384 --spec mtp --spec-k 2 --mtp-draft-vocab $n
  done ) > $HOME/sp-ids/w5-27.log 2>&1 &
( for n in 32768 16384 65536; do
    HIP_VISIBLE_DEVICES=1 bash $S/ids.sh sp w5-35-v$n 128,1024,4096,16384 --spec mtp --spec-k 2 --mtp-draft-vocab $n
  done ) > $HOME/sp-ids/w5-35.log 2>&1 &
wait
cat $HOME/sp-ids/w5-27.log $HOME/sp-ids/w5-35.log

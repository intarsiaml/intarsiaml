#!/usr/bin/env bash
# Lane sp, ON tatooine inside a window that already has radiance down (no stop here).
#   card 0: 27B Chinese and code prompts at ~1k tokens, full head vs --mtp-draft-vocab 65536 (md5 vs spec off, acceptance, tok/s)
#   card 1: 35B no-op against main at 0b84f499 (sp-base): spec off and MTP k=2, 128 and 1k
S=$HOME/intarsiaml-sp/scripts/sp
H=/data/models/radiance/hf-cache/hub
M27=/models/models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/5233554c5fa56afda40150556b95573c2d7d29c0
mkdir -p $HOME/sp-ids
( for a in "off:--spec off" "full:--spec mtp --spec-k 2" "v65536:--spec mtp --spec-k 2 --mtp-draft-vocab 65536"; do
    tag=${a%%:*}; args=${a#*:}
    HIP_VISIBLE_DEVICES=0 MODELS=$H MODEL=$M27 bash $S/ids.sh sp x-t27-$tag cn1k,code1k $args
  done ) > $HOME/sp-ids/x-27.log 2>&1 &
( for t in sp-base sp; do
    HIP_VISIBLE_DEVICES=1 bash $S/ids.sh $t x-35-$t-off 128,1024 --spec off
    HIP_VISIBLE_DEVICES=1 bash $S/ids.sh $t x-35-$t-mtp2 128,1024 --spec mtp --spec-k 2
  done ) > $HOME/sp-ids/x-35.log 2>&1 &
wait
cat $HOME/sp-ids/x-27.log $HOME/sp-ids/x-35.log

#!/usr/bin/env bash
# Lane sp, ON tatooine inside a fleetq job that takes radiance down (stop below, restore in the trap): the measured-cost MTP gate.
#   card 0 (tree sp): `intarsia serve` (--max-batch 1) with the localeval-shaped client, 35B natural-v1 128/1k/4k/16k, 5 reps:
#       off | mtp k=2 INTARSIA_SPEC_COSTGATE=0 (today's behaviour) | mtp k=2 gate on (default)
#   card 1 (tree sp): `generate`, greedy ids and decode tok/s for spec off | gate off | gate on:
#       35B chat template 128/1k/4k/16k, 35B raw completion 128/1k/4k, 27B chat 128/1k/4k/16k
set -uo pipefail
O=$HOME/sp-logs/win10; mkdir -p "$O" $HOME/sp-ids
S=$HOME/intarsiaml-sp/scripts/sp
H=/data/models/radiance/hf-cache/hub
M27=/models/models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/5233554c5fa56afda40150556b95573c2d7d29c0
trap 'bash $HOME/sp-win.sh restore' EXIT
systemctl --user stop radiance; sleep 3
( export HIP_VISIBLE_DEVICES=1
  CHAT="--chat --no-think --temperature 0"
  bash $S/ids.sh sp c35-off 128,1024,4096,16384 $CHAT --spec off
  INTARSIA_SPEC_COSTGATE=0 bash $S/ids.sh sp c35-cg0 128,1024,4096,16384 $CHAT --spec mtp --spec-k 2
  bash $S/ids.sh sp c35-cg1 128,1024,4096,16384 $CHAT --spec mtp --spec-k 2
  bash $S/ids.sh sp r35-off 128,1024,4096 --spec off
  INTARSIA_SPEC_COSTGATE=0 bash $S/ids.sh sp r35-cg0 128,1024,4096 --spec mtp --spec-k 2
  bash $S/ids.sh sp r35-cg1 128,1024,4096 --spec mtp --spec-k 2
  MODELS=$H MODEL=$M27 bash $S/ids.sh sp c27-off 128,1024,4096,16384 $CHAT --spec off
  MODELS=$H MODEL=$M27 INTARSIA_SPEC_COSTGATE=0 bash $S/ids.sh sp c27-cg0 128,1024,4096,16384 $CHAT --spec mtp --spec-k 2
  MODELS=$H MODEL=$M27 bash $S/ids.sh sp c27-cg1 128,1024,4096,16384 $CHAT --spec mtp --spec-k 2
) > $O/generate.log 2>&1 &
arm() { # arm <label> <serve args> [KEY=VAL ...]
  l=$1; sa=$2; shift 2
  SERVE_ARGS="$sa" MAX_BATCH=1 bash $HOME/sp-win.sh up sp w10-$l "$@" 2>&1 | tail -1
  python3 $S/servedclient.py $l 128,1024,4096,16384 5 > $O/served-$l.log 2>&1
  bash $HOME/sp-win.sh down w10-$l 2>&1 | tail -1
}
arm off ""
arm mtp-cg0 "--spec mtp --spec-k 2" INTARSIA_SPEC_COSTGATE=0
arm mtp-cg1 "--spec mtp --spec-k 2"
wait
for f in $O/served-*.log; do cat $f; done
echo "== generate"; cat $O/generate.log
echo "[win10] done"

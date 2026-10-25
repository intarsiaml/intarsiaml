#!/usr/bin/env bash
# Lane sp, ON tatooine inside a fleetq job that takes radiance down (stop below, restore in the trap): why does served MTP gain
# so little at 1k/4k/16k?  card 0: `intarsia serve` (branch tree) with a localeval-shaped client that reads /metrics around every
# request: spec off / mtp k=2 / mtp k=2 with INTARSIA_SPEC_GATE=0 / mtp k=2 with INTARSIA_SPEC_TRACE=1.
# card 1: `generate --chat --no-think` (the chat template, as served) with spec off and mtp k=2 on the same prompts.
set -uo pipefail
O=$HOME/sp-logs/win9; mkdir -p "$O" $HOME/sp-ids
S=$HOME/intarsiaml-sp/scripts/sp
trap 'bash $HOME/sp-win.sh restore' EXIT
systemctl --user stop radiance; sleep 3
( export HIP_VISIBLE_DEVICES=1
  bash $S/ids.sh sp c-off 128,1024,4096,16384 --chat --no-think --temperature 0 --spec off
  bash $S/ids.sh sp c-mtp2 128,1024,4096,16384 --chat --no-think --temperature 0 --spec mtp --spec-k 2
) > $O/chat-generate.log 2>&1 &
arm() { # arm <label> <serve args> [KEY=VAL ...]
  l=$1; sa=$2; shift 2
  SERVE_ARGS="$sa" MAX_BATCH=1 bash $HOME/sp-win.sh up sp w9-$l "$@" 2>&1 | tail -1
  python3 $S/servedclient.py $l 128,1024,4096,16384 3 > $O/served-$l.log 2>&1
  bash $HOME/sp-win.sh down w9-$l 2>&1 | tail -1
}
arm off ""
arm mtp "--spec mtp --spec-k 2"
arm mtp-gate0 "--spec mtp --spec-k 2" INTARSIA_SPEC_GATE=0
arm mtp-trace "--spec mtp --spec-k 2" INTARSIA_SPEC_TRACE=1
wait
for f in $O/served-*.log; do cat $f; done
echo "== generate --chat"; cat $O/chat-generate.log

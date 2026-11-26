#!/usr/bin/env bash
# Window 6 (lane ink-hip-verify, brief 2 part 1), both cards: the draft-length confirmation on the merged head. Binary intarsia-main2 (main 80719483a),
# no --spec, no TAU / CUT env: the spec default (DSpark, cut 0.5) with the recipe's [tuning.hip] mtp_default_k from --recipe copies
# (k3 = recipes/inkling2.toml verbatim; k4 / k5 = that file with only that line changed). Card 1's k3 arm runs without --recipe
# (the built-in recipe) so its counters show the copy = the default path. speed.json (warm-up + the 12 short golden prompts +
# 1021 / 4108 / 16455 ids), 128 tokens, greedy, EOS honoured. Order ABCCBA per card: card 0 k3 k5 k4 k4 k5 k3, card 1 k4 k3 k5 k5 k3 k4.
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-verify
S=$REPO/.scratch
O=$REPO/logs/w6
rm -rf "$O"; mkdir -p "$O/speed"
M=/models/Inkling-2-Mini
B=/work/.scratch/bin/intarsia-${BR:-main2}
T0=$(date +%s)
LIMIT=${LIMIT:-240}
now() { date '+%T'; }
G="generate -m $M --backend hip --max-seq 20480 --profile-every 0 --show-ids --max-tokens 128 --prompt-ids /work/.scratch/prompts/speed.json"
arm() {  # card k rep [default]
  local f=$O/speed/k$2@$3
  local el=$(( $(date +%s) - T0 ))
  if [ $el -gt $LIMIT ]; then echo "[speed] skip k$2@$3 (t+${el}s)"; return; fi
  local R="--recipe /work/.scratch/recipes/inkling2-k$2.toml"
  [ "${4:-}" = default ] && R=""
  if [ "$1" = 0 ]; then
    HIP_VISIBLE_DEVICES=0 bash "$S/run.sh" $B $G $R > $f.out 2> $f.err
  else
    HIP_VISIBLE_DEVICES=1 INK_CONTAINER=ink-hip-verify-run2 bash "$S/run.sh" $B $G $R > $f.out 2> $f.err
  fi
  echo "[speed] k$2@$3 ${4:-recipe} rc $? t+$(( $(date +%s) - T0 ))s | $(grep -aE '^decode overall' $f.err | head -1) | $(grep -aE '^spec: verify' $f.err | head -1)"
}
card0() { arm 0 3 c0a; arm 0 5 c0a; arm 0 4 c0a; arm 0 4 c0b; arm 0 5 c0b; arm 0 3 c0b; echo "[c0] done $(now)"; }
card1() { arm 1 4 c1a; arm 1 3 c1a default; arm 1 5 c1a; arm 1 5 c1b; arm 1 3 c1b default; arm 1 4 c1b; echo "[c1] done $(now)"; }
echo "[stage] start $(now)"
card0 > "$O/c0.log" 2>&1 &
card1 > "$O/c1.log" 2>&1 &
wait
echo "---- card 0"; cat "$O/c0.log"
echo "---- card 1"; cat "$O/c1.log"
echo "[stage] done $(now)"

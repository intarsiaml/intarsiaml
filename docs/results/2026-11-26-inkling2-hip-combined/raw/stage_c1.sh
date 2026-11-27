#!/usr/bin/env bash
# Window C1 (lane ink-hip-combined), both cards: spec-default decode 4 + 4 ABBA, 256 tokens. Arm "old" = intarsia-main2 (main
# 80719483a, its defaults: DSpark k3, no P12, main's verify rows), arm "new" = intarsia-comb (main 178d87ece, its defaults: k5, P12,
# vrow). No --spec / --recipe / TAU / CUT env. speed.json (warm-up + 12 short golden prompts + 1021 / 4108 / 16455 ids), greedy, EOS
# honoured; the same processes give prefill at 4k (4108) and 16k (16455). Card 0 old new new old, card 1 new old old new. An arm
# starts only before t+LIMIT s.
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-verify
S=$REPO/.scratch
O=$REPO/logs/c1
rm -rf "$O"; mkdir -p "$O/speed"
M=/models/Inkling-2-Mini
T0=$(date +%s)
LIMIT=${LIMIT:-200}
now() { date '+%T'; }
for b in main2 comb; do [ -x "$S/bin/intarsia-$b" ] || { echo "[stage] guard: intarsia-$b missing"; exit 3; }; done
run() {  # card cmd...
  local c=$1; shift
  if [ "$c" = 0 ]; then HIP_VISIBLE_DEVICES=0 bash "$S/run.sh" "$@"
  else HIP_VISIBLE_DEVICES=1 INK_CONTAINER=ink-hip-verify-run2 bash "$S/run.sh" "$@"; fi
}
G="generate -m $M --backend hip --max-seq 20480 --profile-every 0 --show-ids --max-tokens 256 --prompt-ids /work/.scratch/prompts/speed.json"
arm() {  # card arm rep
  local f=$O/speed/$2@$3 el=$(( $(date +%s) - T0 )) B=main2
  if [ $el -gt $LIMIT ]; then echo "[speed] skip $2@$3 (t+${el}s)"; return; fi
  [ "$2" = new ] && B=comb
  run "$1" /work/.scratch/bin/intarsia-$B $G > $f.out 2> $f.err
  echo "[speed] $2@$3 rc $? t+$(( $(date +%s) - T0 ))s | $(grep -aoE 'up to [0-9]+ drafts' $f.err | head -1) | $(grep -aE '^decode overall' $f.err | head -1) | $(grep -aE '^spec: verify' $f.err | head -1)"
}
card0() { arm 0 old c0a; arm 0 new c0a; arm 0 new c0b; arm 0 old c0b; echo "[c0] done $(now)"; }
card1() { arm 1 new c1a; arm 1 old c1a; arm 1 old c1b; arm 1 new c1b; echo "[c1] done $(now)"; }
echo "[stage] start $(now) main2 $(md5sum "$S/bin/intarsia-main2" | cut -c1-32) comb $(md5sum "$S/bin/intarsia-comb" | cut -c1-32)"
card0 > "$O/c0.log" 2>&1 &
card1 > "$O/c1.log" 2>&1 &
wait
echo "---- card 0"; cat "$O/c0.log"
echo "---- card 1"; cat "$O/c1.log"
echo "[stage] done $(now)"

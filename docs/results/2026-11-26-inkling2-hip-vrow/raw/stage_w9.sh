#!/usr/bin/env bash
# Window 9 (lane ink-hip-vrow), both cards: spec-default decode, vrow2's new rules (arm "new": blend fuse 8:2, op 75 fused for 1-row
# calls only) against main's rules in the same binary (arm "main": INTARSIA_HIP_INK_BLEND_FUSE=2 INTARSIA_HIP_INK_DEC_FUSED_ROWS=16,
# which is main 12c6c78ac's launch sequence). No --spec, no TAU / CUT env: the built-in recipe (k3). speed.json (warm-up + 12 short
# golden prompts + 1021 / 4108 / 16455 ids), 128 tokens, greedy, EOS honoured. Order per card: card 0 main new new main, card 1 new
# main main new (4 + 4). Then, while time is left, k5 (--recipe .scratch/recipes/inkling2-k5.toml, as #534 ships): card 0 main5 new5,
# card 1 new5 main5.
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-verify
S=$REPO/.scratch
O=$REPO/logs/w9
rm -rf "$O"; mkdir -p "$O/speed"
M=/models/Inkling-2-Mini
B=/work/.scratch/bin/intarsia-${BR:-vrow2}
T0=$(date +%s)
LIMIT=${LIMIT:-220}
now() { date '+%T'; }
G="generate -m $M --backend hip --max-seq 20480 --profile-every 0 --show-ids --max-tokens 128 --prompt-ids /work/.scratch/prompts/speed.json"
arm() {  # card arm rep
  local f=$O/speed/$2@$3
  local el=$(( $(date +%s) - T0 ))
  if [ $el -gt $LIMIT ]; then echo "[speed] skip $2@$3 (t+${el}s)"; return; fi
  local E="" R=""
  case $2 in main*) E="INTARSIA_HIP_INK_BLEND_FUSE=2 INTARSIA_HIP_INK_DEC_FUSED_ROWS=16" ;; esac
  case $2 in *5) R="--recipe /work/.scratch/recipes/inkling2-k5.toml" ;; esac
  if [ "$1" = 0 ]; then
    HIP_VISIBLE_DEVICES=0 bash "$S/run.sh" env $E $B $G $R > $f.out 2> $f.err
  else
    HIP_VISIBLE_DEVICES=1 INK_CONTAINER=ink-hip-verify-run2 bash "$S/run.sh" env $E $B $G $R > $f.out 2> $f.err
  fi
  echo "[speed] $2@$3 rc $? t+$(( $(date +%s) - T0 ))s | $(grep -aE '^decode overall' $f.err | head -1) | $(grep -aE '^spec: verify' $f.err | head -1)"
}
card0() { arm 0 main c0a; arm 0 new c0a; arm 0 new c0b; arm 0 main c0b; arm 0 main5 c0a; arm 0 new5 c0a; echo "[c0] done $(now)"; }
card1() { arm 1 new c1a; arm 1 main c1a; arm 1 main c1b; arm 1 new c1b; arm 1 new5 c1a; arm 1 main5 c1a; echo "[c1] done $(now)"; }
echo "[stage] start $(now)"
card0 > "$O/c0.log" 2>&1 &
card1 > "$O/c1.log" 2>&1 &
wait
echo "---- card 0"; cat "$O/c0.log"
echo "---- card 1"; cat "$O/c1.log"
echo "[stage] done $(now)"

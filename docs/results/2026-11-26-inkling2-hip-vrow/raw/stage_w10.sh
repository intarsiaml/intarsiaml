#!/usr/bin/env bash
# Window 10 (lane ink-hip-vrow), both cards, binary vrow3 = the branch merged with main b7450223e (#534: built-in HIP k5).
# (1) spec-default decode 4 + 4 ABBA at the shipped k5 (no --spec, no --recipe, no TAU / CUT env), arm "new" = the defaults, arm
#     "main" = INTARSIA_HIP_INK_BLEND_FUSE=2 INTARSIA_HIP_INK_DEC_FUSED_ROWS=16 (main's launch sequence) in the same binary; speed.json
#     (warm-up + 12 short golden prompts + 1021 / 4108 / 16455 ids), 128 tokens, greedy. Card 0 main new new main, card 1 new main main new.
# (2) gate-1 greedy md5 on vrow3, 64 tokens: card 0 short (spec default, --spec off), card 1 4k (same); compare with main's (verify2,
#     window 7) and vrow2's (window 8). Speed arms start only before t+LIMIT s, md5 runs only before t+MLIMIT s.
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-verify
S=$REPO/.scratch
O=$REPO/logs/w10
rm -rf "$O"; mkdir -p "$O/speed" "$O/md5"
M=/models/Inkling-2-Mini
BR=${BR:-vrow3}
B=/work/.scratch/bin/intarsia-$BR
[ -x "$S/bin/intarsia-$BR" ] || { echo "[stage] guard: $BR missing"; exit 3; }
T0=$(date +%s)
LIMIT=${LIMIT:-150}
MLIMIT=${MLIMIT:-205}
now() { date '+%T'; }
run() {  # card cmd...
  local c=$1; shift
  if [ "$c" = 0 ]; then HIP_VISIBLE_DEVICES=0 bash "$S/run.sh" "$@"
  else HIP_VISIBLE_DEVICES=1 INK_CONTAINER=ink-hip-verify-run2 bash "$S/run.sh" "$@"; fi
}
G="generate -m $M --backend hip --max-seq 20480 --profile-every 0 --show-ids --max-tokens 128 --prompt-ids /work/.scratch/prompts/speed.json"
arm() {  # card arm rep
  local f=$O/speed/$2@$3 el=$(( $(date +%s) - T0 )) E=""
  if [ $el -gt $LIMIT ]; then echo "[speed] skip $2@$3 (t+${el}s)"; return; fi
  [ "$2" = main ] && E="INTARSIA_HIP_INK_BLEND_FUSE=2 INTARSIA_HIP_INK_DEC_FUSED_ROWS=16"
  run "$1" env $E $B $G > $f.out 2> $f.err
  echo "[speed] $2@$3 rc $? t+$(( $(date +%s) - T0 ))s | $(grep -aoE 'up to [0-9]+ drafts' $f.err | head -1) | $(grep -aE '^decode overall' $f.err | head -1) | $(grep -aE '^spec: verify' $f.err | head -1)"
}
PS="$(cat "$REPO/scripts/cmx/ids-prompts/short-0.txt")"
P4="$(cat "$REPO/scripts/cmx/ids-prompts/4k-0.txt")"
gen() {  # card mode(off|dflt) prompt-name tokens
  local el=$(( $(date +%s) - T0 ))
  if [ $el -gt $MLIMIT ]; then echo "MD5 skip $2 $3 $4 (t+${el}s)"; return; fi
  local f=$O/md5/$BR-$2-$3-$4 X="" PR="$PS"
  [ "$2" = off ] && X="--spec off"
  [ "$3" = 4k ] && PR="$P4"
  run "$1" $B generate -m $M --backend hip --max-seq 8192 --ignore-eos --profile-every 0 --max-tokens "$4" $X -p "$PR" > "$f.out" 2> "$f.err"
  local rc=$? md5; md5=$(sed -e 's/finish=length.*//' -e '/steps=/d' -e '/tok\/s/d' "$f.out" | md5sum | cut -c1-32)
  echo "MD5 $BR $2 $3 $4 $md5 rc $rc | $(grep -aoE 'up to [0-9]+ drafts' "$f.err" | head -1) | $(cat "$f.out" "$f.err" | grep -aoE 'decode [0-9]+ ms \([0-9.]+ tok/s\)' | head -1) | $(grep -aE '^spec: verify' "$f.err" | head -1 | cut -c1-90)"
}
card0() { arm 0 main c0a; arm 0 new c0a; arm 0 new c0b; arm 0 main c0b; gen 0 dflt short 64; gen 0 off short 64; echo "[c0] done $(now)"; }
card1() { arm 1 new c1a; arm 1 main c1a; arm 1 main c1b; arm 1 new c1b; gen 1 dflt 4k 64; gen 1 off 4k 64; echo "[c1] done $(now)"; }
echo "[stage] start $(now) $BR $(md5sum "$S/bin/intarsia-$BR" | cut -c1-32)"
card0 > "$O/c0.log" 2>&1 &
card1 > "$O/c1.log" 2>&1 &
wait
echo "---- card 0"; cat "$O/c0.log"
echo "---- card 1"; cat "$O/c1.log"
echo "[stage] done $(now)"

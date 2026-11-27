#!/usr/bin/env bash
# Lane ink-hip-specflip: the confirmation ABBA of flip B (DFlash2 k4 vs DSpark k5), a copy of lane ink-dflash2's stage_ab_k.sh with the
# output dir logs/conf-k$K, card $CARD (run.sh, default 1) and binary $BR (default main7 = origin/main after #541). Same prompts / arms / analysis.
# cut 0.5); arm "dfl" = --spec dflash --spec-draft /models/Inkling-2-Mini-DFlash2 --spec-k $K. speed.json (raw/, md5 7763e474...),
# $TOK tokens, greedy, EOS honoured. ORDER: "a" = dspark dfl dfl dspark (reps R1 R2), "b" = dfl dspark dspark dfl (reps R1 R2).
# An arm starts only before t+LIMIT s. Out: logs/abk$K/<arm>@<rep>.err/.out (speed.py reads them).
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-specflip
S=$REPO/.scratch
RAW=$REPO/docs/results/2026-11-26-inkling2-hip-specflip/raw
M=/models/Inkling-2-Mini
D=/models/Inkling-2-Mini-DFlash2
BR=${BR:-main7}; K=${K:-4}; TOK=${TOK:-256}; LIMIT=${LIMIT:-195}; ORDER=${ORDER:-a}; R1=${R1:-r1}; R2=${R2:-r2}
O=$REPO/logs/conf-k$K
mkdir -p "$O" "$S/prompts"; cp "$RAW/speed.json" "$S/prompts/"
B=/work/.scratch/bin/intarsia-$BR
[ -x "$S/bin/intarsia-$BR" ] || { echo "[stage] guard: $BR missing"; exit 3; }
T0=$(date +%s)
now() { date '+%T'; }
el() { echo $(( $(date +%s) - T0 )); }
G="generate -m $M --backend hip --max-seq 20480 --profile-every 0 --show-ids --max-tokens $TOK --prompt-ids /work/.scratch/prompts/speed.json"
arm() {  # arm rep
  local f=$O/$1@$2 X=""
  if [ "$(el)" -gt "$LIMIT" ]; then echo "[speed] skip $1@$2 (t+$(el)s)"; return; fi
  [ "$1" = dfl ] && X="--spec dflash --spec-draft $D --spec-k $K"
  bash "$RAW/run.sh" $B $G $X > "$f.out" 2> "$f.err"
  echo "[speed] $1@$2 rc $? t+$(el)s | $(grep -aoE 'up to [0-9]+ drafts' "$f.err" | head -1) | $(grep -aE '^decode overall' "$f.err" | head -1) | $(grep -aE '^spec: verify' "$f.err" | head -1 | cut -c1-100)"
}
echo "[stage] start $(now) card ${CARD:-1} $(md5sum "$RAW/speed.json" | cut -c1-32) $BR $(md5sum "$S/bin/intarsia-$BR" | cut -c1-32) K $K TOK $TOK ORDER $ORDER $R1 $R2"
if [ "$ORDER" = a ]; then arm dspark $R1; arm dfl $R1; arm dfl $R2; arm dspark $R2
else arm dfl $R1; arm dspark $R1; arm dspark $R2; arm dfl $R2; fi
echo "[stage] done $(now) t+$(el)s"

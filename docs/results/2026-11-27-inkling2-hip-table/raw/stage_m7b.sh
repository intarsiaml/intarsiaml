#!/usr/bin/env bash
# Lane ink-hip-table (from lane ink-hip-specflip stage_m7b.sh): the HIP speed table, main7 (4b629184c) vs now (main 351657cc1), ON tatooine inside window.sh. Both cards at once, each its own
# ABBA of main7 (A) vs specA2 (B), fresh processes, greedy, 256 tokens, EOS honoured (stage_ab_k.sh's protocol):
#   default-spec decode: no --spec flags (DSpark k5), speed-m7d.json (0 warm, 1-3 short, 4 len1k, 5 len4k, 6 len16k) -> logs/m7-def/
#   plain decode: --spec off, speed-m7p.json (0 warm, 1-3 short, 4 len4k, 5 len16k)                               -> logs/m7-off/
# Card 0 reps r1 r2, card 1 reps r3 r4 (pairs within a card; speed.py pairs by rep). Orders: card 0 def A B B A then off B A A B; card 1
# def B A A B then off A B B A. No admission (generate runs none), nothing untimed beside the timed runs.
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-table
S=$REPO/.scratch
RAW=$REPO/docs/results/2026-11-27-inkling2-hip-table/raw
A=${A:-main7}; B=${B:-now}
M=/models/Inkling-2-Mini
mkdir -p "$REPO/logs/m7-def" "$REPO/logs/m7-off" "$S/prompts" "$S/home"
cp "$RAW/speed-m7d.json" "$RAW/speed-m7p.json" "$S/prompts/"
for x in $A $B; do [ -x "$S/bin/intarsia-$x" ] || { echo "[m7b] guard: $x missing"; exit 3; }; done
T0=$(date +%s); el() { echo $(( $(date +%s) - T0 )); }
rc() { local c=$1; shift; env CARD=$c CNAME=ink-hip-table-run$c bash "$RAW/run.sh" env HOME=/work/.scratch/home "$@"; }
echo "[m7b] start $(date +%T) A $A $(md5sum "$S/bin/intarsia-$A" | cut -c1-32) $(cat "$S/bin/intarsia-$A.commit" 2>/dev/null | head -1) B $B $(md5sum "$S/bin/intarsia-$B" | cut -c1-32)"
G="generate -m $M --backend hip --max-seq 20480 --profile-every 0 --show-ids --max-tokens 256"
arm() {  # card mode(def|off) bin rep
  local f=$REPO/logs/m7-$2/$3@$4 x=""
  [ $2 = off ] && x="--spec off --prompt-ids /work/.scratch/prompts/speed-m7p.json" || x="--prompt-ids /work/.scratch/prompts/speed-m7d.json"
  rc $1 /work/.scratch/bin/intarsia-$3 $G $x > "$f.out" 2> "$f.err"
  echo "[m7b c$1] $2 $3@$4 rc $? t+$(el)s | $(grep -aE '^decode overall' "$f.err") | $(grep -aE '^prefill overall' "$f.err") | $(grep -aE '^spec: verify' "$f.err" | cut -c1-90)"
}
card0() { arm 0 def $A r1; arm 0 def $B r1; arm 0 def $B r2; arm 0 def $A r2; arm 0 off $B r1; arm 0 off $A r1; arm 0 off $A r2; arm 0 off $B r2; }
card1() { arm 1 def $B r3; arm 1 def $A r3; arm 1 def $A r4; arm 1 def $B r4; arm 1 off $A r3; arm 1 off $B r3; arm 1 off $B r4; arm 1 off $A r4; }
card0 > "$REPO/logs/m7-c0.log" 2>&1 &
card1 > "$REPO/logs/m7-c1.log" 2>&1 &
wait
cat "$REPO/logs/m7-c0.log" "$REPO/logs/m7-c1.log"
echo "[m7b] done $(date +%T) t+$(el)s"

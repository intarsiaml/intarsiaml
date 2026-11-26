#!/usr/bin/env bash
# Window C2 (lane ink-hip-combined), both cards. (1) greedy md5 of new (intarsia-comb, main 178d87ece), generate --ignore-eos, the
# text before finish= (as lane ink-hip-verify's window 4 / lane ink-hip-vrow's window 8): card 0 short-0, card 1 4k-0, each 64 / 256
# tokens x spec default / --spec off. (2) plain --spec off decode 4 + 4 ABBA, 256 tokens, old (intarsia-main2, main 80719483a) vs new,
# prompt file $PF (a cut of speed.json, see the doc). (3) while time is left (t < OLIMIT): old's md5 again (64 tokens). An arm starts
# only before t+LIMIT s.
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-verify
S=$REPO/.scratch
O=$REPO/logs/c2
rm -rf "$O"; mkdir -p "$O/speed" "$O/md5"
M=/models/Inkling-2-Mini
T0=$(date +%s)
LIMIT=${LIMIT:-120}
OLIMIT=${OLIMIT:-140}
PF=${PF:-plain-s4-4k.json}
now() { date '+%T'; }
for b in main2 comb; do [ -x "$S/bin/intarsia-$b" ] || { echo "[stage] guard: intarsia-$b missing"; exit 3; }; done
[ -f "$S/prompts/$PF" ] || { echo "[stage] guard: $PF missing"; exit 3; }
run() {  # card cmd...
  local c=$1; shift
  if [ "$c" = 0 ]; then HIP_VISIBLE_DEVICES=0 bash "$S/run.sh" "$@"
  else HIP_VISIBLE_DEVICES=1 INK_CONTAINER=ink-hip-verify-run2 bash "$S/run.sh" "$@"; fi
}
PS="$(cat "$REPO/scripts/cmx/ids-prompts/short-0.txt")"
P4="$(cat "$REPO/scripts/cmx/ids-prompts/4k-0.txt")"
gen() {  # card binary mode(off|dflt) prompt-name tokens [limit]
  local el=$(( $(date +%s) - T0 ))
  if [ -n "${6:-}" ] && [ $el -gt "$6" ]; then echo "MD5 skip $2 $3 $4 $5 (t+${el}s)"; return; fi
  local f=$O/md5/$2-$3-$4-$5 X="" PR="$PS"
  [ "$3" = off ] && X="--spec off"
  [ "$4" = 4k ] && PR="$P4"
  run "$1" /work/.scratch/bin/intarsia-$2 generate -m $M --backend hip --max-seq 8192 --ignore-eos --profile-every 0 --max-tokens "$5" $X -p "$PR" > "$f.out" 2> "$f.err"
  local rc=$? md5; md5=$(sed -e 's/finish=length.*//' -e '/steps=/d' -e '/tok\/s/d' "$f.out" | md5sum | cut -c1-32)
  echo "MD5 $2 $3 $4 $5 $md5 rc $rc | $(grep -aoE 'up to [0-9]+ drafts' "$f.err" | head -1) | $(cat "$f.out" "$f.err" | grep -aoE 'decode [0-9]+ ms \([0-9.]+ tok/s\)' | head -1) | t+$(( $(date +%s) - T0 ))s"
}
G="generate -m $M --backend hip --max-seq 20480 --profile-every 0 --show-ids --max-tokens 256 --spec off --prompt-ids /work/.scratch/prompts/$PF"
arm() {  # card arm rep
  local f=$O/speed/$2@$3 el=$(( $(date +%s) - T0 )) B=main2
  if [ $el -gt $LIMIT ]; then echo "[plain] skip $2@$3 (t+${el}s)"; return; fi
  [ "$2" = new ] && B=comb
  run "$1" /work/.scratch/bin/intarsia-$B $G > $f.out 2> $f.err
  echo "[plain] $2@$3 rc $? t+$(( $(date +%s) - T0 ))s | $(grep -aE '^decode overall' $f.err | head -1)"
}
card0() {
  for n in 64 256; do for m in dflt off; do gen 0 comb $m short $n; done; done
  arm 0 old c0a; arm 0 new c0a; arm 0 new c0b; arm 0 old c0b
  gen 0 main2 dflt short 64 $OLIMIT; gen 0 main2 off short 64 $OLIMIT
  echo "[c0] done $(now)"
}
card1() {
  for n in 64 256; do for m in dflt off; do gen 1 comb $m 4k $n; done; done
  arm 1 new c1a; arm 1 old c1a; arm 1 old c1b; arm 1 new c1b
  gen 1 main2 dflt 4k 64 $OLIMIT; gen 1 main2 off 4k 64 $OLIMIT
  echo "[c1] done $(now)"
}
echo "[stage] start $(now) PF=$PF LIMIT=$LIMIT OLIMIT=$OLIMIT"
card0 > "$O/c0.log" 2>&1 &
card1 > "$O/c1.log" 2>&1 &
wait
echo "---- card 0"; cat "$O/c0.log"
echo "---- card 1"; cat "$O/c1.log"
echo "[stage] done $(now)"

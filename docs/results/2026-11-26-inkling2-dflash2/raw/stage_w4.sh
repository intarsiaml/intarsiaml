#!/usr/bin/env bash
# Window 4 (lane ink-dflash2), card 0, binary BR (default dfl6 = the PR head): (1) greedy ids gate at k4 and k5: md5.json (golden code
# 101 ids + the 4108-id prompt), --ignore-eos, 256 then 64 tokens, --spec off vs DFlash2 k4 vs k5; (2) only with PAIR=1 (not used: k5 failed the 4k cell after n=2): one ABBA pair for k5 (rep r3:
# DFlash2 k5 then DSpark k5) on speed.json, started only before t+90 s. Out: logs/w4/md5-*, logs/abk5/<arm>@r3.
set -uo pipefail
REPO=$HOME/intarsiaml-ink-dflash2
S=$REPO/.scratch
RAW=$REPO/docs/results/2026-11-26-inkling2-dflash2/raw
O=$REPO/logs/w4
rm -rf "$O"; mkdir -p "$O" "$S/prompts" "$REPO/logs/abk5"
cp "$RAW/speed.json" "$RAW/md5.json" "$S/prompts/"
M=/models/Inkling-2-Mini
D=/models/Inkling-2-Mini-DFlash2
BR=${BR:-dfl6}
B=/work/.scratch/bin/intarsia-$BR
[ -x "$S/bin/intarsia-$BR" ] || { echo "[stage] guard: $BR missing"; exit 3; }
T0=$(date +%s)
now() { date '+%T'; }
el() { echo $(( $(date +%s) - T0 )); }
run() { bash "$RAW/run.sh" "$@"; }
echo "[stage] start $(now) $BR $(md5sum "$S/bin/intarsia-$BR" | cut -c1-32)"
gen() {  # arm(off|k4|k5) tokens
  local f=$O/md5-$1-$2 X="--spec off"
  case "$1" in k4) X="--spec dflash --spec-draft $D --spec-k 4";; k5) X="--spec dflash --spec-draft $D --spec-k 5";; esac
  run $B generate -m $M --backend hip --max-seq 8192 --ignore-eos --profile-every 0 --show-ids --max-tokens "$2" $X \
    --prompt-ids /work/.scratch/prompts/md5.json > "$f.out" 2> "$f.err"
  local rc=$?
  echo "[md5] $1 $2 rc $rc t+$(el)s ids-md5 $(cat "$f.out" "$f.err" | grep -aE '^ids\[' | md5sum | cut -c1-32) | $(grep -aE '^spec: verify' "$f.err" | head -1 | cut -c1-100)"
}
for n in 256 64; do for a in off k4 k5; do [ "$(el)" -lt 100 ] && gen $a $n; done; done
G="generate -m $M --backend hip --max-seq 20480 --profile-every 0 --show-ids --max-tokens 256 --prompt-ids /work/.scratch/prompts/speed.json"
arm() {  # arm rep
  local f=$REPO/logs/abk5/$1@$2 X=""
  [ "$1" = dfl ] && X="--spec dflash --spec-draft $D --spec-k 5"
  run $B $G $X > "$f.out" 2> "$f.err"
  echo "[speed] $1@$2 rc $? t+$(el)s | $(grep -aE '^decode overall' "$f.err" | head -1) | $(grep -aE '^spec: verify' "$f.err" | head -1 | cut -c1-100)"
}
if [ "${PAIR:-0}" = 1 ] && [ "$(el)" -le 90 ]; then arm dfl r3; arm dspark r3; else echo "[speed] pair skipped (t+$(el)s)"; fi
echo "[stage] done $(now) t+$(el)s"

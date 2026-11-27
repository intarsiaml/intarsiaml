#!/usr/bin/env bash
# k sweep window (lane ink-dflash2), card 0, binary BR (dfl5): one process per arm over ksweep.json (warm + 12 golden + 4108 ids),
# 256 tokens, greedy, EOS honoured. Arms in order: dspark (no --spec flags), DFlash2 k7, k6, k5, k4, dspark again. Informational
# (the ABBA used k7, chosen before it ran). An arm starts only before t+LIMIT s. Out: logs/k/<arm>@k.err/.out.
set -uo pipefail
REPO=$HOME/intarsiaml-ink-dflash2
S=$REPO/.scratch
RAW=$REPO/docs/results/2026-11-26-inkling2-dflash2/raw
O=$REPO/logs/k
rm -rf "$O"; mkdir -p "$O" "$S/prompts"; cp "$RAW/ksweep.json" "$S/prompts/"
M=/models/Inkling-2-Mini
D=/models/Inkling-2-Mini-DFlash2
BR=${BR:-dfl5}; LIMIT=${LIMIT:-170}
B=/work/.scratch/bin/intarsia-$BR
[ -x "$S/bin/intarsia-$BR" ] || { echo "[stage] guard: $BR missing"; exit 3; }
T0=$(date +%s)
now() { date '+%T'; }
el() { echo $(( $(date +%s) - T0 )); }
G="generate -m $M --backend hip --max-seq 8192 --profile-every 0 --show-ids --max-tokens 256 --prompt-ids /work/.scratch/prompts/ksweep.json"
arm() {  # name [k]
  local f=$O/$1@k X=""
  if [ "$(el)" -gt "$LIMIT" ]; then echo "[k] skip $1 (t+$(el)s)"; return; fi
  [ -n "${2:-}" ] && X="--spec dflash --spec-draft $D --spec-k $2"
  bash "$RAW/run.sh" $B $G $X > "$f.out" 2> "$f.err"
  echo "[k] $1 rc $? t+$(el)s | $(grep -aE '^decode overall' "$f.err" | head -1) | $(grep -aE '^spec: verify' "$f.err" | head -1 | cut -c1-100)"
}
echo "[stage] start $(now) $BR $(md5sum "$S/bin/intarsia-$BR" | cut -c1-32)"
arm dspark1; arm k7 7; arm k6 6; arm k5 5; arm k4 4; arm dspark2
echo "[stage] done $(now) t+$(el)s"

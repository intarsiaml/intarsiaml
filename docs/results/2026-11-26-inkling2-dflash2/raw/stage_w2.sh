#!/usr/bin/env bash
# Window 2 (lane ink-dflash2), card 0, binary BR (default dfl4 = branch head with embed_norm on by default). Each step only before
# its time limit:
# (1) greedy ids gate: md5.json (golden code 101 ids, 4108 ids), --ignore-eos, 64 and 256 tokens: --spec off vs DFlash2 k7
# (2) trace (INTARSIA_SPEC_TRACE=1, 256 tokens, trace.json): DFlash2 k7 (per-pass accepted, draft ms, verify ms -> k choice)
# (3) INTARSIA_DFLASH2_HEAD=mxfp4 on var.json (warm, code, prose 300, code_tests), DFlash2 k7, no trace; and the bf16 head again
# (4) tests/inkling2_dflash2.rs on HIP, the built test executable (INTARSIA_INKLING2_BACKEND=hip): greedy_ids, admitted
set -uo pipefail
REPO=$HOME/intarsiaml-ink-dflash2
S=$REPO/.scratch
RAW=$REPO/docs/results/2026-11-26-inkling2-dflash2/raw
O=$REPO/logs/w2
rm -rf "$O"; mkdir -p "$O" "$S/prompts"
cp "$RAW/speed.json" "$RAW/md5.json" "$RAW/trace.json" "$RAW/var.json" "$S/prompts/"
M=/models/Inkling-2-Mini
D=/models/Inkling-2-Mini-DFlash2
BR=${BR:-dfl4}
B=/work/.scratch/bin/intarsia-$BR
[ -x "$S/bin/intarsia-$BR" ] || { echo "[stage] guard: $BR missing"; exit 3; }
TB=$(grep -aoE "Executable tests/inkling2_dflash2.rs \(/target/[^)]*\)" "$REPO/logs/build-mtest-inkling2_dflash2.log" | tail -1 | sed -e "s/.*(//" -e "s/)//")
T0=$(date +%s)
now() { date '+%T'; }
el() { echo $(( $(date +%s) - T0 )); }
run() { bash "$RAW/run.sh" "$@"; }
DFL="--spec dflash --spec-draft $D --spec-k 7"
echo "[stage] start $(now) $BR $(md5sum "$S/bin/intarsia-$BR" | cut -c1-32) test bin ${TB:-none}"
gen() {  # arm(off|dfl) tokens
  local f=$O/md5-$1-$2 X="--spec off"
  [ "$1" = dfl ] && X="$DFL"
  run $B generate -m $M --backend hip --max-seq 8192 --ignore-eos --profile-every 0 --show-ids --max-tokens "$2" $X \
    --prompt-ids /work/.scratch/prompts/md5.json > "$f.out" 2> "$f.err"
  local rc=$?
  echo "[md5] $1 $2 rc $rc t+$(el)s ids-md5 $(cat "$f.out" "$f.err" | grep -aE '^ids\[' | md5sum | cut -c1-32) | $(grep -aE '^spec: verify' "$f.err" | head -1 | cut -c1-100)"
}
for n in 64 256; do
  for a in off dfl; do [ "$(el)" -lt 70 ] && gen $a $n; done
done
trace_run() {
  local f=$O/trace-dfl
  run env INTARSIA_SPEC_TRACE=1 $B generate -m $M --backend hip --max-seq 8192 --profile-every 0 --show-ids --max-tokens 256 $DFL \
    --prompt-ids /work/.scratch/prompts/trace.json > "$f.out" 2> "$f.err"
  echo "[trace] dfl rc $? t+$(el)s | $(grep -aE '^spec: verify' "$f.err" | head -1 | cut -c1-100)"
}
[ "$(el)" -lt 95 ] && trace_run
var() {  # name env...
  local f=$O/var-$1; shift
  run env "$@" $B generate -m $M --backend hip --max-seq 8192 --profile-every 0 --show-ids --max-tokens 256 $DFL \
    --prompt-ids /work/.scratch/prompts/var.json > "$f.out" 2> "$f.err"
  echo "[var] $(basename "$f") rc $? t+$(el)s | $(grep -aE '^decode overall' "$f.err" | head -1) | $(grep -aE '^spec: verify' "$f.err" | head -1 | cut -c1-100)"
}
[ "$(el)" -lt 130 ] && var mxfp4 INTARSIA_DFLASH2_HEAD=mxfp4
[ "$(el)" -lt 150 ] && var bf16 INTARSIA_DFLASH2_HEAD=bf16
if [ -n "$TB" ] && [ "$(el)" -lt 165 ]; then
  run env INTARSIA_INKLING2_BACKEND=hip timeout 85 "$TB" --test-threads 1 greedy_ids admitted > "$O/tiny-hip.log" 2>&1
  echo "[tiny] rc $? t+$(el)s | $(grep -aE '^test |^test result|panicked' "$O/tiny-hip.log" | head -5 | paste -sd ' ' -)"
fi
echo "[stage] done $(now) t+$(el)s"

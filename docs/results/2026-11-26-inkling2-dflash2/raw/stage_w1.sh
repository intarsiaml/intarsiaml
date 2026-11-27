#!/usr/bin/env bash
# Window 1 (lane ink-dflash2), card 0, binary dfl2 (branch head). Each step only before its time limit, in the order (3) (4) (5) (2) (1):
# (1) tests/inkling2_dflash2.rs on HIP (INTARSIA_INKLING2_BACKEND=hip): taps = DSpark's, greedy ids = spec off, admitted kernels only
# (2) admission: kernels test --backend hip --op dyn_conv --model /models/Inkling-2-Mini --quick (H 2304, K 2, G 144)
# (3) greedy ids gate: md5.json (golden code 101 ids, 4108 ids), --ignore-eos, 64 and 256 tokens: --spec off vs DFlash2 k7
# (4) trace (INTARSIA_SPEC_TRACE=1, 256 tokens, trace.json): DFlash2 k7 and the default (DSpark k5): per-pass accepted, draft ms, verify ms
# (5) variants on var.json (warm, code, prose 300, code_tests) without trace: INTARSIA_DFLASH2_EMBED_NORM=1, INTARSIA_DFLASH2_HEAD=mxfp4 (tokens per pass, tok/s)
set -uo pipefail
REPO=$HOME/intarsiaml-ink-dflash2
S=$REPO/.scratch
RAW=$REPO/docs/results/2026-11-26-inkling2-dflash2/raw
O=$REPO/logs/w1
rm -rf "$O"; mkdir -p "$O"
M=/models/Inkling-2-Mini
D=/models/Inkling-2-Mini-DFlash2
BR=${BR:-dfl2}
B=/work/.scratch/bin/intarsia-$BR
[ -x "$S/bin/intarsia-$BR" ] || { echo "[stage] guard: $BR missing"; exit 3; }
mkdir -p "$S/prompts"; cp "$RAW/speed.json" "$RAW/md5.json" "$RAW/trace.json" "$RAW/var.json" "$S/prompts/"
T0=$(date +%s)
now() { date '+%T'; }
el() { echo $(( $(date +%s) - T0 )); }
run() { bash "$RAW/run.sh" "$@"; }
DFL="--spec dflash --spec-draft $D --spec-k 7"
echo "[stage] start $(now) $BR $(md5sum "$S/bin/intarsia-$BR" | cut -c1-32)"
# (3)
gen() {  # arm(off|dfl) tokens
  local f=$O/md5-$1-$2 X="--spec off"
  [ "$1" = dfl ] && X="$DFL"
  run $B generate -m $M --backend hip --max-seq 8192 --ignore-eos --profile-every 0 --show-ids --max-tokens "$2" $X \
    --prompt-ids /work/.scratch/prompts/md5.json > "$f.out" 2> "$f.err"
  local rc=$?
  echo "[md5] $1 $2 rc $rc t+$(el)s ids-md5 $(cat "$f.out" "$f.err" | grep -aE '^ids\[' | md5sum | cut -c1-32) | $(grep -aE '^spec: verify' "$f.err" | head -1 | cut -c1-100)"
}
for n in 64 256; do
  for a in off dfl; do [ "$(el)" -lt 100 ] && gen $a $n; done
done
# (4)
tr() {  # arm(dfl|dflt)
  local f=$O/trace-$1 X=""
  [ "$1" = dfl ] && X="$DFL"
  run env INTARSIA_SPEC_TRACE=1 $B generate -m $M --backend hip --max-seq 8192 --profile-every 0 --show-ids --max-tokens 256 $X \
    --prompt-ids /work/.scratch/prompts/trace.json > "$f.out" 2> "$f.err"
  echo "[trace] $1 rc $? t+$(el)s | $(grep -aoE 'up to [0-9]+ drafts' "$f.err" | head -1) | $(grep -aE '^spec: verify' "$f.err" | head -1 | cut -c1-100)"
}
[ "$(el)" -lt 130 ] && tr dfl
[ "$(el)" -lt 165 ] && tr dflt
# (5)
var() {  # name env...
  local f=$O/var-$1; shift
  run env "$@" $B generate -m $M --backend hip --max-seq 8192 --profile-every 0 --show-ids --max-tokens 256 $DFL \
    --prompt-ids /work/.scratch/prompts/var.json > "$f.out" 2> "$f.err"
  echo "[var] $(basename $f) rc $? t+$(el)s | $(grep -aE '^decode overall' "$f.err" | head -1) | $(grep -aE '^spec: verify' "$f.err" | head -1 | cut -c1-100)"
}
[ "$(el)" -lt 195 ] && var embnorm INTARSIA_DFLASH2_EMBED_NORM=1
[ "$(el)" -lt 215 ] && var mxfp4 INTARSIA_DFLASH2_HEAD=mxfp4
# (2)
if [ "$(el)" -lt 225 ]; then
  run timeout 20 $B kernels test --backend hip --op dyn_conv --model $M --quick > "$O/kt-dynconv.log" 2>&1
  echo "[admission] dyn_conv rc $? t+$(el)s | $(grep -aiE 'fail|pass|admit' "$O/kt-dynconv.log" | tail -2 | tr '\n' ' ')"
fi
# (1)
if [ "$(el)" -lt 230 ]; then
  run env INTARSIA_INKLING2_BACKEND=hip timeout 30 cargo test --release -q -p intarsia-models --test inkling2_dflash2 -- --test-threads 1 \
    taps_equal greedy_ids admitted > "$O/tiny-hip.log" 2>&1
  echo "[tiny] rc $? t+$(el)s | $(grep -aE '^test result|FAILED|panicked' "$O/tiny-hip.log" | head -4 | tr '\n' ' ')"
fi
echo "[stage] done $(now) t+$(el)s"

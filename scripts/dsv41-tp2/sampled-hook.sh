# Lane dsv41/sampled-spec: sourced by run-full.sh (HOOK=this file; CAP default, CHUNK=engine). One window, one binary
# ($SPARK_DIR/bin/intarsia), started fresh per arm, with INTARSIA_DSV41_SPEC_SAMPLED unset ("off", main's behaviour) or 1 ("on"):
#   gate-off, gate-on   golden + golden_long + decode prompts (greedy, 13 prompts, 1120 tokens) against the stored ids of main
#                       (prefill3's gate-tok-main-c2048.st): identical with the flag off and with it on
#   tr                  on, INTARSIA_DSV41_SPEC_TRACE=1: catspeed set at temperature 0.7, then at 0 (the greedy acceptance of the same
#                       requests): retention = sampled over greedy acceptance per position. Below 0.5 the lane stops here (kill test)
#   off1 on1 off2 on2   ABAB: catspeed set at 0.7 and at 1.0 (same fixed seeds per request in both arms), 8 constrained json_schema
#                       requests at 0.7; the on arms run with INTARSIA_DSV41_SPEC_RETENTION = the retention measured in tr
#   goff gon            G-test arms: 1500 sequential short sampled requests each (gtest.py), disjoint seeds, SPEC_CHOOSE=0 SPEC_K=4 in gon
# Results in $OUT; PASS/FAIL lines in $OUT/steps.log ("RESULT ...").
TOK=${TOKENIZER:-/opt/code/intarsiaml/.worktrees/dsv41-official/mia/tokenizer.json}
RD=$REPO/docs/results
REF=$RD/2026-11-05-dsv41-prefill3/raw/ab/gate-tok-main-c2048.st
PROMPTS="$REPO/tools/prompts/golden.jsonl $REPO/tools/prompts/golden_long.jsonl $S/replay-prompts-decode.jsonl"
GN=${GN:-1500}
lib() { uv run --no-project --with numpy --with safetensors python "$S/replay-test-lib.py" "$@"; }
result() { log "RESULT $*"; }
arm() { # NAME ENV cmd...: fresh ranks with extra ENV, the command, the logs kept
  local name=$1 env=$2; shift 2
  start_ranks "$env" || return 1
  "$@"; local rc=$?
  scp -q "spark1:$SPARK_REL/logs/serve-$TAG.log" "$OUT/serve-$name.log" 2>/dev/null; scp -q "spark2:$SPARK_REL/logs/worker-$TAG.log" "$OUT/worker-$name.log" 2>/dev/null
  kill_ours
  return $rc
}
gold() { # OUT NOTE
  ( cd "$REPO" && perl -e 'alarm shift; exec @ARGV' 1500 uv run --with numpy --with safetensors --with tokenizers python tools/dsv41/golden_tf.py tokens \
      --base-url "$SURL/v1" --model dsv41-intarsia --no-draft-field --tokenizer "$TOK" --out "$1" --note "$2" --prompts $PROMPTS ) 2>&1 | tail -4 | tee -a "$OUT/steps.log"
}
specn() { local n; n=$(r spark1 "grep -ac 'dsv41: spec [0-9]' $LOGD/serve-$TAG.log; true" 2>/dev/null | tail -1); echo "${n:-0}"; }
catseg() { # LABEL TEMP: the 32-request set at TEMP; its speculation lines, tokens a round and acceptance by position, speed by kind
  local n0; n0=$(specn)
  perl -e 'alarm shift; exec @ARGV' 1500 python3 "$S/sampcat.py" "$SURL/v1" "$OUT/cat-$1" "$2" --tokens 256 > "$OUT/cat-$1.txt" 2>&1
  r spark1 "grep -a 'dsv41: spec [0-9]' $LOGD/serve-$TAG.log | tail -n +$(( n0 + 1 ))" > "$OUT/spec-$1.txt" 2>/dev/null
  tail -1 "$OUT/cat-$1.txt" | tee -a "$OUT/steps.log"
  python3 "$S/catspeed.py" "$1=$OUT/cat-$1" 2>&1 | tee "$OUT/catspeed-$1.txt" | tee -a "$OUT/steps.log"
  python3 "$S/specpos.py" "$OUT/spec-$1.txt" "$1" | tee -a "$OUT/steps.log"
  python3 "$S/specsum.py" "$OUT/spec-$1.txt" "$1" "$(python3 -c "
import json,sys
for f in ['/opt/code/deepeval/datasets/categories-rcp.jsonl','$S/chat-rcp.jsonl']:
    for l in open(f):
        if l.startswith('{'): print(json.loads(l)['name'])" | paste -sd, -)" 2>&1 | grep -E "^spec $1 (code|prose|chat|other|all):" | tee -a "$OUT/steps.log"
}
trace_run() {
  catseg tr-t07 0.7
  catseg tr-t0 0
  RET=$(python3 "$S/specpos.py" ratio "$OUT/spec-tr-t07.txt" "$OUT/spec-tr-t0.txt" | tee -a "$OUT/steps.log" | sed -n 's/.*weighted \([0-9.]*\);.*/\1/p')
  log "measured retention $RET"
}
abab_run() { catseg "$1-t07" 0.7; catseg "$1-t10" 1.0
  python3 "$S/conscheck.py" "$SURL/v1" "$OUT/cons-$1.json" 0.7 2>&1 | tee "$OUT/cons-$1.txt" | tail -3 | tee -a "$OUT/steps.log"; }
gt_run() { python3 "$S/gtest.py" run "$SURL/v1" "$OUT/g-$1.jsonl" "$GN" "$2" 2>&1 | tail -2 | tee -a "$OUT/steps.log"; }

# ---- gate: greedy ids against main's, flag off and on
for who in off on; do
  e=""; [[ $who == on ]] && e="INTARSIA_DSV41_SPEC_SAMPLED=1"
  arm "gate-$who" "$e" gold "$OUT/gate-tok-$who.st" "sampled-spec gate flag $who"
  lib tokcmp "$REF" "$OUT/gate-tok-$who.st" --name-a main --name-b "flag-$who" --require-identical > "$OUT/gate-compare-$who.txt" 2>&1; rc=$?
  tail -n 4 "$OUT/gate-compare-$who.txt" | tee -a "$OUT/steps.log"
  result "gate greedy ids vs main, flag $who: $([[ $rc == 0 ]] && echo PASS || echo FAIL)"
done
# ---- kill test: retention
RET=""
arm tr "INTARSIA_DSV41_SPEC_SAMPLED=1 INTARSIA_DSV41_SPEC_TRACE=1" trace_run
if [[ -z $RET ]] || python3 -c "import sys; sys.exit(0 if float('${RET:-0}') >= 0.5 else 1)"; then
  result "kill test: retention $RET (>= 0.5): continue"
  RETC=$(python3 -c "print(round(min(1.0, max(0.3, float('$RET'))), 2))")
  # ---- speed ABAB (+ constrained validity)
  for pass in 1 2; do
    arm "off$pass" "" abab_run "off$pass"
    arm "on$pass" "INTARSIA_DSV41_SPEC_SAMPLED=1 INTARSIA_DSV41_SPEC_RETENTION=$RETC" abab_run "on$pass"
  done
  # ---- G-test arms
  arm goff "" gt_run goff 1
  arm gon "INTARSIA_DSV41_SPEC_SAMPLED=1 INTARSIA_DSV41_SPEC_CHOOSE=0 INTARSIA_DSV41_SPEC_K=4" gt_run gon 1000001
  uv run --no-project --with numpy --with scipy python "$S/gtest.py" test "$OUT/g-goff.jsonl" "$OUT/g-gon.jsonl" 2>&1 | tee "$OUT/gtest.txt" | tee -a "$OUT/steps.log"
  result "G-test spec-sampled vs off: $([[ ${PIPESTATUS[0]} == 0 ]] && echo PASS || echo FAIL)"
else
  result "kill test: retention $RET < 0.5: STOP, no speed window"
fi

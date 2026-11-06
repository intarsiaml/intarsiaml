# Lane dsv41/sampled-spec, window 3: sourced by run-full.sh (HOOK=this file). Uses the helpers of sampled-hook.sh.
#   coff, con   constrained json_schema requests (conscheck.py, 16 requests, every reply must finish and match the schema) at temperature
#               0.7 and at 0, flag off and on (retention 1.0)
#   r08         flag on with retention 0.8 (the compiled default): the catspeed set at 0.7, against window 2's retention 1.0
HELPERS_ONLY=1 source "$S/sampled-hook.sh"
cons_run() { local t
  for t in 0.7 0; do
    python3 "$S/conscheck.py" "$SURL/v1" "$OUT/cons-$1-t$t.json" $t --n 16 2>&1 | tee "$OUT/cons-$1-t$t.txt" | tail -4 | tee -a "$OUT/steps.log"
  done
  r spark1 "grep -a 'dsv41: spec [0-9]' $LOGD/serve-$TAG.log" > "$OUT/spec-cons-$1.txt" 2>/dev/null
  python3 "$S/specpos.py" "$OUT/spec-cons-$1.txt" "cons-$1" | tee -a "$OUT/steps.log"; }
arm coff "" cons_run coff
arm con "INTARSIA_DSV41_SPEC_SAMPLED=1 INTARSIA_DSV41_SPEC_RETENTION=1.0" cons_run con
arm r08 "INTARSIA_DSV41_SPEC_SAMPLED=1 INTARSIA_DSV41_SPEC_RETENTION=0.8" catseg r08-t07 0.7
# replication of window 2's G-test with fresh seeds (window 2 position 4 had p 0.0043): same arms, seeds 5000001.. (off) and 6000001.. (on)
arm goff2 "" gt_run goff2 5000001
arm gon2 "INTARSIA_DSV41_SPEC_SAMPLED=1 INTARSIA_DSV41_SPEC_CHOOSE=0 INTARSIA_DSV41_SPEC_K=4" gt_run gon2 6000001
uv run --no-project --with numpy --with scipy python "$S/gtest.py" test "$OUT/g-goff2.jsonl" "$OUT/g-gon2.jsonl" 2>&1 | tee "$OUT/gtest.txt" | tee -a "$OUT/steps.log"
result "G-test replication: $([[ ${PIPESTATUS[0]} == 0 ]] && echo PASS || echo FAIL)"

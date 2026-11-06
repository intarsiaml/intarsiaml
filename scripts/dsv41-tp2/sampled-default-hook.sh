# Lane dsv41/sampled-spec, default-on check: sourced by run-full.sh (HOOK=this file). One arm, no INTARSIA_DSV41_SPEC_SAMPLED set:
# the startup log says sampled and constrained speculation is on, and the greedy golden + golden_long + decode prompts (1120 tokens,
# 640 of them the golden set) equal main's stored ids. Then one sampled request (T 0.7, seed) must show speculation rounds in the log.
HELPERS_ONLY=1 source "$S/sampled-hook.sh"
dflt_run() {
  r spark1 "grep -a 'also for sampled and constrained' $LOGD/serve-$TAG.log" | tee "$OUT/startup-line.txt" | tee -a "$OUT/steps.log"
  gold "$OUT/gate-tok-default.st" "sampled-spec default-on gate"
  python3 "$S/sampcat.py" "$SURL/v1" "$OUT/cat-default" 0.7 --dataset "$S/chat-rcp.jsonl" > "$OUT/cat-default.txt" 2>&1
  r spark1 "grep -a 'dsv41: spec [0-9]' $LOGD/serve-$TAG.log | tail -8" | tee -a "$OUT/steps.log"
}
arm default "" dflt_run
lib tokcmp "$REF" "$OUT/gate-tok-default.st" --name-a main --name-b default-on --require-identical > "$OUT/gate-compare-default.txt" 2>&1; rc=$?
tail -n 3 "$OUT/gate-compare-default.txt" | tee -a "$OUT/steps.log"
result "default-on: greedy ids vs main $([[ $rc == 0 ]] && echo PASS || echo FAIL); startup line $([[ -s $OUT/startup-line.txt ]] && echo present || echo MISSING)"

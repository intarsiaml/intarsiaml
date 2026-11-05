# Lane dsv41/prefill3: sourced by run-full.sh (HOOK=this file; CAP=66000 CHUNK=engine). One window, bin/intarsia.main (origin/main's
# binary, copied before the candidate was built) against bin/intarsia (the candidate), same ranks, same prompts:
#   gate  golden ids + golden_long + the decode prompts (replay-prompts-decode.jsonl: d_1200, d_4106 = a resumed prefill, d_612) at
#         chunk 2048 and 512, main vs candidate: every continuation identical
#   longp 32K, 65K and the 4K prompt, 16 greedy tokens with the top 5 logprobs, at chunk 2048: bit-identical (cmp_longp.py)
#   speed localeval speed (1K 4K 16K 32K, --gen 8 as step 4 of the replay doc), main, cand, main, cand
#   diag  candidate: TTFT of streamed requests by ignore_eos / min_tokens (ttft_probe.py), localeval speed --gen 8 against --gen 128
#   nsys  rank 0 and rank 1, the 4K prompt three times, candidate (class table against main's and TensorFold's)
# Results in $OUT/{gate,longp,speed,diag,nsys}-*; PASS/FAIL lines in $OUT/steps.log ("RESULT ...").
BIN_MAIN="$SPARK_DIR/bin/intarsia.main"; BIN_CAND="$SPARK_DIR/bin/intarsia"
TOK=${TOKENIZER:-/opt/code/intarsiaml/.worktrees/dsv41-official/mia/tokenizer.json}
RD=$REPO/docs/results
PROMPTS="$REPO/tools/prompts/golden.jsonl $REPO/tools/prompts/golden_long.jsonl $S/replay-prompts-decode.jsonl"
LEH="hook scripts/dsv41-tp2/prefill3-ab-hook.sh: rank 0 on spark1 (serve, tp-remote), rank 1 on spark2 (tp-worker), cap ${CAP:-32768}, chunk engine default"
lehook() { ( cd /opt/code/deepeval && perl -e 'alarm shift; exec @ARGV' 1500 uv run localeval "$@" --url "$SURL/v1" --model dsv41-intarsia --thinking off --recipe "$LEH" ); }
lib() { uv run --no-project --with numpy --with safetensors python "$S/replay-test-lib.py" "$@"; }
result() { log "RESULT $*"; }
# pfreq is defined inside run_steps in run-full.sh: define it here (one max_tokens 1 request with a unique first line)
pfreq() { local t0; t0=$(date +%s.%N)
  curl -s -m 300 "$SURL/v1/completions" -H 'content-type: application/json' -d "$(python3 -c 'import json,sys; print(json.dumps({"model":"dsv41-intarsia","prompt":"Run id %s.\n" % sys.argv[2] + open(sys.argv[1]).read(),"max_tokens":1,"temperature":0}))' "$1" "$2-$RANDOM")" > "$OUT/pf-$2.json"
  echo "pf $2 wall $(python3 -c 'import sys,time; print("%.3f" % (time.time() - float(sys.argv[1])))' "$t0") s $(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("usage"))' "$OUT/pf-$2.json")" | tee -a "$OUT/steps.log"; }
# arm NAME BIN CHUNK cmd...: fresh ranks (CHUNK empty = the engine default), the command, the logs kept
arm() {
  local name=$1 bin=$2 chunk=$3; shift 3
  BIN=$bin; NSYS=${ARM_NSYS:-0}
  start_ranks "${chunk:+INTARSIA_DSV41_CHUNK=$chunk}" || return 1
  "$@"; local rc=$?
  scp -q "spark1:$SPARK_REL/logs/serve-$TAG.log" "$OUT/serve-$name.log" 2>/dev/null; scp -q "spark2:$SPARK_REL/logs/worker-$TAG.log" "$OUT/worker-$name.log" 2>/dev/null
  kill_ours
  return $rc
}
gold() { # OUT NOTE
  ( cd "$REPO" && perl -e 'alarm shift; exec @ARGV' 1500 uv run --with numpy --with safetensors --with tokenizers python tools/dsv41/golden_tf.py tokens \
      --base-url "$SURL/v1" --model dsv41-intarsia --no-draft-field --tokenizer "$TOK" --out "$1" --note "$2" --prompts $PROMPTS ) 2>&1 | tail -4 | tee -a "$OUT/steps.log"
}
longp() { # DIR
  mkdir -p "$1"
  local f
  for f in $RD/2026-11-02-dsv41-prefill/prompts/long32k.txt $RD/2026-11-03-dsv41-attn-fusion/prompts/long64k.txt $RD/2026-11-02-dsv41-prefill/prompt4k.txt; do
    curl -s -m 900 "$SURL/v1/completions" -H 'content-type: application/json' -d "$(python3 -c 'import json,sys; print(json.dumps({"model":"dsv41-intarsia","prompt":open(sys.argv[1]).read(),"max_tokens":16,"temperature":0,"logprobs":5}))' "$f")" > "$1/longp-$(basename "$f" .txt).json"
    log "longp $(basename "$f") $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("usage"), repr(d["choices"][0]["text"][:50]))' "$1/longp-$(basename "$f" .txt).json" 2>&1 | tail -1)"
  done
}
speedrun() { # LABEL
  lehook speed --sizes 1024,4096,16384,32768 --gen 8 --reps 3 --label "$1" 2>&1 | tail -22 | tee "$OUT/$1.txt" | tee -a "$OUT/steps.log"
}
diagrun() {
  python3 "$RD/2026-11-05-dsv41-prefill3/ttft_probe.py" "$SURL/v1" "$RD/2026-11-05-dsv41-prefill3/filler4k.txt" 3 2>&1 | tee "$OUT/diag-ttft.txt" | tee -a "$OUT/steps.log"
  lehook speed --sizes 4096 --gen 8 --reps 3 --label dsv41-prefill3-diag-gen8 2>&1 | tail -12 | tee "$OUT/diag-gen8.txt" | tee -a "$OUT/steps.log"
  lehook speed --sizes 4096 --gen 128 --reps 3 --label dsv41-prefill3-diag-gen128 2>&1 | tail -12 | tee "$OUT/diag-gen128.txt" | tee -a "$OUT/steps.log"
}
nsysrun() {
  local f=$RD/2026-11-02-dsv41-prefill/prompt4k.txt rep rk h
  pfreq "$f" warm; pfreq "$f" warm
  r spark1 "nsys start --session=dsv41r0x$TAG -o $LOGD/nsys-r0-$TAG --force-overwrite=true --sample=none --cpuctxsw=none" 2>&1 | tail -2 | tee -a "$OUT/steps.log"
  r spark2 "nsys start --session=dsv41r1x$TAG -o $LOGD/nsys-r1-$TAG --force-overwrite=true --sample=none --cpuctxsw=none" 2>&1 | tail -2 | tee -a "$OUT/steps.log"
  for rep in 1 2 3; do sleep 3; pfreq "$f" traced-$rep; done; sleep 3
  r spark1 "nsys stop --session=dsv41r0x$TAG" 2>&1 | tail -2 | tee -a "$OUT/steps.log"
  r spark2 "nsys stop --session=dsv41r1x$TAG" 2>&1 | tail -2 | tee -a "$OUT/steps.log"
  for rk in 0 1; do
    h=spark$((rk + 1))
    r $h "nsys stats --report cuda_gpu_trace --format csv --output $LOGD/nsys-r$rk-$TAG $LOGD/nsys-r$rk-$TAG.nsys-rep > /dev/null 2>&1" | tee -a "$OUT/steps.log"
    scp -q "$h:$SPARK_REL/logs/nsys-r$rk-$TAG*gpu_trace*.csv" "$OUT/" 2>/dev/null
  done
}

# ---- gates: golden / decode ids, main vs candidate
for chunk in 2048 512; do
  for who in main cand; do
    bin=$BIN_MAIN; [[ $who == cand ]] && bin=$BIN_CAND
    arm "gate-$who-c$chunk" "$bin" "$chunk" gold "$OUT/gate-tok-$who-c$chunk.st" "prefill3 gate $who chunk $chunk"
  done
  lib tokcmp "$OUT/gate-tok-main-c$chunk.st" "$OUT/gate-tok-cand-c$chunk.st" --name-a main --name-b cand --require-identical > "$OUT/gate-compare-c$chunk.txt" 2>&1; rc=$?
  tail -n 4 "$OUT/gate-compare-c$chunk.txt" | tee -a "$OUT/steps.log"
  result "gate ids chunk $chunk main vs cand: $([[ $rc == 0 ]] && echo PASS || echo FAIL)"
done
# ---- longp: top-5 logprobs bit for bit
for who in main cand; do
  bin=$BIN_MAIN; [[ $who == cand ]] && bin=$BIN_CAND
  arm "longp-$who" "$bin" "" longp "$OUT/longp-$who"
done
python3 "$RD/2026-11-02-dsv41-prefill/cmp_longp.py" "$OUT/longp-main" "$OUT/longp-cand" 2>&1 | tee "$OUT/longp-compare.txt" | tee -a "$OUT/steps.log"
result "longp main vs cand: $([[ ${PIPESTATUS[0]} == 0 ]] && echo PASS || echo FAIL)"
# ---- speed ABAB
for pass in 1 2; do
  arm "speed-main$pass" "$BIN_MAIN" "" speedrun "dsv41-prefill3-speed-main$pass"
  arm "speed-cand$pass" "$BIN_CAND" "" speedrun "dsv41-prefill3-speed-cand$pass"
done
# ---- diag on the candidate
arm diag "$BIN_CAND" "" diagrun
# ---- nsys on the candidate (the 4K prompt, three requests)
ARM_NSYS=1; arm nsys "$BIN_CAND" "" nsysrun; ARM_NSYS=0; NSYS=0

# Lane dsv41/prefill3: sourced by run-full.sh (HOOK=this file). Why does localeval's 4K prefill (chat API, 128 forced tokens, filler
# prompt) take 3.3 s where a max_tokens-1 completion of the natural prompt takes 2.6 s, and what does INTARSIA_DSV41_REPLAY=0 cost?
# Per config (replay default, replay off): the same requests through completions / chat with several max_tokens, then localeval speed
# exactly as step 4 of docs/results/2026-11-03-dsv41-replay.md ran it (--gen 8). Last config: nsys (rank 0 and rank 1) of three requests.
# Needs DIAG_FILLER (a prompt file, filler(201, 4096) of localeval + its instruction) and DIAG_NATURAL (prompt4k.txt).
# Output: $OUT/diag-*.txt, $OUT/nsys-*.
: "${DIAG_FILLER:?}" "${DIAG_NATURAL:?}"
LEHOOK="hook scripts/dsv41-tp2/prefill3-diag-hook.sh: rank 0 spark1 serve --tp 2 --tp-remote, rank 1 spark2 tp-worker, cap ${CAP:-32768}, chunk engine default"
lehook() { ( cd /opt/code/deepeval && uv run localeval "$@" --url "$SURL/v1" --model dsv41-intarsia --thinking off --recipe "$LEHOOK" ); }
# dreq LABEL FILE MAX_TOKENS API(comp|chat): one non-streamed request with a unique first line; wall time and usage to $OUT/diag.txt
dreq() {
  local label=$1 f=$2 mt=$3 api=$4 ep=completions t0
  [[ $api == chat ]] && ep=chat/completions
  python3 - "$api" "$f" "$mt" "$label-$RANDOM" > "$OUT/payload.json" <<'PY'
import json, sys
api, f, mt, nonce = sys.argv[1:5]
text = "Run id %s.\n" % nonce + open(f).read()
d = {"model": "dsv41-intarsia", "max_tokens": int(mt), "temperature": 0}
if api == "chat": d["messages"] = [{"role": "user", "content": text}]
else: d["prompt"] = text
print(json.dumps(d))
PY
  t0=$(date +%s.%N)
  curl -s -m 300 "$SURL/v1/$ep" -H 'content-type: application/json' -d @"$OUT/payload.json" > "$OUT/resp.json"
  echo "diag $label $api mt=$mt wall $(python3 -c 'import sys,time; print("%.3f" % (time.time() - float(sys.argv[1])))' "$t0") s $(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("usage"))' "$OUT/resp.json" 2>&1 | tail -1)" | tee -a "$OUT/diag.txt"
}
diag_round() { # CONFIG-NAME
  local n=$1 rep
  dreq warm "$S/../../docs/results/2026-11-02-dsv41-prefill/prompt1k.txt" 1 comp
  for rep in 1 2 3; do
    dreq "$n-natural-comp1" "$DIAG_NATURAL" 1 comp
    dreq "$n-filler-comp1" "$DIAG_FILLER" 1 comp
    dreq "$n-filler-chat1" "$DIAG_FILLER" 1 chat
    dreq "$n-filler-chat8" "$DIAG_FILLER" 8 chat
    dreq "$n-filler-chat128" "$DIAG_FILLER" 128 chat
  done
  log "$n: localeval speed, step 4 of the replay doc (4096 and 32768, --gen 8, --reps 2)"
  lehook speed --sizes 4096,32768 --gen 8 --reps 2 --label "dsv41-prefill3-diag-$n" 2>&1 | tail -25 | tee -a "$OUT/diag.txt"
}
start_ranks ""; diag_round replay; kill_ours
start_ranks "INTARSIA_DSV41_REPLAY=0"; diag_round off; kill_ours
# nsys: three requests (filler completions, filler chat with 16 tokens, natural completions)
NSYS=1; start_ranks ""
dreq warm "$DIAG_NATURAL" 1 comp; dreq warm "$DIAG_FILLER" 1 chat
r spark1 "nsys start --session=dsv41r0x$TAG -o $LOGD/nsys-r0-$TAG --force-overwrite=true --sample=none --cpuctxsw=none" 2>&1 | tail -2 | tee -a "$OUT/steps.log"
r spark2 "nsys start --session=dsv41r1x$TAG -o $LOGD/nsys-r1-$TAG --force-overwrite=true --sample=none --cpuctxsw=none" 2>&1 | tail -2 | tee -a "$OUT/steps.log"
sleep 3; dreq traced-filler-comp1 "$DIAG_FILLER" 1 comp
sleep 3; dreq traced-filler-chat16 "$DIAG_FILLER" 16 chat
sleep 3; dreq traced-natural-comp1 "$DIAG_NATURAL" 1 comp
sleep 3
r spark1 "nsys stop --session=dsv41r0x$TAG" 2>&1 | tail -2 | tee -a "$OUT/steps.log"
r spark2 "nsys stop --session=dsv41r1x$TAG" 2>&1 | tail -2 | tee -a "$OUT/steps.log"
for rk in 0 1; do
  h=spark$((rk + 1))
  r $h "nsys stats --report cuda_gpu_trace --format csv --output $LOGD/nsys-r$rk-$TAG $LOGD/nsys-r$rk-$TAG.nsys-rep > /dev/null 2>&1; ls -la $LOGD/nsys-r$rk-$TAG*" | tee -a "$OUT/steps.log"
  scp -q "$h:$SPARK_REL/logs/nsys-r$rk-$TAG*gpu_trace*.csv" "$OUT/" 2>/dev/null
done
kill_ours

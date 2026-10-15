#!/bin/zsh
# perf.sh OUTDIR "label:bin ..." SIZES : localeval speed passes, one fresh `intarsia serve` per entry (chunk 2048 from the
# recipe, not overridden), macmon 100 ms. Run inside an exclusive fleetq slot on endor (FLEETQ_JOB set by the caller).
OUT=$1; ORDER=(${=2}); SIZES=$3
SC=/private/tmp/claude-501/-opt-code-intarsiaml/0e8026c5-6dd0-4b83-b375-5ff0a66894fc/scratchpad
MODEL=/Users/josef/models/Qwen3-8B
export INTARSIA_CPU_FREE_MEM_GB=60 PYTHONUNBUFFERED=1
mkdir -p $OUT
SPID=""; MPID=""
cleanup() { [[ -n "$SPID" ]] && kill $SPID 2>/dev/null; [[ -n "$MPID" ]] && kill $MPID 2>/dev/null; }
trap cleanup EXIT
/Users/josef/bin/macmon pipe -i 100 > $OUT/macmon.jsonl 2> $OUT/macmon.err &
MPID=$!
i=0
for ent in $ORDER; do
  i=$((i+1)); lab=${ent%%:*}; bin=${ent#*:}; label=p$i-$lab
  echo "##### $label $(date -Iseconds)"
  $SC/bin/$bin serve --model $MODEL --backend metal --device 0 --host 127.0.0.1 --port 8093 --served-model-name intarsia-metal --max-seq 20480 --profile-every 0 --no-supervisor --no-ui 2>&1 >$OUT/$label-server.out | $SC/ts.py > $OUT/$label-server.err &
  for k in {1..120}; do curl -sf http://127.0.0.1:8093/health >/dev/null 2>&1 && break; sleep 1; done
  SPID=$(pgrep -n -f "$bin serve.*--port 8093")
  echo "$label start $(date +%s.%N)" >> $OUT/events.log
  (cd /opt/code/deepeval && uv run localeval speed --url http://127.0.0.1:8093/v1 --model intarsia-metal --sizes $SIZES --gen ${GEN:-32} --reps ${REPS:-3} --thinking off --label prefillbal-$label 2>&1 | $SC/ts.py) > $OUT/$label-localeval.txt
  echo "$label end $(date +%s.%N)" >> $OUT/events.log
  grep -E "tok" $OUT/$label-localeval.txt | cut -c1-200 | tail -12
  kill $SPID; sleep 4; kill -9 $SPID 2>/dev/null; SPID=""
  for k in {1..30}; do lsof -iTCP:8093 -sTCP:LISTEN -nP >/dev/null 2>&1 || break; sleep 1; done
  sleep 5
done
echo PERF_DONE

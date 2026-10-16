#!/bin/zsh
# passes4.sh OUTDIR "label:bin:chunk ..." SIZES : localeval speed passes, one fresh server per entry, macmon 100 ms, engine trace on stderr
OUT=$1; ORDER=(${=2}); SIZES=$3
SC=/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad
MODEL=/Users/josef/models/Qwen3-8B
export INTARSIA_CPU_FREE_MEM_GB=60 PYTHONUNBUFFERED=1 INTARSIA_PREFILL_TRACE=1
mkdir -p $OUT
SPID=""; MPID=""
cleanup() { [[ -n "$SPID" ]] && kill $SPID 2>/dev/null; [[ -n "$MPID" ]] && kill $MPID 2>/dev/null; }
trap cleanup EXIT
/Users/josef/bin/macmon pipe -i 100 > $OUT/macmon.jsonl 2> $OUT/macmon.err &
MPID=$!
i=0
for ent in $ORDER; do
  i=$((i+1)); lab=${ent%%:*}; rest=${ent#*:}; bin=${rest%%:*}; c=${rest##*:}; label=p$i-$lab-c$c
  echo "##### $label $(date -Iseconds)"
  $SC/$bin serve --model $MODEL --backend metal --device 0 --host 127.0.0.1 --port 8093 --served-model-name intarsia-metal --max-seq 8192 --profile-every 0 --no-supervisor --no-ui --prefill-chunk $c 2>&1 >$OUT/$label-server.out | $SC/ts.py > $OUT/$label-server.err &
  for k in {1..120}; do curl -sf http://127.0.0.1:8093/health >/dev/null 2>&1 && break; sleep 1; done
  SPID=$(pgrep -n -f "$bin serve.*--port 8093")
  echo "$label start $(date +%s.%N)" >> $OUT/events.log
  (cd /opt/code/deepeval && uv run localeval speed --url http://127.0.0.1:8093/v1 --model intarsia-metal --sizes $SIZES --gen 256 --reps 3 --thinking off --label p3noise-$label 2>&1 | $SC/ts.py) > $OUT/$label-localeval.txt
  echo "$label end $(date +%s.%N)" >> $OUT/events.log
  grep -E "tok rep" $OUT/$label-localeval.txt | grep -E " 4[01][0-9][0-9] tok" | cut -c1-150
  kill $SPID; sleep 4; kill -9 $SPID 2>/dev/null; SPID=""
  for k in {1..30}; do lsof -iTCP:8093 -sTCP:LISTEN -nP >/dev/null 2>&1 || break; sleep 1; done
  sleep 5
done

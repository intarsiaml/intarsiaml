#!/usr/bin/env bash
# mcdiag on spark1: per server one event in the middle of a run of requests: 2 warm-ups, PRE requests of 300 tokens (code-gen), one request of 800 tokens (code-gen:
# the probe, INTARSIA_SPEC_COPY_PROBE=450, fires in it), POST requests of 300 tokens; arms differ by env only (same binary), interleaved
# rounds with the arm order reversed on even rounds (ABBA). One server per arm and round; INTARSIA_SPEC_TRACE=1 always. Analysis: scripts/mcdiag/serve2_an.py.
#   fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 40 --label "mcdiag: serve event" -- ssh spark1 'W=s2 ROUNDS=3 ARMS="base:X=1 chain:INTARSIA_SPEC_COPY=256+INTARSIA_SPEC_COPY_PROBE=450" bash -s' < scripts/mcdiag/serve2.sh
# ARMS "name:ENV=val[+ENV2=val]" (X=1: none; BIN=<path> picks another binary); ROUNDS (2); PRE / POST (6 requests before / after the event).
set -uo pipefail
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
R=$HOME/intarsiaml-mcdiag; W=${W:?W}; O=$R/out/$W; mkdir -p $O
H=/home/nvidia/.cache/huggingface/hub
M35=$(ls -d $H/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/* | head -1)
BIN=${BIN:-$R/bin/intarsia}; PORT=${PORT:-8197}; ROUNDS=${ROUNDS:-2}; REPS=${REPS:-3}
PRE=${PRE:-6}; POST=${POST:-6}; ARMS=${ARMS:?ARMS}; C=$R/src/scripts/mcopy/cell.py
python3 $R/src/scripts/mcopy/prompts.py $R/cells > /dev/null
stop() { pkill -f "intarsia[a-z0-9-]* serve.*--port $PORT" 2>/dev/null; for i in $(seq 1 30); do pgrep -f "intarsia[a-z0-9-]* serve.*--port $PORT" >/dev/null || break; sleep 1; done; }
trap stop EXIT
python3 - $R/cells <<'EOP'
import json, sys
d = json.load(open(sys.argv[1] + "/code-gen.json")); d["max_tokens"] = 300
json.dump(d, open(sys.argv[1] + "/code-gen300.json", "w"))
EOP
for r in $(seq 1 $ROUNDS); do
  order="$ARMS"; [ $((r % 2)) = 0 ] && order=$(echo $ARMS | awk '{for (i=NF;i>=1;i--) printf "%s ", $i}')
  for spec in $order; do
    a=${spec%%:*}; e=${spec#*:}
    stop
    for i in $(seq 1 40); do curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1 || break; sleep 1; done
    curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { echo "port $PORT already answers (a server of another run?): stopping"; exit 1; }
    env ${e//+/ } INTARSIA_SPEC_TRACE=1 nohup $BIN serve --backend cuda --model $M35 --model-id m --max-batch 1 --max-seq 8192 --spec mtp --spec-k 2 --profile-every 0 --mem-budget 40 \
      --no-supervisor --host 127.0.0.1 --port $PORT > $O/serve-$a-r$r.log 2>&1 < /dev/null &
    for i in $(seq 1 120); do pgrep -f "intarsia[a-z0-9-]* serve.*--port $PORT" >/dev/null || { echo "serve died ($a)"; tail -5 $O/serve-$a-r$r.log; exit 1; }; curl -sf -m 3 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; sleep 3; done
    echo "$a r$r up $(date +%T) free: $(free -g | awk 'NR==2{print $7}') GB"
    python3 $C $PORT $R/cells/chat-short.json 2 warm$r > /dev/null 2>&1
    python3 $C $PORT $R/cells/code-gen300.json $PRE pre$r --out $O/txt-$a-r$r >> $O/$a-r$r.jsonl 2>&1
    python3 $C $PORT $R/cells/code-gen.json 1 mid$r --out $O/txt-$a-r$r >> $O/$a-r$r.jsonl 2>&1
    python3 $C $PORT $R/cells/code-gen300.json $POST post$r --out $O/txt-$a-r$r >> $O/$a-r$r.jsonl 2>&1
    curl -s http://127.0.0.1:$PORT/metrics | grep -E '^intarsia_(spec_(steps|proposed|accepted|copy)|graph_|decode_eager)' | sed 's/{[^}]*}//' | tr '\n' ' ' > $O/metrics-$a-r$r.txt
    echo "$a r$r: $(grep -c '"toks"' $O/$a-r$r.jsonl) requests, shas $(python3 -c "import json,sys; print(' '.join(json.loads(l)['sha'][:4] for l in open('$O/$a-r$r.jsonl') if l.startswith('{')))"), chains $(grep -ac '^spec: copy token' $O/serve-$a-r$r.log)"
  done
done
echo "serve2 done"

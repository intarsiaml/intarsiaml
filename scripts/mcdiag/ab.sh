#!/usr/bin/env bash
# mcdiag on spark1 (the mcopy A/B harness): served A/B, `serve --max-batch 1 --spec mtp --spec-k 2`, arms differ by env only (same binary), interleaved
# rounds with the arm order reversed on even rounds (ABBA). One server per arm and round, cells in a fixed order, REPS reps each.
#   fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 90 --label "mcdiag: A/B" -- ssh spark1 'W=ab1 ROUNDS=2 ARMS="base:X=1 n12:INTARSIA_SPEC_COPY=12" bash -s' < scripts/mcopy/ab.sh
# ARMS "name:ENV=val[+ENV2=val]" (X=1: none; BIN=<path> picks another binary; an arm named `off` serves --spec off; values may hold commas, INTARSIA_SPEC_COPY=20,15); CELLS (default: all five of prompts.py); ROUNDS (2); REPS (3).
set -uo pipefail
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
R=$HOME/intarsiaml-mcdiag; W=${W:?W}; O=$R/out/$W; mkdir -p $O
H=/home/nvidia/.cache/huggingface/hub
M35=$(ls -d $H/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/* | head -1)
BIN=${BIN:-$R/bin/intarsia}; PORT=${PORT:-8097}; ROUNDS=${ROUNDS:-2}; REPS=${REPS:-3}
CELLS=${CELLS:-"copy3k agent-edit json-xform chat-short reason"}; ARMS=${ARMS:?ARMS}; C=$R/src/scripts/mcopy/cell.py
python3 $R/src/scripts/mcopy/prompts.py $R/cells > /dev/null
stop() { pkill -f "intarsia[a-z-]* serve.*--port $PORT" 2>/dev/null; for i in $(seq 1 30); do pgrep -f "intarsia[a-z-]* serve.*--port $PORT" >/dev/null || break; sleep 1; done; }
trap stop EXIT
for r in $(seq 1 $ROUNDS); do
  order="$ARMS"; [ $((r % 2)) = 0 ] && order=$(echo $ARMS | awk '{for (i=NF;i>=1;i--) printf "%s ", $i}')
  for spec in $order; do
    a=${spec%%:*}; e=${spec#*:}
    B=$BIN; case $e in *BIN=*) B=$(echo "$e" | tr "+" "\n" | sed -n "s/^BIN=//p"); e=$(echo "$e" | tr "+" "\n" | grep -v "^BIN=" | tr "\n" "+"); [ -n "$e" ] || e=X=1 ;; esac
    stop
    SA="--spec mtp --spec-k 2"; [ "$a" = off ] && SA="--spec off"
    env ${e//+/ } nohup $B serve --backend cuda --model $M35 --model-id m --max-batch 1 --max-seq 8192 $SA --profile-every 0 --mem-budget 40 \
      --no-supervisor --host 127.0.0.1 --port $PORT > $O/serve-$a-r$r.log 2>&1 < /dev/null &
    for i in $(seq 1 120); do pgrep -f "intarsia[a-z-]* serve.*--port $PORT" >/dev/null || { echo "serve died ($a)"; tail -5 $O/serve-$a-r$r.log; exit 1; }; curl -sf -m 3 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; sleep 3; done
    echo "$a r$r up $(date +%T) free: $(free -g | awk 'NR==2{print $7}') GB"
    python3 $C $PORT $R/cells/chat-short.json 2 warm$r > /dev/null 2>&1
    for c in $CELLS; do python3 $C $PORT $R/cells/$c.json $REPS c$r --out $O/txt-$a-r$r >> $O/$a-r$r.jsonl 2>&1; echo "$a r$r $c: $(tail -1 $O/$a-r$r.jsonl | cut -c1-160)"; done
    curl -s http://127.0.0.1:$PORT/metrics | grep -E '^intarsia_(spec_(steps|proposed|accepted|copy)|graph_|decode_eager)' | sed 's/{[^}]*}//' | tr '\n' ' ' > $O/metrics-$a-r$r.txt
    echo "$a r$r metrics: $(cat $O/metrics-$a-r$r.txt)"
  done
done
echo "ab done"

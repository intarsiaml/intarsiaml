#!/usr/bin/env bash
# copyflipc (from sideeager ab.sh) on spark1: served A/B (`serve --max-batch 1 --spec mtp --spec-k 2`), one server per arm and round, arm order reversed on even rounds (ABBA), cells in a fixed
# order, REPS reps each; arms differ by env only (BIN=<path> in an arm picks another binary); INTARSIA_SPEC_TRACE=1 only with TRACE=1 (timing runs: off, main defaults); SAMPLE=1 adds the per-thread cpu sampler (scripts/cpupin/tsample.py).
#   fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 90 --label "copyflipc: A/B" -- ssh spark1 'W=ab1 ROUNDS=4 REPS=3 CELLS="chat-short code-gen reason agent-edit agent-batch copy3k json-xform" ARMS="base:X=1 n20:INTARSIA_SPEC_COPY=20" bash ~/intarsiaml-copyflipc/src/scripts/copyflipc/ab.sh'
# ARMS "name:ENV=val[+ENV2=val]" (X=1: none); CELLS (default: five of prompts.py); ROUNDS (2); REPS (3).
# ARMS "name:ENV=val[+ENV2=val]" (X=1: none; BIN=<path> picks another binary); CELLS (default: all five of prompts.py); ROUNDS (2); REPS (3); NOSAMPLE=1 skips the sampler.
set -uo pipefail
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
R=$HOME/intarsiaml-copyflipc; W=${W:?W}; O=$R/out/$W; mkdir -p $O
H=/home/nvidia/.cache/huggingface/hub
M35=$(ls -d $H/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/* | head -1)
BIN=${BIN:-$R/bin/intarsia}; PORT=${PORT:-8397}; ROUNDS=${ROUNDS:-2}; REPS=${REPS:-3}
CELLS=${CELLS:-"copy3k agent-edit json-xform chat-short reason"}; ARMS=${ARMS:?ARMS}; C=$R/src/scripts/mcopy/cell.py
python3 $R/src/scripts/mcopy/prompts.py $R/cells > /dev/null
pat() { echo "$R/bin/[a-z0-9-]+ serve.*--port $PORT"; }
stop() { pkill -f "$(pat)" 2>/dev/null; for i in $(seq 1 30); do pgrep -f "$(pat)" >/dev/null || break; sleep 1; done; }
SPID=""
# free-memory watchdog (kills this lane's processes when available memory < 15 GB), always on while the script runs
nohup bash $R/src/scripts/copyflipc/wd.sh > $O/wd.log 2>&1 < /dev/null &
WD=$!
# copyflipc: log the co-resident vLLM's request counters every second (unix time, running, waiting, prompt / generation token totals)
( while :; do echo "$(date +%s.%N) $(curl -s -m 1 http://100.120.88.104:8000/metrics | grep -E '^vllm:(num_requests_running|num_requests_waiting|prompt_tokens_total|generation_tokens_total)\{' | awk '{printf "%s ", $2}')"; sleep 1; done ) > $O/vllm-${W}.log 2>&1 < /dev/null &
VP=$!
trap 'stop; kill $SPID $WD $VP 2>/dev/null' EXIT
for r in $(seq 1 $ROUNDS); do
  order="$ARMS"; [ $((r % 2)) = 0 ] && order=$(echo $ARMS | awk '{for (i=NF;i>=1;i--) printf "%s ", $i}')
  for spec in $order; do
    a=${spec%%:*}; e=${spec#*:}
    B=$BIN; case $e in *BIN=*) B=$(echo "$e" | tr "+" "\n" | sed -n "s/^BIN=//p"); e=$(echo "$e" | tr "+" "\n" | grep -v "^BIN=" | tr "\n" "+"); [ -n "$e" ] || e=X=1 ;; esac
    stop
    for i in $(seq 1 40); do curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1 || break; sleep 1; done
    curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { echo "port $PORT already answers: stopping"; exit 1; }
    ts=$(echo " ${e//+/ } " | sed -n 's/.* TASKSET=\([^ ]*\).*/\1/p'); pre=""; [ -n "$ts" ] && pre="taskset -c $ts"
    env ${e//+/ } ${TRACE:+INTARSIA_SPEC_TRACE=1} $pre nohup $B serve --backend cuda --model $M35 --model-id m --max-batch 1 --max-seq 8192 --spec mtp --spec-k 2 --profile-every 0 --mem-budget 40 \
      --no-supervisor --host 127.0.0.1 --port $PORT > $O/serve-$a-r$r.log 2>&1 < /dev/null &
    for i in $(seq 1 120); do pgrep -f "$(pat)" >/dev/null || { echo "serve died ($a)"; tail -5 $O/serve-$a-r$r.log; exit 1; }; curl -sf -m 3 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; sleep 3; done
    echo "$a r$r up $(date +%T) free: $(free -g | awk 'NR==2{print $7}') GB"
    SPID=""
    if [ -n "${SAMPLE:-}" ]; then
      SP=$(pgrep -f "$(pat)" | head -1)
      taskset -c 11 python3 $R/src/scripts/cpupin/tsample.py $SP 0.1 > $O/ts-$a-r$r.tsv 2>/dev/null < /dev/null &
      SPID=$!
    fi
    nohup bash $R/src/scripts/samp64/clk.sh $O/clk-$a-r$r.log > /dev/null 2>&1 < /dev/null &
    CK=$!
    python3 $C $PORT $R/cells/chat-short.json 2 warm$r > /dev/null 2>&1
    for c in $CELLS; do python3 $C $PORT $R/cells/$c.json $REPS c$r --out $O/txt-$a-r$r >> $O/$a-r$r.jsonl 2>&1; echo "$a r$r $c: $(tail -1 $O/$a-r$r.jsonl | cut -c1-160)"; done
    [ -n "$SPID" ] && kill $SPID 2>/dev/null
    kill $CK 2>/dev/null
    curl -s http://127.0.0.1:$PORT/metrics | grep -E '^intarsia_(spec_(steps|proposed|accepted|copy)|graph_|decode_eager)' | sed 's/{[^}]*}//' | tr '\n' ' ' > $O/metrics-$a-r$r.txt
    echo "$a r$r metrics: $(cat $O/metrics-$a-r$r.txt)"
  done
done
echo "ab done"

#!/usr/bin/env bash
# cpupin on spark1: served A/B (`serve --max-batch 1 --spec mtp --spec-k 2`), one server per arm and round, arm order reversed on even rounds (ABBA), cells in a fixed
# order, REPS reps each; arms differ by env only, optionally pinned with TASKSET=<cpu list> in the arm's env (X925 = 5-9,15-19; A725 = 0-4,10-14, checked with
# cpu_capacity / MIDR on spark1). Every server is sampled by scripts/cpupin/tsample.py (per thread cpu, 100 ms) on A725 cpu 11. INTARSIA_SPEC_TRACE=1 always.
#   (scripts/cpupin/build.sh syncs this directory to ~/intarsiaml-cpupin/src) fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 60 --label "cpupin: A/B" -- ssh spark1 'W=pin1 ROUNDS=4 REPS=8 CELLS=code-gen ARMS="base:X=1 baseP:TASKSET=5-9,15-19 n20:INTARSIA_SPEC_COPY=20 n20P:INTARSIA_SPEC_COPY=20+TASKSET=5-9,15-19" bash ~/intarsiaml-cpupin/src/scripts/cpupin/ab.sh'
# ARMS "name:ENV=val[+ENV2=val]" (X=1: none; BIN=<path> picks another binary); CELLS (default: all five of prompts.py); ROUNDS (2); REPS (3); NOSAMPLE=1 skips the sampler.
set -uo pipefail
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
R=$HOME/intarsiaml-cpupin; W=${W:?W}; O=$R/out/$W; mkdir -p $O
H=/home/nvidia/.cache/huggingface/hub
M35=$(ls -d $H/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/* | head -1)
BIN=${BIN:-$R/bin/intarsia}; PORT=${PORT:-8297}; ROUNDS=${ROUNDS:-2}; REPS=${REPS:-3}
CELLS=${CELLS:-"copy3k agent-edit json-xform chat-short reason"}; ARMS=${ARMS:?ARMS}; C=$R/src/scripts/mcopy/cell.py
python3 $R/src/scripts/mcopy/prompts.py $R/cells > /dev/null
pat() { echo "$R/bin/[a-z0-9-]+ serve.*--port $PORT"; }
stop() { pkill -f "$(pat)" 2>/dev/null; for i in $(seq 1 30); do pgrep -f "$(pat)" >/dev/null || break; sleep 1; done; }
SPID=""
# free-memory watchdog (kills this lane's processes when available memory < 15 GB), always on while the script runs
nohup bash $R/src/scripts/cpupin/wd.sh > $O/wd.log 2>&1 < /dev/null &
WD=$!
trap 'stop; kill $SPID $WD 2>/dev/null' EXIT
for r in $(seq 1 $ROUNDS); do
  order="$ARMS"; [ $((r % 2)) = 0 ] && order=$(echo $ARMS | awk '{for (i=NF;i>=1;i--) printf "%s ", $i}')
  for spec in $order; do
    a=${spec%%:*}; e=${spec#*:}
    B=$BIN; case $e in *BIN=*) B=$(echo "$e" | tr "+" "\n" | sed -n "s/^BIN=//p"); e=$(echo "$e" | tr "+" "\n" | grep -v "^BIN=" | tr "\n" "+"); [ -n "$e" ] || e=X=1 ;; esac
    stop
    for i in $(seq 1 40); do curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1 || break; sleep 1; done
    curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { echo "port $PORT already answers: stopping"; exit 1; }
    ts=$(echo " ${e//+/ } " | sed -n 's/.* TASKSET=\([^ ]*\).*/\1/p'); pre=""; [ -n "$ts" ] && pre="taskset -c $ts"
    env ${e//+/ } INTARSIA_SPEC_TRACE=1 $pre nohup $B serve --backend cuda --model $M35 --model-id m --max-batch 1 --max-seq 8192 --spec mtp --spec-k 2 --profile-every 0 --mem-budget 40 \
      --no-supervisor --host 127.0.0.1 --port $PORT > $O/serve-$a-r$r.log 2>&1 < /dev/null &
    for i in $(seq 1 120); do pgrep -f "$(pat)" >/dev/null || { echo "serve died ($a)"; tail -5 $O/serve-$a-r$r.log; exit 1; }; curl -sf -m 3 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; sleep 3; done
    echo "$a r$r up $(date +%T) free: $(free -g | awk 'NR==2{print $7}') GB"
    SPID=""
    if [ -z "${NOSAMPLE:-}" ]; then
      SP=$(pgrep -f "$(pat)" | head -1)
      taskset -c 11 python3 $R/src/scripts/cpupin/tsample.py $SP 0.1 > $O/ts-$a-r$r.tsv 2>/dev/null < /dev/null &
      SPID=$!
    fi
    python3 $C $PORT $R/cells/chat-short.json 2 warm$r > /dev/null 2>&1
    for c in $CELLS; do python3 $C $PORT $R/cells/$c.json $REPS c$r --out $O/txt-$a-r$r >> $O/$a-r$r.jsonl 2>&1; echo "$a r$r $c: $(tail -1 $O/$a-r$r.jsonl | cut -c1-160)"; done
    [ -n "$SPID" ] && kill $SPID 2>/dev/null
    curl -s http://127.0.0.1:$PORT/metrics | grep -E '^intarsia_(spec_(steps|proposed|accepted|copy)|graph_|decode_eager)' | sed 's/{[^}]*}//' | tr '\n' ' ' > $O/metrics-$a-r$r.txt
    echo "$a r$r metrics: $(cat $O/metrics-$a-r$r.txt)"
  done
done
echo "ab done"

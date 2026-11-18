#!/usr/bin/env bash
# postchain on spark1 (from cpupin's serve2.sh): per server one event in the middle of a run of requests: 2 warm-ups (chat-short), one REF request of
# 500 tokens (code-gen: the reference text of probe kind `a`; the probe, at 450 with INTARSIA_SPEC_COPY_PROBE_ONCE=2, does not fire in it), PRE
# requests of 300 tokens (code-gen), MID requests of 800 tokens (code-gen: the probe fires in the first; real n20 chains come at 590+ generated
# tokens, so only here), POST requests of 300 tokens; arms differ by env only (same binary), interleaved rounds with the arm order reversed on even
# rounds (ABBA). One server per arm and round; INTARSIA_SPEC_TRACE=1 always.
# postchain2: RSTART=<first round number> (ABBA order continues across jobs); REFCELL=code-gen.json makes REF 800 tokens (so probe kind `a` has a reference past g = 754). Analysis: scripts/postchain2/an2.py.
#   fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 60 --label "postchain: round 1" -- ssh spark1 'W=r1 ROUNDS=4 ARMS="base:X=1 a7:INTARSIA_SPEC_COPY=256+INTARSIA_SPEC_COPY_PROBE_ONCE=2+INTARSIA_SPEC_COPY_PROBE=450,a7" bash -s' < scripts/postchain2/serve.sh
# ARMS "name:ENV=val[+ENV2=val]" (X=1: none; BIN=<path> picks another binary); ROUNDS (2); PRE / POST (6 requests before / after the event).
set -uo pipefail
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
R=$HOME/intarsiaml-postchain2; W=${W:?W}; O=$R/out/$W; mkdir -p $O
H=/home/nvidia/.cache/huggingface/hub
M35=$(ls -d $H/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/* | head -1)
BIN=${BIN:-$R/bin/intarsia}; PORT=${PORT:-8297}; ROUNDS=${ROUNDS:-2}; REPS=${REPS:-3}
TAG=${TAG:-c1}; PRE=${PRE:-6}; POST=${POST:-10}; MID=${MID:-2}; ARMS=${ARMS:?ARMS}; C=$R/src/scripts/mcopy/cell.py
python3 $R/src/scripts/mcopy/prompts.py $R/cells > /dev/null
stop() { pkill -f "$R/bin/[a-z0-9-]+ serve.*--port $PORT" 2>/dev/null; for i in $(seq 1 30); do pgrep -f "$R/bin/[a-z0-9-]+ serve.*--port $PORT" >/dev/null || break; sleep 1; done; }
nohup bash $R/src/scripts/postchain2/wd.sh > $O/wd.log 2>&1 < /dev/null &
WD=$!
trap 'stop; kill $WD 2>/dev/null' EXIT
python3 - $R/cells <<'EOP'
import json, sys
for n in (300, 500):
    d = json.load(open(sys.argv[1] + "/code-gen.json")); d["max_tokens"] = n
    json.dump(d, open(sys.argv[1] + f"/code-gen{n}.json", "w"))
EOP
for r in $(seq ${RSTART:-1} $((${RSTART:-1} + ROUNDS - 1))); do
  order="$ARMS"; [ $((r % 2)) = 0 ] && order=$(echo $ARMS | awk '{for (i=NF;i>=1;i--) printf "%s ", $i}')
  for spec in $order; do
    a=${spec%%:*}; e=${spec#*:}
    stop
    for i in $(seq 1 40); do curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1 || break; sleep 1; done
    curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { echo "port $PORT already answers (a server of another run?): stopping"; exit 1; }
    # TASKSET=<cpu list> in an arm's env pins the server (cpu 5-9, 15-19: Cortex-X925, 0-4, 10-14: Cortex-A725)
    ts=$(echo " ${e//+/ } " | sed -n 's/.* TASKSET=\([^ ]*\).*/\1/p'); pre=""; [ -n "$ts" ] && pre="taskset -c $ts"
    env ${e//+/ } INTARSIA_SPEC_TRACE=1 $pre nohup $BIN serve --backend cuda --model $M35 --model-id m --max-batch 1 --max-seq 8192 --spec mtp --spec-k 2 --profile-every 0 --mem-budget 40 \
      --no-supervisor --host 127.0.0.1 --port $PORT > $O/serve-$a-r$r.log 2>&1 < /dev/null &
    for i in $(seq 1 120); do pgrep -f "$R/bin/[a-z0-9-]+ serve.*--port $PORT" >/dev/null || { echo "serve died ($a)"; tail -5 $O/serve-$a-r$r.log; exit 1; }; curl -sf -m 3 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; sleep 3; done
    echo "$a r$r up $(date +%T) free: $(free -g | awk 'NR==2{print $7}') GB"
    python3 $C $PORT $R/cells/chat-short.json 2 warm$r > /dev/null 2>&1
    python3 $C $PORT $R/cells/${REFCELL:-code-gen500.json} 1 $TAG --out $O/txt-$a-r$r >> $O/$a-r$r.jsonl 2>&1
    python3 $C $PORT $R/cells/code-gen300.json $PRE pre$r --out $O/txt-$a-r$r >> $O/$a-r$r.jsonl 2>&1
    # MID uses REF's tag: cell.py puts `[tag-rep] ` before the prompt, so MID rep 0 has REF's prompt (and greedy text) and probe `a` its reference.
    # TAG c1: the prompt of cpupin's pin1 r1 / mcdiag real3 r1, whose 800-token replies make real n20 chains (rep 0 at position 828, rep 1 at 665)
    python3 $C $PORT $R/cells/code-gen.json $MID $TAG --out $O/txt-$a-r$r-mid >> $O/$a-r$r.jsonl 2>&1
    python3 $C $PORT $R/cells/code-gen300.json $POST post$r --out $O/txt-$a-r$r >> $O/$a-r$r.jsonl 2>&1
    curl -s http://127.0.0.1:$PORT/metrics | grep -E '^intarsia_(spec_(steps|proposed|accepted|copy)|graph_|decode_eager)' | sed 's/{[^}]*}//' | tr '\n' ' ' > $O/metrics-$a-r$r.txt
    echo "$a r$r: $(grep -c '"toks"' $O/$a-r$r.jsonl) requests, shas $(python3 -c "import json,sys; print(' '.join(json.loads(l)['sha'][:4] for l in open('$O/$a-r$r.jsonl') if l.startswith('{')))"), chains $(grep -ac '^spec: copy token' $O/serve-$a-r$r.log), probes: $(grep -a '^spec: probe' $O/serve-$a-r$r.log | tr '\n' ';')"
  done
done
echo "serve2 done"

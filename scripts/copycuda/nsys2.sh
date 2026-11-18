#!/usr/bin/env bash
# copycuda nsys, runs ON spark1: per server one nsys collection (CUDA trace, graphs traced per node) that starts right after `nsys launch` (server start, load and warm-up included) and covers
# everything until the last request: 2 warm-ups (chat-short), PRE short requests (code-gen 300 tokens, each ends with an eager 1-draft MTP step), then MID code-gen requests of
# 800 tokens (MID0, MID1 ...). The server then stops. Per-server analysis: nsys_mode.py (slow / fast graph launches in order, eager kernels per stream) and nsys_req.py.
# ARMS "name:ENV=val[+ENV2=val]" in RUN order with distinct names (b1:X=1 k1:INTARSIA_SIDE_EAGER=0 ...). PRE (3), MID (2).
#   fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 40 --label "copycuda: nsys" -- ssh spark1 'W=ns1 ARMS="..." bash -s' < scripts/copycuda/nsys2.sh
set -uo pipefail
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
R=$HOME/intarsiaml-copycuda; W=${W:?W}; O=$R/out/$W; mkdir -p $O
H=/home/nvidia/.cache/huggingface/hub
M35=$(ls -d $H/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/* | head -1)
BIN=${BIN:-$R/bin/intarsia}; PORT=${PORT:-8397}; TAG=${TAG:-c1}; PRE=${PRE:-3}; MID=${MID:-2}; ARMS=${ARMS:?ARMS}; C=$R/src/scripts/mcopy/cell.py
python3 $R/src/scripts/mcopy/prompts.py $R/cells > /dev/null
pat() { echo "$R/bin/[a-z0-9-]+ serve.*--port $PORT"; }
stop() { pkill -f "$(pat)" 2>/dev/null; for i in $(seq 1 30); do pgrep -f "$(pat)" >/dev/null || break; sleep 1; done; }
nohup bash $R/src/scripts/copycuda/wd.sh > $O/wd.log 2>&1 < /dev/null &
WD=$!
( while :; do echo "$(date +%s.%N) $(curl -s -m 1 http://100.120.88.104:8000/metrics | grep -E '^vllm:(num_requests_running|num_requests_waiting|prompt_tokens_total|generation_tokens_total)\{' | awk '{printf "%s ", $2}')"; sleep 1; done ) > $O/vllm-${W}.log 2>&1 < /dev/null &
VP=$!
trap 'stop; kill $WD $VP 2>/dev/null' EXIT
python3 - $R/cells <<'EOP'
import json, sys
d = json.load(open(sys.argv[1] + "/code-gen.json")); d["max_tokens"] = 300
json.dump(d, open(sys.argv[1] + "/code-gen300.json", "w"))
EOP
for spec in $ARMS; do
  a=${spec%%:*}; e=${spec#*:}; S=se$a$$
  stop
  curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { echo "port $PORT already answers: stopping"; exit 1; }
  env ${e//+/ } INTARSIA_SPEC_TRACE=1 nohup nsys launch --session-new=$S --trace=cuda --cuda-graph-trace=node $BIN serve --backend cuda --model $M35 --model-id m --max-batch 1 --max-seq 8192 --spec mtp --spec-k 2 --profile-every 0 --mem-budget 40 \
    --no-supervisor --host 127.0.0.1 --port $PORT > $O/serve-$a.log 2>&1 < /dev/null &
  SPID=""
  for i in $(seq 1 60); do nsys sessions list < /dev/null 2>/dev/null | grep -q " $S" && break; sleep 0.5; done
  for i in $(seq 1 20); do nsys start --session=$S -o $O/$a --force-overwrite=true -s none --cpuctxsw=none --export=sqlite < /dev/null > $O/nsys-start-$a.log 2>&1 && break; sleep 0.5; done
  for i in $(seq 1 120); do pgrep -f "$(pat)" >/dev/null || { [ $i -gt 5 ] && { echo "serve died ($a)"; tail -5 $O/serve-$a.log; exit 1; }; }; curl -sf -m 3 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; sleep 3; done
  echo "$a up $(date +%T) free: $(free -g | awk 'NR==2{print $7}') GB"
  SP=""; for p in $(pgrep -f "$(pat)"); do [ "$(basename "$(readlink /proc/$p/exe 2>/dev/null)")" = intarsia ] && SP=$p; done
  [ -n "$SP" ] && { taskset -c 11 python3 $R/src/scripts/cpupin/tsample.py $SP 0.1 > $O/ts-$a.tsv 2>/dev/null < /dev/null & SPID=$!; }
  python3 $C $PORT $R/cells/chat-short.json 2 warm >> $O/$a.jsonl 2>&1
  python3 $C $PORT $R/cells/code-gen300.json $PRE pre >> $O/$a.jsonl 2>&1
  python3 $C $PORT $R/cells/code-gen.json $MID $TAG >> $O/$a.jsonl 2>&1
  for x in ${EXTRA:-}; do python3 $C $PORT $R/cells/$x.json 1 ex >> $O/$a.jsonl 2>&1; done   # copycuda: extra cells (one rep each) after MID, e.g. EXTRA="agent-edit copy3k" (chains fire)
  kill $SPID 2>/dev/null
  nsys stop --session=$S < /dev/null > $O/nsys-stop-$a.log 2>&1
  echo "$a: $(grep -c '"toks"' $O/$a.jsonl) requests, shas $(python3 -c "import json,sys; print(' '.join(json.loads(l)['sha'][:4] for l in open('$O/$a.jsonl') if l.startswith('{')))"), $(ls -la $O/$a.sqlite 2>/dev/null | awk '{print $5}') bytes sqlite"
  stop; nsys shutdown --session=$S < /dev/null > /dev/null 2>&1
done
echo "nsys done"

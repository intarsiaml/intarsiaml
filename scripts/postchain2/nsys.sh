#!/usr/bin/env bash
# postchain2 step 3, runs ON spark1: per server the serve.sh design up to MID0 (2 warm-ups, REF 500, PRE x6, MID0 800 tokens: an n20 server
# makes its real chain at 828 here), then an nsys collection (CUDA trace, graphs traced per node) over MID1 only, then the server stops.
# ARMS "name:ENV=val[+ENV2=val]" in RUN order with distinct names (e.g. b1:X=1 n1:INTARSIA_SPEC_COPY=20 n2:... b2:X=1). Analysis: nsys_an.py.
#   fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 30 --label "postchain2: nsys" -- ssh spark1 'W=ns1 ARMS="..." bash -s' < scripts/postchain2/nsys.sh
set -uo pipefail
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
R=$HOME/intarsiaml-postchain2; W=${W:?W}; O=$R/out/$W; mkdir -p $O
H=/home/nvidia/.cache/huggingface/hub
M35=$(ls -d $H/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/* | head -1)
BIN=${BIN:-$R/bin/intarsia}; PORT=${PORT:-8297}; TAG=${TAG:-c1}; ARMS=${ARMS:?ARMS}; C=$R/src/scripts/mcopy/cell.py
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
# MID1 alone: cell.py numbers reps from 0, so a 2-rep MID with rep 0 skipped = the [c1-1] prompt
sed 's/^for rep in range(reps):/for rep in range(int(__import__("os").environ.get("REP0", "0")), reps):/' $C > $O/cell.py
for spec in $ARMS; do
  a=${spec%%:*}; e=${spec#*:}; S=pc2$a$$
  stop
  curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { echo "port $PORT already answers: stopping"; exit 1; }
  env ${e//+/ } INTARSIA_SPEC_TRACE=1 nohup nsys launch --session-new=$S --trace=cuda --cuda-graph-trace=node $BIN serve --backend cuda --model $M35 --model-id m --max-batch 1 --max-seq 8192 --spec mtp --spec-k 2 --profile-every 0 --mem-budget 40 \
    --no-supervisor --host 127.0.0.1 --port $PORT > $O/serve-$a.log 2>&1 < /dev/null &
  for i in $(seq 1 120); do pgrep -f "$R/bin/[a-z0-9-]+ serve.*--port $PORT" >/dev/null || { [ $i -gt 5 ] && { echo "serve died ($a)"; tail -5 $O/serve-$a.log; exit 1; }; }; curl -sf -m 3 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; sleep 3; done
  echo "$a up $(date +%T) free: $(free -g | awk 'NR==2{print $7}') GB"
  python3 $C $PORT $R/cells/chat-short.json 2 warm > /dev/null 2>&1
  python3 $C $PORT $R/cells/code-gen500.json 1 $TAG >> $O/$a.jsonl 2>&1
  python3 $C $PORT $R/cells/code-gen300.json 6 pre >> $O/$a.jsonl 2>&1
  python3 $C $PORT $R/cells/code-gen.json 1 $TAG >> $O/$a.jsonl 2>&1
  nsys start --session=$S -o $O/$a --force-overwrite=true -s none --cpuctxsw=none --export=sqlite < /dev/null
  REP0=1 python3 $O/cell.py $PORT $R/cells/code-gen.json 2 $TAG >> $O/$a.jsonl 2>&1
  nsys stop --session=$S < /dev/null > $O/nsys-stop-$a.log 2>&1
  echo "$a: $(grep -c '"toks"' $O/$a.jsonl) requests, shas $(python3 -c "import json,sys; print(' '.join(json.loads(l)['sha'][:4] for l in open('$O/$a.jsonl') if l.startswith('{')))"), chains $(grep -ac '^spec: copy token' $O/serve-$a.log), $(ls -la $O/$a.sqlite 2>/dev/null | awk '{print $5}') bytes sqlite"
  stop; nsys shutdown --session=$S < /dev/null > /dev/null 2>&1
done
echo "nsys done"

#!/usr/bin/env bash
# sidebatch on spark2: per arm one `intarsia serve --max-batch 8` server, then gate.py's waves, metrics, counter lines, stop. Env only differs between arms.
#   fleetq run --host spark2 --quiet gpu,cpu --mem 50 --expect 40 --label "sidebatch: gates" -- ssh -n spark2 'W=g1 ARMS="u:X=1 e0:INTARSIA_SIDE_EAGER=0" WAVES=g8,g8s,s8,long,longmix bash -s' < scripts/sidebatch/serve.sh
# ARMS "name:ENV=val[+ENV2=val][:extra serve args]" (X=1: no env). 
set -uo pipefail
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
R=$HOME/intarsiaml-sidebatch; W=${W:?W}; O=$R/out/$W; mkdir -p $O
H=/home/nvidia/.cache/huggingface/hub
M35=$(ls -d $H/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/* | head -1)
BIN=${BIN:-$R/bin/intarsia}; PORT=${PORT:-8298}; ARMS=${ARMS:?ARMS}; WAVES=${WAVES:-g8,g8s,s8,long,longmix}; MAXSEQ=${MAXSEQ:-20480}
python3 $R/src/scripts/mcopy/prompts.py $R/cells > /dev/null
stop() { pkill -f "$R/bin/[a-z0-9-]+ serve.*--port $PORT" 2>/dev/null; for i in $(seq 1 40); do pgrep -f "$R/bin/[a-z0-9-]+ serve.*--port $PORT" >/dev/null || break; sleep 1; done; }
nohup bash $R/src/scripts/sidebatch/wd.sh > $O/wd.log 2>&1 < /dev/null &
WD=$!
trap 'stop; kill $WD 2>/dev/null' EXIT
for spec in $ARMS; do
  IFS=: read -r a e x <<< "$spec"
  stop
  curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { echo "port $PORT already answers (a server of another run?): stopping"; exit 1; }
  env ${e//+/ } nohup $BIN serve --backend cuda --model $M35 --model-id m --max-batch 8 --max-seq $MAXSEQ --profile-every 0 --mem-budget 40 \
    --no-supervisor --host 127.0.0.1 --port $PORT ${x//,/ } > $O/serve-$a.log 2>&1 < /dev/null &
  for i in $(seq 1 120); do pgrep -f "$R/bin/[a-z0-9-]+ serve.*--port $PORT" >/dev/null || { echo "serve died ($a)"; tail -5 $O/serve-$a.log; exit 1; }; curl -sf -m 3 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; sleep 3; done
  echo "$a up $(date +%T) free: $(free -g | awk 'NR==2{print $7}') GB env=[$e] args=[$x]"
  # warm-up (one short request) so the first batched graphs are not captured under the first wave's measurement
  python3 $R/src/scripts/sidebatch/gate.py $PORT $R/cells $O $a-warm warm > /dev/null 2>&1 < /dev/null; rm -f $O/$a-warm*
  python3 $R/src/scripts/sidebatch/gate.py $PORT $R/cells $O $a $WAVES < /dev/null > $O/client-$a.log 2>&1
  curl -s http://127.0.0.1:$PORT/metrics | grep -E '^intarsia_(spec_|graph_|batch_|decode_eager)' | sed 's/{[^}]*}//' > $O/metrics-$a.txt
  echo "$a done $(date +%T): $(grep -c sha $O/$a.jsonl) requests, errors $(grep -c error $O/$a.jsonl); counters: $(grep -a '^sidebatch:' $O/serve-$a.log | tail -1)"
  stop
done
echo "serve done"

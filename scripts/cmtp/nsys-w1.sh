#!/usr/bin/env bash
# Lane cmtp W1: nsys capture of a served 35B decode window on spark1 (run under fleetq run --quiet gpu,cpu, ssh spark1 'bash -s CASE' < this).
#   CASE=single: --max-batch 1 (MTP default), one 4k stream.  CASE=b16: --max-batch 16, 16 streams of ~1k tokens.
# Extra env: GT (graph trace: graph|node, default graph), ENVX (extra env for serve, e.g. a knob name; none shipped), TAG, GEN.
set -uo pipefail
ENVX=${ENVX:-}
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
R=$HOME/intarsiaml-cmtp; O=$R/out/w1; mkdir -p $O
H=/home/nvidia/.cache/huggingface/hub
M35=$(ls -d $H/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/* | head -1)
BIN=${BIN:-$R/target-dev/release/intarsia}
CASE=${1:?single|b16}; TAG=${TAG:-$CASE}; GT=${GT:-graph}; GEN=${GEN:-1200}; PORT=${PORT:-8096}
P4K=$R/src/scripts/dfh/ids-prompts/4k-0.txt
case $CASE in
  single) SARGS="--max-batch 1"; NS=1; CH=-; PF=$P4K ;;
  b16)    SARGS="--max-batch 16"; NS=16; CH=3700; PF=$P4K ;;
esac
SES=cmtp$$
cleanup() { nsys stop --session=$SES >/dev/null 2>&1; pkill -f "intarsia serve.*--port $PORT" 2>/dev/null; sleep 2; }
trap cleanup EXIT
env $ENVX nsys launch --session-new=$SES -t cuda,nvtx,osrt --cuda-graph-trace=$GT \
  $BIN serve --backend cuda --model $M35 --model-id m $SARGS --profile-every 0 --mem-budget 40 --max-seq 8192 --no-supervisor --host 127.0.0.1 --port $PORT \
  > $O/serve-$TAG.log 2>&1 < /dev/null &
for i in $(seq 1 80); do pgrep -f "intarsia serve" >/dev/null || { echo "serve died"; cat $O/serve-$TAG.log; exit 1; }; curl -sf -m 3 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; sleep 3; done
echo "up $(date +%T)"
python3 $R/src/scripts/cmtp/gen.py $PORT $NS 64 $PF $CH 2 warm > $O/warm-$TAG.txt 2>&1   # captures graphs
nsys start --session=$SES -o $O/$TAG --force-overwrite=true --sample=process-tree --cpuctxsw=none 2>&1 | tail -2
python3 $R/src/scripts/cmtp/gen.py $PORT $NS $GEN $PF $CH 1 prof > $O/prof-$TAG.txt 2>&1
nsys stop --session=$SES 2>&1 | tail -3
curl -s http://127.0.0.1:$PORT/metrics | grep -E '^intarsia_(spec|graph)' | sed 's/{[^}]*}//' > $O/metrics-$TAG.txt
cat $O/prof-$TAG.txt; grep -a -E "spec|graph" $O/serve-$TAG.log | head -8 | cut -c1-200

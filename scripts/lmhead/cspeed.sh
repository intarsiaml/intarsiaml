#!/usr/bin/env bash
# Lane lmhead-fp8-default, CUDA decode speed, FROM THE MAC inside a spark1 quiet slot (localeval runs here, the server on spark1):
#   fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 60 --label "..." -- scripts/lmhead/cspeed.sh <out dir>
# Visits (ABAB per mode): D = the default (recipe fp8 head), O = INTARSIA_LMHEAD_QUANT=off; mtp = the recipe default (MTP k2), plain = --spec off.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
O=${1:?out dir}; mkdir -p "$O"; PORT=8297
VISITS=${VISITS:-"mtp.D mtp.O mtp.D mtp.O plain.D plain.O plain.D plain.O"}
say() { echo "$(date +%T) $*" | tee -a "$O/steps.log"; }
S="ssh -n -o BatchMode=yes spark1"
stop() { $S 'test -f ~/lmh-cuda/serve.pid && kill $(cat ~/lmh-cuda/serve.pid) 2>/dev/null; sleep 3; test -f ~/lmh-cuda/serve.pid && kill -9 $(cat ~/lmh-cuda/serve.pid) 2>/dev/null; rm -f ~/lmh-cuda/serve.pid; true'; }
trap stop EXIT
n=0
for v in $VISITS; do
  n=$((n+1)); mode=${v%.*}; arm=${v#*.}; extra=""; [ $mode = plain ] && extra="--spec off"; envs=""; [ $arm = O ] && envs="INTARSIA_LMHEAD_QUANT=off"
  say "visit $n $v: starting server"; stop
  $S "cd ~/lmh-cuda && export PATH=\$HOME/.cargo/bin:/usr/local/cuda/bin:\$PATH && nohup env $envs target-head/release/intarsia serve --backend cuda --model models/Qwen3.8-27B-Quark-AWQ-MXFP4 --served-model-name qwen38-27b --max-seq 8192 --mem-budget 45 --profile-every 0 --no-supervisor $extra --host 0.0.0.0 --port $PORT > out/serve-$n-$v.log 2>&1 < /dev/null & echo \$! > ~/lmh-cuda/serve.pid"
  up=0; for i in $(seq 1 120); do curl -sf -m 3 http://spark1:$PORT/health > /dev/null 2>&1 && { up=1; break; }; sleep 3; done
  [ $up = 1 ] || { say "visit $n $v: server did not come up"; continue; }
  $S "grep -ah 'lm_head quantised\|speculative decoding' ~/lmh-cuda/out/serve-$n-$v.log | head -2 | cut -c1-160" | tee -a "$O/steps.log"
  ( cd /opt/code/deepeval && uv run localeval speed --url http://spark1:$PORT/v1 --model qwen38-27b --thinking off --sizes 1024 --gen 128 --reps 3 \
      --label "lmh-cuda-$n-$v" --recipe "intarsia serve cuda 27B $mode head=$arm (lane lmhead-fp8-default)" --wait-idle 60 ) > "$O/speed-$n-$v.log" 2>&1 < /dev/null
  say "item $n $v speed: exit $? $(grep -aoE 'results/[^ ]+' "$O/speed-$n-$v.log" | tail -1)"
done
stop; say done

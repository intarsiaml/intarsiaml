#!/usr/bin/env bash
# Window C, spark1 side (vLLM down, INSIDE scripts/cuda-decode/window.sh, driven by scripts/cuda-prefill3/wC-localeval.sh):
# for each arm argument (name:ENV=V,ENV=V; tree p3-dev) start `intarsia serve` on 0.0.0.0:8090 (the localeval target
# spark-intarsia-35b of deploy/localeval/targets-snippet.toml), print "READY <arm>", wait until the driver creates
# ~/intarsia-decfuse/out/p3C/<arm>.done (at most ARM_MAX_SEC, default 420 s), stop the server.
set -u
P=$HOME/intarsia-decfuse
BIN=$P/p3-dev/target/acc/release/intarsia
M=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
O=$P/out/p3C; mkdir -p $O; rm -f $O/*.done
gb=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1024 / 1024 )); echo "MemAvailable ${gb} GB"; (( gb >= 60 )) || { echo "ABORT: memory"; exit 1; }
echo "bin: $($BIN --version 2>&1 | head -1)"
(( $# > 0 )) || { echo "usage: wC-serve.sh <name:ENV=V,..> ..."; exit 2; }
for spec in "$@"; do
  arm=${spec%%:*}; envs=""; [[ "$spec" == *:* ]] && envs=${spec#*:}; envs=${envs//,/ }
  echo "== arm $arm env '$envs' $(date +%T)"
  nohup env $envs $BIN serve --backend cuda --model $M --port 8090 --host 0.0.0.0 --served-model-name intarsia-35b --max-seq 20480 --mem-budget 45 --profile-every 0 > $O/serve-$arm.log 2>&1 < /dev/null &
  spid=$!
  up=0; for i in $(seq 1 120); do curl -sf http://127.0.0.1:8090/health >/dev/null && { up=1; break; }; sleep 2; done
  if [ $up = 1 ]; then
    echo "READY $arm $(date +%T)"
    for i in $(seq 1 ${ARM_MAX_SEC:-420}); do [ -f $O/$arm.done ] && break; sleep 1; done
    [ -f $O/$arm.done ] && echo "arm $arm done $(date +%T)" || echo "arm $arm: no done marker after ${ARM_MAX_SEC:-420} s"
  else
    echo "arm $arm: server did not come up (see $O/serve-$arm.log)"
  fi
  kill $spid; for i in $(seq 1 30); do kill -0 $spid 2>/dev/null || break; sleep 1; done; kill -9 $spid 2>/dev/null
  sleep 2
done
echo "== p3C done $(date +%T)"

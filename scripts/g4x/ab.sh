#!/usr/bin/env bash
# G4X: one round of interleaved speed A/B arms for Gemma-4-26B-A4B-NVFP4 on spark2 (run INSIDE a fleetq slot, --quiet gpu,cpu,
# --mem >= 30). A fresh `intarsia serve` per arm, `bench speed` is the only client.
#   ab.sh <round> <name:binary[:ENV=VAL[,ENV=VAL]]> ...      SERVE_ARGS=--a=b+--c=d among the env entries adds `serve` arguments
# Env: SIZES (128,1024,4096,16384) GEN (128) REPS (3) OUT (~/intarsiaml-g4x/replay-out/ab; replay-out survives the rsync --delete of scripts/deploy.sh) BENCH_ARGS (extra `bench speed` arguments)
set -u
M=${MODEL_DIR:-$HOME/models/Gemma-4-26B-A4B-NVFP4}; NAME=g4x; PORT=${PORT:-8091}
SIZES=${SIZES:-128,1024,4096,16384}; GEN=${GEN:-128}; REPS=${REPS:-3}; OUT=${OUT:-$HOME/intarsiaml-g4x/replay-out/ab}
RND=$1; shift
gb=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1024 / 1024 )); echo "MemAvailable ${gb} GB"; (( gb >= 45 )) || { echo "ABORT: memory"; exit 1; }
for spec in "$@"; do
  arm=${spec%%:*}; rest=${spec#*:}; bin=${rest%%:*}; envs=""; [[ "$rest" == *:* ]] && envs=${rest#*:}; envs=${envs//,/ }
  sargs=""; for e in $envs; do [[ $e == SERVE_ARGS=* ]] && sargs=${e#SERVE_ARGS=}; done; sargs=${sargs//+/ }
  envs=$(for e in $envs; do [[ $e == SERVE_ARGS=* ]] || printf '%s ' "$e"; done)
  out=$OUT/$arm-r$RND; rm -rf $out; mkdir -p $out
  echo "== round $RND arm $arm (bin $bin env '$envs') $(date +%T)"
  nohup env $envs $bin serve --backend cuda --model $M --port $PORT --host 127.0.0.1 --served-model-name $NAME --max-seq ${MAX_SEQ:-20480} --mem-budget ${MEM:-30} --profile-every 0 $sargs > $out/serve.log 2>&1 < /dev/null &
  spid=$!
  up=0; for i in $(seq 1 150); do curl -sf http://127.0.0.1:$PORT/health >/dev/null && { up=1; break; }; sleep 2; done
  [ $up = 1 ] || { echo "server did not come up, see $out/serve.log"; kill $spid; exit 1; }
  $bin bench speed --url http://127.0.0.1:$PORT --model $NAME --sizes $SIZES --gen $GEN --reps $REPS --warmup 1 --thinking off --label g4x-$arm-r$RND --out-dir $out --at-most ${BENCH_ARGS:-} > $out/bench.log 2>&1; echo $? > $out/bench.rc
  kill $spid; for i in $(seq 1 30); do kill -0 $spid 2>/dev/null || break; sleep 1; done; kill -9 $spid 2>/dev/null
  tail -n 8 $out/bench.log | cut -c1-200
  grep -aE "WARNING|graph" $out/serve.log | head -5 | cut -c1-300
  sleep 2
done
echo "== ab round $RND done $(date +%T)"

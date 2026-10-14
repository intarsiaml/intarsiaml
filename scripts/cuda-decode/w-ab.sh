#!/usr/bin/env bash
# One round of A/B arms on spark1 (INSIDE scripts/cuda-decode/window.sh: vLLM is down). Same procedure as
# scripts/prefill-cuda/ab-ours.sh / scripts/phase1b-acceptance.sh stage_ab: a fresh `intarsia serve` per arm, the bench is
# the only client, `bench speed --warmup 1 --thinking off --require-graphs`, --profile-every 0.
#   w-ab.sh <35b|8b> <round> <arm> ...           arm = name:tree[:ENV=VAL[,ENV=VAL]]   (trees under ~/intarsia-decfuse)
#   e.g.  w-ab.sh 35b 1 base:base dev:dev nofuse:dev:INTARSIA_NO_FUSION=1
# Env: SIZES (128,1024,4096) GEN (128) REPS (5) OUT (~/intarsia-decfuse/out/ab)
set -u
P=$HOME/intarsia-decfuse
case "$1" in
  35b) M=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5; MEM=45; NAME=intarsia-35b ;;
  8b)  M=/home/nvidia/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218; MEM=40; NAME=intarsia-8b ;;
  *) echo "model 35b|8b"; exit 2 ;;
esac
MODEL=$1; RND=$2; shift 2
SIZES=${SIZES:-128,1024,4096}; GEN=${GEN:-128}; REPS=${REPS:-5}; PORT=8090; AB=${OUT:-$P/out/ab}
gb=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1024 / 1024 )); echo "MemAvailable ${gb} GB"; (( gb >= 60 )) || { echo "ABORT: memory"; exit 1; }
for spec in "$@"; do
  arm=${spec%%:*}; rest=${spec#*:}; tree=${rest%%:*}; envs=""; [[ "$rest" == *:* ]] && envs=${rest#*:}; envs=${envs//,/ }
  bin=$P/$tree/target/acc/release/intarsia
  out=$AB/$MODEL-$arm-r$RND; rm -rf $out; mkdir -p $out
  echo "== $MODEL round $RND arm $arm (tree $tree env '$envs') $(date +%T)"
  nohup env $envs $bin serve --backend cuda --model $M --port $PORT --host 127.0.0.1 --served-model-name $NAME --max-seq ${MAX_SEQ:-20480} --mem-budget $MEM --profile-every 0 > $out/serve.log 2>&1 < /dev/null &
  spid=$!
  up=0; for i in $(seq 1 120); do curl -sf http://127.0.0.1:$PORT/health >/dev/null && { up=1; break; }; sleep 2; done
  [ $up = 1 ] || { echo "server did not come up, see $out/serve.log"; kill $spid; exit 1; }
  $bin bench speed --url http://127.0.0.1:$PORT --model $NAME --sizes $SIZES --gen $GEN --reps $REPS --warmup 1 --thinking off --label cudadecode-$MODEL-$arm-r$RND --out-dir $out --require-graphs > $out/bench.log 2>&1; echo $? > $out/bench.rc
  kill $spid; for i in $(seq 1 30); do kill -0 $spid 2>/dev/null || break; sleep 1; done; kill -9 $spid 2>/dev/null
  tail -n 6 $out/bench.log | cut -c1-160
  sleep 2
done
echo "== w-ab $MODEL round $RND done $(date +%T)"

#!/usr/bin/env bash
# think-len: run ON spark1 under `fleetq run --host spark1 --quiet gpu ...`. Starts the soak binary (main 45e6ce56b, stamp ab35c-45e6ce56b023)
# in the Bc shape (--max-batch 2 --prefix-store on, bf16 KV), runs probe.py against it, stops it, then again with --kv-dtype fp8.
# Never touches the vLLM services. usage: spark1_probe.sh <arms, default "bf16 fp8"> [kinds, default on,off,tf]
# arms: bf16 | fp8 | nomoe (INTARSIA_CUDA_DISABLE=moe_experts_mma_bf16) | simt (SIMT attention instead of flash / split-KV) | nomoe-simt | bf16b (a repeat of bf16)
set -uo pipefail
D=~/intarsia-think-len; BIN=$D/bin/intarsia; OUT=$D/out; PORT=8291
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
: "${FLEETQ_JOB:?run under fleetq run}"
for arm in ${1:-bf16 fp8}; do
  extra=""; denv=""
  SIMT=attn_decode_bf16_splitkv,attn_prefill_bf16_splitkv,attn_prefill_bf16_flash
  case $arm in
    fp8) extra="--kv-dtype fp8";;
    nomoe) denv=moe_experts_mma_bf16;;
    simt) denv=$SIMT;;
    nomoe-simt) denv=moe_experts_mma_bf16,$SIMT;;
  esac
  echo "[$(date +%T)] serve $arm"
  env ${denv:+INTARSIA_CUDA_DISABLE=$denv} $BIN serve --backend cuda --model $M35 --port $PORT --host 127.0.0.1 --served-model-name guidance --max-seq 32768 --mem-budget 40 \
     --profile-every 0 --max-batch 2 --prefix-store on $extra > $OUT/serve-$arm.log 2>&1 < /dev/null &
  SP=$!
  up=0; for i in $(seq 1 120); do curl -sf http://127.0.0.1:$PORT/health >/dev/null && { up=1; break; }; kill -0 $SP 2>/dev/null || break; sleep 2; done
  if [[ $up != 1 ]]; then echo "serve $arm did not come up"; tail -20 $OUT/serve-$arm.log; kill $SP 2>/dev/null; continue; fi
  echo "[$(date +%T)] up; MemAvailable MB $(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)"
  python3 $D/probe.py http://127.0.0.1:$PORT guidance $D/probe-inputs.json $OUT/probe-intarsia-$arm.json ${2:-on,off,tf}
  echo "[$(date +%T)] MemAvailable MB $(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)"
  curl -sf http://127.0.0.1:$PORT/metrics > $OUT/serve-$arm.metrics
  kill $SP; for i in $(seq 1 30); do kill -0 $SP 2>/dev/null || break; sleep 1; done; kill -9 $SP 2>/dev/null
  sleep 3
done
echo "[$(date +%T)] done"

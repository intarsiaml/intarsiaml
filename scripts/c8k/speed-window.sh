#!/usr/bin/env bash
# Lane c8k: same-window prefill / batched-decode comparison of IntarsiaML CUDA at prefill chunk 2048 vs 8192, with vLLM in the window.
# Mac side, run INSIDE a fleetq slot on spark2 that takes down retro:
#   fleetq run --host spark2 --quiet gpu,cpu --mem 85 --takes-down retro --expect 130 --label "..." -- scripts/c8k/speed-window.sh <outname> <rounds> <tree>
# The retro container (missioncntrl-guidance) is stopped, a test vLLM is started with its launcher flags (as lane vr: --max-num-seqs 16 so
# that 4 streams are not capped; prefill does not depend on it), and the retro container is started again on exit (trap), then checked.
# Per round every arm runs once (round 2 reversed): vllm | i2048 | i8192. Per intarsia arm: a solo server (`serve`, as in lane pf) for the
# localeval prefill sweep 1k/4k/16k (gen 128, 3 reps + 1 warmup), then a `--max-batch 4` server for 4 streams at 1k and 4k
# (localeval `--concurrency 4`: aggregate decode) and the ITL driver (scripts/c8k/itl4.py: burst + stagger). The busy chunk follows
# the chunk of the arm (the 35B on CUDA defaults it to the solo chunk). Same three calls for the vLLM arm.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
H=spark2; HN=spark-fa00; PORT=8190; PROD=missioncntrl-guidance; NAME=c8k-vllm
OUTN=$1; ROUNDS=$2; TREE=$3
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$D/../.."
OUT=$REPO/docs/results/2026-10-24-c8k-prefill-chunk-8192/raw/$OUTN; mkdir -p "$OUT"
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
rsh() { ssh -n -o BatchMode=yes "$H" "$@"; }
log() { echo "[$(date +%T)] $*" | tee -a "$OUT/run.log"; }
SPID=""
stop_server() { [[ -n "$SPID" ]] && { rsh "kill $SPID 2>/dev/null; for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do kill -0 $SPID 2>/dev/null || exit 0; sleep 1; done; kill -9 $SPID 2>/dev/null; sleep 3" ; SPID=""; }; }
T_DOWN=$(date +%s)
restore() {
  stop_server
  rsh "docker rm -f $NAME >/dev/null 2>&1; docker start $PROD" >> "$OUT/run.log" 2>&1
  for i in $(seq 1 90); do curl -sf -m 5 http://$HN:8000/v1/models > /dev/null 2>&1 && break; sleep 5; done
  log "restore: retro healthy $(( $(date +%s) - T_DOWN )) s after the stop"
  curl -s -m 60 http://$HN:8000/v1/models | head -c 300 | tee -a "$OUT/run.log"; echo
  curl -s -m 120 http://$HN:8000/v1/chat/completions -H 'content-type: application/json' -d '{"model":"guidance","messages":[{"role":"user","content":"Say OK."}],"max_tokens":16,"chat_template_kwargs":{"enable_thinking":false}}' | head -c 400 | tee -a "$OUT/run.log"; echo
}
trap restore EXIT
log "stopping $PROD"; rsh "docker stop $PROD" >> "$OUT/run.log" 2>&1
VLLM_RECIPE="docker run $NAME missioncntrl/vllm-openai:65b7662d vllm serve nvidia/Qwen3.6-35B-A3B-NVFP4 --served-model-name guidance capcom --quantization modelopt_fp4 --max-model-len 131072 --max-num-seqs 16 --max-num-batched-tokens 8192 --kv-cache-dtype fp8 --kv-cache-memory-bytes 8589934592 --gpu-memory-utilization 0.32 --enable-prefix-caching --enable-chunked-prefill (retro launcher flags, max-num-seqs 16)"
start_vllm() {
  rsh "docker rm -f $NAME >/dev/null 2>&1; docker run -d --name $NAME --gpus all --network host --ipc host --shm-size 16g -v /home/nvidia/.cache/huggingface:/root/.cache/huggingface -e HF_HUB_OFFLINE=1 -e VLLM_USE_FLASHINFER_MOE_FP4=0 -e VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel --entrypoint vllm missioncntrl/vllm-openai:65b7662d serve nvidia/Qwen3.6-35B-A3B-NVFP4 --host 100.96.95.89 --port 8000 --served-model-name guidance capcom --quantization modelopt_fp4 --max-model-len 131072 --max-num-seqs 16 --max-num-batched-tokens 8192 --kv-cache-dtype fp8 --kv-cache-memory-bytes 8589934592 --gpu-memory-utilization $1 --enable-prefix-caching --enable-prompt-tokens-details --enable-chunked-prefill --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3" >> "$OUT/run.log" 2>&1
  local i
  for i in $(seq 1 100); do
    curl -sf -m 5 http://$HN:8000/v1/models > /dev/null 2>&1 && { log "vllm ready after $((i*5)) s (util $1)"; return 0; }
    rsh "docker ps -q --filter name=$NAME | grep -q ." || { log "vllm container died"; return 1; }
    sleep 5
  done; return 1
}
start_vllm 0.32 || { rsh "docker logs $NAME 2>&1 | tail -20" >> "$OUT/run.log"; start_vllm 0.40; } || { log "vllm did not start"; exit 1; }
rsh "docker logs $NAME 2>&1 | grep -i -E 'KV cache|Maximum concurrency' | head -3" | tee -a "$OUT/run.log"
le() { # le <label> <url> <model> <recipe> <speed args...>
  local label=$1 url=$2 model=$3 recipe=$4; shift 4
  ( cd /opt/code/deepeval && FLEETQ_JOB=$FLEETQ_JOB uv run localeval speed --url "$url" --model "$model" --thinking off --label "$label" --recipe "$recipe" "$@" ) > "$OUT/$label.out" 2>&1
  echo "$label: exit $? $(grep -aE 'saved .*result.json' "$OUT/$label.out" | head -1 | sed 's/.*saved //')" | tee -a "$OUT/index.txt"
  grep -aE 'tok rep|streams: aggregate' "$OUT/$label.out" | cut -c1-200 | tee -a "$OUT/run.log"
}
itl() { # itl <label> <url> <model>
  ( cd /opt/code/deepeval && FLEETQ_JOB=$FLEETQ_JOB uv run python "$D/itl4.py" --url "$2" --model "$3" --reps 2 --out "$OUT/$1.json" ) > "$OUT/$1.out" 2>&1
  echo "$1: exit $?" | tee -a "$OUT/index.txt"; grep -aE '^\[itl4\]' "$OUT/$1.out" | cut -c1-330 | tee -a "$OUT/run.log"
}
serve() { # serve <name> <chunk> <extra args>   -> sets SPID, waits healthy
  local name=$1 chunk=$2; shift 2
  SPID=$(rsh "cd ~/intarsia-decfuse; nohup ~/intarsia-decfuse/$TREE/target/acc/release/intarsia serve --backend cuda --model $M35 --port $PORT --host 0.0.0.0 --served-model-name intarsia-35b --max-seq 20480 --mem-budget 45 --profile-every 0 --prefill-chunk $chunk $* > ~/intarsia-decfuse/out/c8k-serve-$name.log 2>&1 < /dev/null & echo \$!")
  local i up=0
  for i in $(seq 1 150); do curl -sf "http://$HN:$PORT/health" > /dev/null && { up=1; break; }; sleep 2; done
  if [[ $up != 1 ]]; then log "server $name did not come up"; rsh "tail -5 ~/intarsia-decfuse/out/c8k-serve-$name.log" | tee -a "$OUT/run.log"; stop_server; return 1; fi
  rsh "grep -aE 'prefill chunk|continuous batching' ~/intarsia-decfuse/out/c8k-serve-$name.log | head -3" | cut -c1-200 | tee -a "$OUT/run.log"
}
run_arm() { # run_arm <round> <arm>
  local r=$1 arm=$2 chunk
  log "== round $r arm $arm"
  if [[ $arm == vllm ]]; then
    local u=http://$HN:8000/v1
    le "c8k-vllm-r$r" $u guidance "$VLLM_RECIPE" --sizes 1024,4096,16384 --gen 128 --reps 3
    le "c8k-vllm-conc1k-r$r" $u guidance "$VLLM_RECIPE" --sizes 1024 --gen 128 --reps 1 --concurrency 4
    le "c8k-vllm-conc4k-r$r" $u guidance "$VLLM_RECIPE" --sizes 4096 --gen 128 --reps 1 --concurrency 4
    itl "c8k-vllm-itl-r$r" $u guidance
    return
  fi
  chunk=${arm#i}
  local rec="intarsia serve --backend cuda --max-seq 20480 --mem-budget 45 --prefill-chunk $chunk tree=$TREE (lt default)"
  serve "$arm-r$r-solo" $chunk && {
    le "c8k-$arm-r$r" http://$HN:$PORT/v1 intarsia-35b "$rec" --sizes 1024,4096,16384 --gen 128 --reps 3
    stop_server; }
  serve "$arm-r$r-b4" $chunk --max-batch 4 && {
    le "c8k-$arm-conc1k-r$r" http://$HN:$PORT/v1 intarsia-35b "$rec --max-batch 4" --sizes 1024 --gen 128 --reps 1 --concurrency 4
    le "c8k-$arm-conc4k-r$r" http://$HN:$PORT/v1 intarsia-35b "$rec --max-batch 4" --sizes 4096 --gen 128 --reps 1 --concurrency 4
    itl "c8k-$arm-itl-r$r" http://$HN:$PORT/v1 intarsia-35b
    curl -s http://$HN:$PORT/metrics | grep -aE '^intarsia_batch_(itl|chunk_call)_seconds|^intarsia_batch_(mixed|prefill)' > "$OUT/c8k-$arm-metrics-r$r.txt" 2>&1
    stop_server; }
}
for r in $(seq 1 $ROUNDS); do
  if (( r % 2 == 0 )); then for a in i8192 i2048 vllm; do run_arm $r $a; done
  else for a in vllm i2048 i8192; do run_arm $r $a; done; fi
done
log "== done"

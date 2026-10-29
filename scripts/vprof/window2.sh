#!/usr/bin/env bash
# Lane vprof: window 2 (ncu). spark1-vllm (flight/guidance/capcom) taken down by fleetq (--takes-down), vLLM run with the vr16 flags
# (--max-num-seqs 16) under ncu (cuda graphs on, --graph-profiling node, vLLM /start_profile), one 16-stream burst; service restored before release.
#   fleetq run --host spark1 --takes-down flight,guidance,capcom --quiet gpu,cpu --mem 45 --expect 30 --label "vprof: vLLM B=16 ncu" -- scripts/vprof/window2.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
HOST=spark1; H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT=${OUT:-$H/../../docs/results/2026-10-29-vprof-vllm-step/raw/ncu}; mkdir -p "$OUT"
R='$HOME/intarsiaml-vprof'; NAME=vprof-vllm; PORT=8011
IMG=missioncntrl/vllm-openai:65b7662d
rsh() { ssh -n -o BatchMode=yes "$HOST" "$@"; }
RESTORED=0
restore() {
  (( RESTORED )) && return
  echo "== restore $(date +%T)"
  rsh "docker kill -s INT $NAME >/dev/null 2>&1; sleep 5; docker rm -f $NAME >/dev/null 2>&1; docker start missioncntrl-guidance"
  local i; for i in $(seq 1 150); do
    rsh "curl -sf -m 5 http://100.120.88.104:8000/v1/models >/dev/null" && { RESTORED=1; echo "restored: /v1/models answers $(date +%T)"; return; }; sleep 5; done
  echo "RESTORE FAILED: /v1/models does not answer"
}
trap 'restore' EXIT INT TERM
rsh "mkdir -p $R/out"
scp -q -o BatchMode=yes "$H/burst.py" "$H/tail.py" "$H/../../tools/prompts/corpus_docs.txt" "$HOST:intarsiaml-vprof/out/"
rsh "docker inspect missioncntrl-guidance --format '{{json .Config.Cmd}}'" > "$OUT/vllm-original-cmd.json"
rsh "docker inspect missioncntrl-guidance --format '{{json .Config.Env}}'" > "$OUT/vllm-original-env.json"
rsh "docker run --rm -v /opt/nvidia/nsight-compute/2025.3.1:/opt/nvidia/nsight-compute/2025.3.1:ro --entrypoint /opt/nvidia/nsight-compute/2025.3.1/ncu $IMG --version" || { echo "ncu does not run in the image"; exit 1; }
echo "== stop original $(date +%T)"
rsh "docker stop missioncntrl-guidance >/dev/null"
# vr16 flags (docs/results/2026-10-27-vr16-vllm-seqs16/raw/vllm-seqs16-launcher.sh) under ncu; host ncu mounted into the container
rsh "docker run -d --name $NAME --restart no --gpus all --network host --ipc host --shm-size 16g \
  -v /home/nvidia/.cache/huggingface:/root/.cache/huggingface --cap-add SYS_ADMIN -v /opt/nvidia/nsight-compute/2025.3.1:/opt/nvidia/nsight-compute/2025.3.1:ro \
  -v \$HOME/intarsiaml-vprof/out:/out \
  -e HF_HUB_OFFLINE=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e VLLM_USE_FLASHINFER_MOE_FP4=0 -e VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel \
  --entrypoint /opt/nvidia/nsight-compute/2025.3.1/ncu $IMG \
  --target-processes all --profile-from-start off --graph-profiling node --kernel-name-base demangled \
  -k 'regex:marlin_moe_wna16|gated_delta_rule|GemmUniversal|BatchPrefillWithPagedKV|marlin::Marlin|causal_conv1d' --launch-count 840 \
  --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed \
  --csv --log-file /out/ncu.csv -f \
  vllm serve nvidia/Qwen3.6-35B-A3B-NVFP4 --host 127.0.0.1 --port $PORT --served-model-name guidance capcom \
  --quantization modelopt_fp4 --max-model-len 131072 --max-num-seqs 16 --max-num-batched-tokens 8192 \
  --kv-cache-dtype fp8 --kv-cache-memory-bytes 8589934592 --gpu-memory-utilization 0.32 \
  --enable-prefix-caching --enable-prompt-tokens-details --enable-chunked-prefill \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 --profiler-config '{\"profiler\":\"cuda\"}'" || exit 1
rsh "docker inspect $NAME --format '{{json .Config.Cmd}}'" > "$OUT/vprof-ncu-cmd.json"
echo "== waiting for vLLM $(date +%T)"
up=0; for i in $(seq 1 180); do
  rsh "curl -sf -m 5 http://127.0.0.1:$PORT/v1/models >/dev/null" && { up=1; break; }
  rsh "docker ps -q -f name=$NAME" | grep -q . || { echo "container exited"; break; }; sleep 5; done
rsh "docker logs $NAME 2>&1" > "$OUT/vllm-startup.log"
if [[ $up != 1 ]]; then echo "vLLM did not come up"; tail -20 "$OUT/vllm-startup.log"; exit 1; fi
echo "== vLLM up $(date +%T)"
C="python3 \$HOME/intarsiaml-vprof/out/burst.py $PORT"; CORP=\$HOME/intarsiaml-vprof/out/corpus_docs.txt
{ rsh "$C 2 32 warm $CORP"; sleep 3
  rsh "($C 16 256 a $CORP > \$HOME/intarsiaml-vprof/out/burst-a.txt 2>&1 &) ; sleep 6; curl -s -m 20 -X POST http://127.0.0.1:$PORT/start_profile -w 'start_profile http %{http_code}\n'"
  # ncu serialises + replays the profiled kernels; the stalled engine keeps B=16 for those steps. Wait for the csv to stop growing.
  for i in $(seq 1 50); do sleep 6; rsh "grep -q '^burst a' \$HOME/intarsiaml-vprof/out/burst-a.txt" && break; done
  rsh "cat \$HOME/intarsiaml-vprof/out/burst-a.txt | cut -c1-160; ls -l \$HOME/intarsiaml-vprof/out/ncu.csv; wc -l \$HOME/intarsiaml-vprof/out/ncu.csv"
  rsh "curl -s -m 20 -X POST http://127.0.0.1:$PORT/stop_profile -w 'stop_profile http %{http_code}\n'"
} 2>&1 | tee "$OUT/bursts.txt"
echo "== stop vLLM (SIGINT) $(date +%T)"
rsh "docker kill -s INT $NAME >/dev/null"
for i in $(seq 1 120); do rsh "docker ps -q -f name=$NAME" | grep -q . || break; sleep 5; done   # up to 10 min
rsh "docker ps -q -f name=$NAME" | grep -q . && { echo "container still running after 10 min: not removing"; exit 1; }
rsh "docker logs $NAME 2>&1 | tail -30" > "$OUT/vllm-shutdown.log"
rsh "docker run --rm --entrypoint chown -v \$HOME/intarsiaml-vprof/out:/out $IMG -R \$(id -u):\$(id -g) /out; docker rm -f $NAME >/dev/null 2>&1; ls -la \$HOME/intarsiaml-vprof/out"
echo "== bring the standing service back $(date +%T)"
restore
rsh "cd \$HOME/intarsiaml-vprof/out && ls -la && gzip -k -f ncu.csv" 2>&1 | tee "$OUT/ncu-ls.log"
scp -q "$HOST:intarsiaml-vprof/out/ncu.csv.gz" "$OUT/" ; echo "== done $(date +%T)"

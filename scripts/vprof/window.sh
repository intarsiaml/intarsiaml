#!/usr/bin/env bash
# Lane vprof: ONE spark1 window. spark1-vllm (flight/guidance/capcom) taken down by fleetq (--takes-down), vLLM run with the vr16 flags
# (--max-num-seqs 16) under nsys (cuda graphs on, --cuda-graph-trace=node), driven by 3 bursts; service restored before release.
#   fleetq run --host spark1 --takes-down flight,guidance,capcom --quiet gpu,cpu --mem 45 --expect 30 --label "vprof: vLLM B=16 nsys" -- scripts/vprof/window.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
HOST=spark1; H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT=${OUT:-$H/../../docs/results/2026-10-29-vprof-vllm-step/raw}; mkdir -p "$OUT"
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
echo "== stop original $(date +%T)"
rsh "docker stop missioncntrl-guidance >/dev/null"
# vr16 flags (docs/results/2026-10-27-vr16-vllm-seqs16/raw/vllm-seqs16-launcher.sh) under nsys; host nsys mounted into the container
rsh "docker run -d --name $NAME --restart no --gpus all --network host --ipc host --shm-size 16g \
  -v /home/nvidia/.cache/huggingface:/root/.cache/huggingface -v /opt/nvidia/nsight-systems/2025.3.2:/opt/nvidia/nsight-systems/2025.3.2:ro \
  -v \$HOME/intarsiaml-vprof/out:/out \
  -e HF_HUB_OFFLINE=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e VLLM_USE_FLASHINFER_MOE_FP4=0 -e VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel \
  --entrypoint /opt/nvidia/nsight-systems/2025.3.2/target-linux-sbsa-armv8/nsys $IMG \
  profile -o /out/nsys --force-overwrite true --trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-graph-trace=node --trace-fork-before-exec=true --stats=false \
  vllm serve nvidia/Qwen3.6-35B-A3B-NVFP4 --host 127.0.0.1 --port $PORT --served-model-name guidance capcom \
  --quantization modelopt_fp4 --max-model-len 131072 --max-num-seqs 16 --max-num-batched-tokens 8192 \
  --kv-cache-dtype fp8 --kv-cache-memory-bytes 8589934592 --gpu-memory-utilization 0.32 \
  --enable-prefix-caching --enable-prompt-tokens-details --enable-chunked-prefill \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3" || exit 1
rsh "docker inspect $NAME --format '{{json .Config.Cmd}}'" > "$OUT/vprof-cmd.json"
echo "== waiting for vLLM $(date +%T)"
up=0; for i in $(seq 1 180); do
  rsh "curl -sf -m 5 http://127.0.0.1:$PORT/v1/models >/dev/null" && { up=1; break; }
  rsh "docker ps -q -f name=$NAME" | grep -q . || { echo "container exited"; break; }; sleep 5; done
rsh "docker logs $NAME 2>&1" > "$OUT/vllm-startup.log"
if [[ $up != 1 ]]; then echo "vLLM did not come up"; tail -20 "$OUT/vllm-startup.log"; exit 1; fi
echo "== vLLM up $(date +%T)"
C="python3 \$HOME/intarsiaml-vprof/out/burst.py $PORT"; CORP=\$HOME/intarsiaml-vprof/out/corpus_docs.txt
{ rsh "$C 2 32 warm $CORP"; sleep 4
  rsh "$C 16 256 a $CORP"; sleep 4
  rsh "DISTINCT=1 $C 16 256 b $CORP"; sleep 4
  rsh "$C 1 128 c $CORP"; sleep 3
} 2>&1 | tee "$OUT/bursts.txt"
rsh "curl -s http://127.0.0.1:$PORT/metrics | grep -E '^vllm:(time_per_output_token|inter_token_latency|iteration_tokens|request_decode_time|e2e_request_latency|num_preemptions|generation_tokens|prompt_tokens)' | grep -v bucket" > "$OUT/metrics.txt"
echo "== stop vLLM (SIGINT -> nsys writes the report) $(date +%T)"
rsh "docker kill -s INT $NAME >/dev/null"
for i in $(seq 1 120); do rsh "docker ps -q -f name=$NAME" | grep -q . || break; sleep 5; done   # up to 10 min for the report to be written
rsh "docker ps -q -f name=$NAME" | grep -q . && { echo "container still running after 10 min: not removing"; exit 1; }
rsh "docker logs $NAME 2>&1 | tail -30" > "$OUT/vllm-shutdown.log"
rsh "docker run --rm --entrypoint chown -v \$HOME/intarsiaml-vprof/out:/out $IMG -R \$(id -u):\$(id -g) /out; docker rm -f $NAME >/dev/null 2>&1; ls -la \$HOME/intarsiaml-vprof/out"
echo "== bring the standing service back $(date +%T)"
restore
echo "== nsys stats $(date +%T) (box is back up; CPU only)"
rsh "export PATH=/usr/local/bin:\$PATH; cd \$HOME/intarsiaml-vprof/out && nsys stats -r cuda_gpu_trace -f csv --force-export=true -o trace nsys.nsys-rep < /dev/null > stats.log 2>&1; nsys stats -r cuda_api_trace -f csv -o api nsys.nsys-rep < /dev/null > stats-api.log 2>&1; grep -E \"Name|cudaGraphLaunch\" api_cuda_api_trace.csv | gzip > graphlaunch.csv.gz; ls -la; python3 tail.py trace_cuda_gpu_trace.csv 4 tail4.csv.gz" 2>&1 | tee "$OUT/stats-tail.log"
rsh "cd \$HOME/intarsiaml-vprof/out && ls -la" >> "$OUT/stats-tail.log"
echo "== done $(date +%T)"

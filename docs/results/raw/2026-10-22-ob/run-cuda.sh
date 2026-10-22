#!/usr/bin/env bash
# ob lane, CUDA leg. Run from the repo root on endor inside a fleetq slot on spark1 (job number in the FLEETQ_JOB env var):
#   run-cuda.sh <stage>       stages: deploy | start | tui | check | stop
# vLLM is never touched; the server uses ~20 GB (Qwen3-8B bf16) on its own port.
set -euo pipefail
: "${FLEETQ_JOB:?run inside a fleetq slot}"
HOST=spark1; HOSTIP=192.168.1.95
R=docs/results/raw/2026-10-22-ob/cuda; mkdir -p $R
M8=/home/nvidia/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
PORT=${PORT:-18612}
case "${1:?stage}" in
deploy)
  scripts/deploy.sh $HOST --backend cuda --ref HEAD --smoke none 2>&1 | tee $R/deploy.log ;;
start)
  ssh -o BatchMode=yes $HOST bash -s 2>&1 <<REMOTE | tee $R/start.log
set -e
T=\$(ls -dt ~/intarsia-deploy/*/ | head -1); BIN=\$(ls \$T/target/release/intarsia)
echo "tree \$T bin \$BIN"
ss -ltn | grep -q ":$PORT " && { echo "port $PORT busy"; exit 3; }
setsid nohup \$BIN serve --backend cuda --model $M8 --host 0.0.0.0 --port $PORT --served-model-name Qwen3-8B \
  --max-seq 8192 --profile-every 16 --mem-budget 24 --no-ui > ~/ob-serve.log 2>&1 < /dev/null &
echo \$! > ~/ob-serve.pid
for i in \$(seq 1 120); do curl -sf localhost:$PORT/v1/models >/dev/null && { echo up after \${i}x2s; exit 0; }; sleep 2; done
echo "server did not come up"; tail -20 ~/ob-serve.log; exit 1
REMOTE
  ;;
check)
  U=http://$HOSTIP:$PORT
  sed "s|__TARGET__|$HOSTIP:$PORT|;s|__HOST__|spark1-cuda|" docs/results/raw/2026-10-22-ob/prom.yml > $R/prom.yml
  docker rm -f ob-prom >/dev/null 2>&1 || true
  docker run -d --name ob-prom -p 19099:9090 -v "$PWD/$R/prom.yml:/etc/prometheus/prometheus.yml:ro" \
    -v "$PWD/deploy/observability/prometheus/alerts.yml:/etc/prometheus/alerts.yml:ro" prom/prometheus:v3.5.0 \
    --config.file=/etc/prometheus/prometheus.yml >/dev/null
  trap 'docker rm -f ob-prom >/dev/null 2>&1 || true' EXIT
  sleep 8
  INTARSIA_URL=$U python3 docs/results/raw/2026-10-22-ob/load.py 4 | tee $R/load.log
  (cd /opt/code/deepeval && uv run localeval speed --url $U/v1 --model Qwen3-8B --sizes 128,1024,4096 --gen 64 --reps 1 \
     --concurrency 2 --label ob-cuda-dash 2>&1) | tee $R/localeval-speed.log
  curl -s $U/metrics > $R/metrics.txt
  sleep 15
  python3 docs/results/raw/2026-10-22-ob/qcheck.py deploy/observability/grafana/dashboards/intarsia.json http://localhost:19099 | tee $R/qcheck.txt
  curl -s http://localhost:19099/api/v1/rules > $R/rules.json
  curl -s http://localhost:19099/api/v1/targets > $R/targets.json ;;
tui)
  ssh -o BatchMode=yes $HOST bash -s 2>&1 <<REMOTE | tee $R/tui-snapshot.txt
set -e
T=\$(ls -dt ~/intarsia-deploy/*/ | head -1)
\$T/target/release/intarsia tui --url http://127.0.0.1:$PORT --snapshot --chat "Explain KV caches in two sentences." --size 150x52 --hold 3
REMOTE
  ;;
stop)
  ssh -o BatchMode=yes $HOST 'kill $(cat ~/ob-serve.pid) 2>&1; sleep 3; ss -ltn | grep :'$PORT' || echo stopped' < /dev/null | tee $R/stop.log ;;
esac

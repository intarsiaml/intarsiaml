#!/usr/bin/env bash
# Lane H27P, ON tatooine INSIDE a GPU window (win.sh): the context `serve --max-batch 16` admits with the base build (chunk 256) and this branch
# (chunk 2048 + peel 16), --mem-budget 29 and a --max-seq far above what fits, so the loader's own clamp is printed ("context N tokens").
# Also the plan's prefill workspace if the log prints it. Containers are named h27p-ctx and removed.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
O=${O:-$HOME/intarsiaml-h27p-logs/ctx}; mkdir -p "$O"
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
for arm in base dev; do for mb in 16 1; do
  bin=h27p-intarsia; [ $arm = base ] && bin=base-intarsia
  docker rm -f h27p-ctx >/dev/null 2>&1
  docker run -d --name h27p-ctx --network host --device /dev/kfd --device /dev/dri \
    --group-add $(stat -c %g /dev/kfd) --group-add $(stat -c %g /dev/dri/renderD128) --group-add video --ipc host \
    --security-opt seccomp=unconfined --user $(id -u):$(id -g) -e HOME=/tmp -e GPU_MAX_HW_QUEUES=1 -e HIP_VISIBLE_DEVICES=0,1 \
    -v $R:/work -v /data/models:/models:ro -w /work $IMG /work/$bin serve --backend hip --model /models/radiance/hf-cache/hub/models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/5233554c5fa56afda40150556b95573c2d7d29c0 \
    --tp 2 --devices 0,1 --max-batch $mb --profile-every 0 --mem-budget 29 --max-seq 262144 --no-supervisor --host 127.0.0.1 --port 8097 > /dev/null 2>&1
  for i in $(seq 1 60); do docker logs h27p-ctx 2>&1 | grep -aq "context [0-9]* tokens" && break; sleep 3; done
  echo "ctx $arm max-batch $mb: $(docker logs h27p-ctx 2>&1 | grep -aoE 'prefill chunk [0-9]+|context [0-9]+ tokens' | tr '\n' ' ')"
  docker logs h27p-ctx 2>&1 | grep -aiE "workspace|scratch|activation|kv pool|kv cache|budget" | head -5 > "$O/log-$arm-$mb.txt"
  docker rm -f h27p-ctx >/dev/null 2>&1
done; done

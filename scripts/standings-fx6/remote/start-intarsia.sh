#!/usr/bin/env bash
. "$(dirname "$0")/common.sh"
REPO=/home/josef/intarsiaml-fx6; VOL=intarsia-cargo-target-fx6
[ "${1:-}" = old ] && { REPO=/home/josef/intarsiaml-standings; VOL=intarsia-cargo-target-standings; }
docker run -d --name fx6-intarsia --network host $DEV --user "$(id -u):$(id -g)" -e HOME=/tmp -e GPU_MAX_HW_QUEUES=1 -e HIP_VISIBLE_DEVICES=0 \
  -v $REPO:/work -v /data/models/intarsia:/models:ro -v $VOL:/target -w /work $IMG \
  /target/release/intarsia serve --backend hip --model /models/Qwen3.6-35B-A3B-NVFP4 --profile-every 0 --mem-budget 29 \
  --host 0.0.0.0 --port 8090 --served-model-name qwen36-35b-intarsia-hip --max-seq 17408

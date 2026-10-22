#!/usr/bin/env bash
# hipfire ad10b3d, MQ4R, recipe from /opt/code/amd_r9700_pro_tuning/overnight-2026-09-26/approved/serve-bench.sh (sequential mode);
# config: reasoning.mode=off, speculation.mode=off. Bound to all interfaces for the few minutes of the exclusive window (localeval must reach it by host name).
W=/home/josef/bench-r9700/approved-2026-09-26
docker run -d --name fx6-hipfire --device /dev/kfd --device /dev/dri --group-add 993 --group-add 44 --security-opt seccomp=unconfined --network host \
  -v $W/hipfire-home:/root/.hipfire -v /data/models/hipfire:/root/.hipfire/models:ro -v $W/kcache:/var/cache/hipfire -e HIP_VISIBLE_DEVICES=0 \
  hipfire:ad10b3d serve --model qwen3.6-35b-a3b.mq4r 0.0.0.0:11435

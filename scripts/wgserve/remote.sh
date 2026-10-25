#!/usr/bin/env bash
# Lane wg, ON tatooine (copy: ~/wg-serve.sh): the served-35B comparison window helper.
#   remote.sh build        cargo build --release --features hip -p intarsia-cli in the wg tree/volume (cpu job, radiance up); stamps "serve"
#   remote.sh build-base   the same for ~/intarsiaml-wg-base (origin/main, `sync.sh base`) in volume wgbase  (arm "main")
#   remote.sh start ARM    ARM = off | opt1 | main : docker run -d wg-serve on :8090 (35B NVFP4, hip, --max-batch 16)
#   remote.sh stop
#   remote.sh hold [SECS]  run inside wg-win.sh (radiance down): wait until ~/wg-logs/served.done exists or SECS (default 1500), then stop the server
#   remote.sh done         touch the done file (ends hold)
# off = branch binary, no knob (identical to main by construction); opt1 = branch binary + INTARSIA_HIP_WMMA_ROWS=1 INTARSIA_HIP_WMMA_TABLE=fp8a8=all,bf16=all.
set -uo pipefail
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
DEVF="--device /dev/kfd --device /dev/dri --group-add $(stat -c %g /dev/kfd) --group-add $(stat -c %g /dev/dri/renderD128) --group-add video --ipc host --security-opt seccomp=unconfined"
LOG=$HOME/wg-logs; mkdir -p "$LOG"; DONE=$LOG/served.done
case ${1:-} in
  build)
    INTARSIA_REPO=$HOME/intarsiaml-wg INTARSIA_TARGET_VOL=intarsia-cargo-target-wg $HOME/intarsiaml-wg/deploy/tatooine/dev.sh \
      cargo build --release --locked --features hip -p intarsia-cli 2>&1 | tail -4 && bash "$HOME/wg-win.sh" --stamp serve ;;
  build-base)
    INTARSIA_REPO=$HOME/intarsiaml-wg-base INTARSIA_TARGET_VOL=intarsia-cargo-target-wgbase $HOME/intarsiaml-wg/deploy/tatooine/dev.sh \
      cargo build --release --locked --features hip -p intarsia-cli 2>&1 | tail -4 ;;
  stop) docker rm -f wg-serve >/dev/null 2>&1; true ;;
  start)
    arm=${2:?arm}; repo=$HOME/intarsiaml-wg; vol=intarsia-cargo-target-wg; envs=()
    case $arm in
      off) ;;
      opt1) envs=(-e INTARSIA_HIP_WMMA_ROWS=1 -e INTARSIA_HIP_WMMA_TABLE=fp8a8=all,bf16=all) ;;
      main) repo=$HOME/intarsiaml-wg-base; vol=intarsia-cargo-target-wgbase ;;
      *) echo "unknown arm $arm"; exit 2 ;;
    esac
    docker rm -f wg-serve >/dev/null 2>&1
    docker run -d --name wg-serve --network host $DEVF --user "$(id -u):$(id -g)" -e HOME=/tmp -e GPU_MAX_HW_QUEUES=1 -e HIP_VISIBLE_DEVICES=0 ${envs[@]+"${envs[@]}"} \
      -v "$repo":/work -v /data/models/intarsia:/models:ro -v "$vol":/target -w /work "$IMG" \
      /target/release/intarsia serve --backend hip --model /models/Qwen3.6-35B-A3B-NVFP4 --max-batch 16 --profile-every 0 --mem-budget 29 \
      --host 0.0.0.0 --port 8090 --served-model-name wg-35b --max-seq 8192 ;;
  hold)
    rm -f "$DONE"; end=$(( $(date +%s) + ${2:-1500} ))
    while [ ! -e "$DONE" ] && [ "$(date +%s)" -lt "$end" ]; do sleep 2; done
    echo "[hold] $([ -e "$DONE" ] && echo done-file || echo TIMEOUT) $(date +%T)"; bash "$0" stop ;;
  done) touch "$DONE" ;;
  *) sed -n 2,10p "$0"; exit 2 ;;
esac

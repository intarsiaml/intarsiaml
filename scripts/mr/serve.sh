#!/usr/bin/env bash
# Lane mr, ON tatooine inside the fleetq slot that holds the cards with radiance down: the 27B MXFP4 server for localeval.
#   serve.sh up <dev|base> <max-batch> [extra serve args]     start `intarsia serve` on :8294 (card 0), wait until it answers
#   serve.sh down <label>                                     save the log to ~/mr-logs/<label>.log, remove the container
set -uo pipefail
LOGDIR=$HOME/mr-logs; mkdir -p "$LOGDIR"
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
H=/data/models/radiance/hf-cache/hub
M27=/models/models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/5233554c5fa56afda40150556b95573c2d7d29c0
cmd=${1:?}; shift
case "$cmd" in
  up)
    tree=$1; mb=$2; shift 2
    case "$tree" in dev) vol=intarsia-cargo-target-mr; repo=$HOME/mr-hip ;; base) vol=intarsia-cargo-target-mrbase; repo=$HOME/mr-hip/.base ;; *) exit 2 ;; esac
    docker rm -f intarsia-serve >/dev/null 2>&1
    docker run -d --name intarsia-serve --network host --device /dev/kfd --device /dev/dri \
      --group-add "$(stat -c %g /dev/kfd)" --group-add "$(stat -c %g /dev/dri/renderD128)" --group-add video \
      --ipc host --security-opt seccomp=unconfined --user "$(id -u):$(id -g)" -e HOME=/tmp -e GPU_MAX_HW_QUEUES=1 \
      -e HIP_VISIBLE_DEVICES=0 -e INTARSIA_HOSTNAME=tatooine \
      -v "$repo":/work -v "$H":/models:ro -v "$vol":/target -w /work "$IMG" \
      /target/release/intarsia serve --backend hip --model "$M27" --host 0.0.0.0 --port 8294 --served-model-name intarsia \
      --max-seq ${MAX_SEQ:-8192} --max-batch "$mb" "$@" >/dev/null
    for i in $(seq 1 90); do
      if curl -s -m 3 http://localhost:8294/v1/models | grep -q '"data"'; then echo "[serve] $tree max-batch $mb serving after $((i * 5)) s"; exit 0; fi
      if [ "$(docker inspect -f '{{.State.Running}}' intarsia-serve 2>/dev/null)" != true ]; then echo "[serve] $tree max-batch $mb EXITED:"; docker logs --tail 25 intarsia-serve 2>&1 | cut -c1-300; exit 1; fi
      sleep 5
    done
    echo "[serve] no answer in 450 s"; exit 1 ;;
  down)
    label=$1
    docker logs intarsia-serve > "$LOGDIR/$label.log" 2>&1
    docker rm -f intarsia-serve >/dev/null 2>&1
    echo "[serve] saved $LOGDIR/$label.log" ;;
esac

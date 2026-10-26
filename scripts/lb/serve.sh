#!/usr/bin/env bash
# Lane lb, ON tatooine inside the fleetq slot that holds card 0 with radiance down: the 35B server for localeval.
#   serve.sh up <dev|base> <max-batch>     start `intarsia serve` on :8294 (card 0), wait until it answers
#   serve.sh down <label>                  save the log to ~/intarsiaml-lb/logs/<label>.log, remove the container
# Env passed through to the server: every INTARSIA_HIP_* variable set in the caller's environment.
set -uo pipefail
LOGDIR=$HOME/intarsiaml-lb/logs; mkdir -p "$LOGDIR"
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
cmd=${1:?}; shift
case "$cmd" in
  up)
    tree=$1; mb=$2; shift 2
    case "$tree" in dev|base) vol=intarsia-cargo-lb-$tree; repo=$HOME/intarsiaml-lb/$tree ;; *) exit 2 ;; esac
    envs=(); for v in $(compgen -e | grep -E '^INTARSIA_HIP_' || true); do envs+=(-e "$v=${!v}"); done
    docker rm -f intarsia-lb-serve >/dev/null 2>&1
    docker run -d --name intarsia-lb-serve --network host --device /dev/kfd --device /dev/dri \
      --group-add "$(stat -c %g /dev/kfd)" --group-add "$(stat -c %g /dev/dri/renderD128)" --group-add video \
      --ipc host --security-opt seccomp=unconfined --user "$(id -u):$(id -g)" -e HOME=/tmp -e GPU_MAX_HW_QUEUES=1 \
      -e HIP_VISIBLE_DEVICES=0 -e INTARSIA_HOSTNAME=tatooine "${envs[@]}" \
      -v "$repo":/work -v /data/models/intarsia:/models:ro -v "$vol":/target -w /work "$IMG" \
      /target/release/intarsia serve --backend hip --model /models/Qwen3.6-35B-A3B-NVFP4 --host 0.0.0.0 --port 8294 \
      --served-model-name intarsia --max-seq ${MAX_SEQ:-18432} --max-batch "$mb" "$@" >/dev/null
    for i in $(seq 1 90); do
      if curl -s -m 3 http://localhost:8294/v1/models | grep -q '"data"'; then echo "[serve] $tree max-batch $mb serving after $((i * 5)) s"; exit 0; fi
      if [ "$(docker inspect -f '{{.State.Running}}' intarsia-lb-serve 2>/dev/null)" != true ]; then echo "[serve] $tree max-batch $mb EXITED:"; docker logs --tail 25 intarsia-lb-serve 2>&1 | cut -c1-300; exit 1; fi
      sleep 5
    done
    echo "[serve] no answer in 450 s"; exit 1 ;;
  down)
    label=$1
    docker logs intarsia-lb-serve > "$LOGDIR/$label.log" 2>&1
    docker rm -f intarsia-lb-serve >/dev/null 2>&1
    echo "[serve] saved $LOGDIR/$label.log" ;;
esac

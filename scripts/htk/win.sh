#!/usr/bin/env bash
# Lane htk: serve helper that runs ON tatooine inside a fleetq job that takes radiance down (one 35B server on card 0, port 8296).
#   win.sh up <label> [KEY=VAL ...]   stop radiance (if up), start `intarsia serve` from ~/intarsiaml-htk/src/.bin/intarsia
#   win.sh down <label>               save the server log to ~/intarsiaml-htk/logs/<label>.log and remove the container
#   win.sh restore                    start radiance, wait until it answers (<= 3 min), print its state
# Never leaves radiance down: `restore` is idempotent and is the last step of every window.
set -uo pipefail
R=$HOME/intarsiaml-htk; mkdir -p $R/logs
now() { date '+%F %T'; }
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
cmd=${1:?}; shift
case "$cmd" in
  up)
    label=$1; shift
    docker rm -f intarsia-htk-serve >/dev/null 2>&1
    if systemctl --user is-active --quiet radiance; then echo "[win] stopping radiance $(now)"; systemctl --user stop radiance; sleep 3; fi
    envs=(); for kv in "$@"; do envs+=(-e "$kv"); done
    docker run -d --name intarsia-htk-serve --network host --device /dev/kfd --device /dev/dri \
      --group-add "$(stat -c %g /dev/kfd)" --group-add "$(stat -c %g /dev/dri/renderD128)" --group-add video \
      --ipc host --security-opt seccomp=unconfined --user "$(id -u):$(id -g)" -e HOME=/tmp -e GPU_MAX_HW_QUEUES=1 \
      -e HIP_VISIBLE_DEVICES=0 -e INTARSIA_HOSTNAME=tatooine "${envs[@]}" \
      -v "$R/src":/work -v /data/models/intarsia:/models:ro -w /work "$IMG" \
      /work/.bin/intarsia serve --backend hip --model /models/Qwen3.6-35B-A3B-NVFP4 --host 0.0.0.0 --port 8296 \
      --served-model-name intarsia --max-seq 8192 --max-batch 16 ${SERVE_ARGS:-} >/dev/null
    for i in $(seq 1 60); do
      if curl -s -m 3 http://localhost:8296/v1/models | grep -q '"data"'; then echo "[win] $label serving after $((i * 5)) s $(now)"; exit 0; fi
      if [ "$(docker inspect -f '{{.State.Running}}' intarsia-htk-serve 2>/dev/null)" != true ]; then echo "[win] server exited"; docker logs --tail 30 intarsia-htk-serve; exit 1; fi
      sleep 5
    done
    echo "[win] server did not answer in 300 s"; exit 1
    ;;
  down)
    label=$1
    docker logs intarsia-htk-serve > "$R/logs/$label.log" 2>&1
    docker rm -f intarsia-htk-serve >/dev/null 2>&1
    echo "[win] saved $R/logs/$label.log ($(wc -l < "$R/logs/$label.log") lines)"
    ;;
  restore)
    docker rm -f intarsia-htk-serve >/dev/null 2>&1
    if systemctl --user is-active --quiet radiance; then echo "[win] radiance already active"; fi
    echo "[win] starting radiance $(now)"
    systemctl --user start radiance
    for i in $(seq 1 36); do
      if curl -s -m 3 http://localhost:8080/v1/models | grep -q '"data":\['; then
        echo "[win] radiance answers $(now) (after $((i * 5)) s); $(systemctl --user is-active radiance)"; exit 0
      fi
      sleep 5
    done
    echo "[win] RADIANCE DID NOT ANSWER within 180 s $(now); $(systemctl --user is-active radiance)"; exit 1
    ;;
esac

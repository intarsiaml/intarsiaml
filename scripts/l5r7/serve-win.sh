#!/usr/bin/env bash
# Lane l5r7, ON tatooine inside a fleetq job that takes radiance down (one 35B server on card 1 = HIP_VISIBLE_DEVICES=1, port 8296, container intarsia-serve-l5).
#   serve-win.sh up <dev|base> <label> [KEY=VAL ...]   stop radiance, start `intarsia serve` from the snapshot (NAME) or the merge-base tree
#   serve-win.sh down <label>                          save the server log to ~/intarsiaml-l5-logs/<label>.log, remove the container
#   serve-win.sh restore                               remove the container, start radiance, wait until it answers (<= 3 min)
# `restore` is idempotent and the last step of every window.
set -uo pipefail
LOGDIR=$HOME/intarsiaml-l5-logs; mkdir -p "$LOGDIR"
now() { date '+%F %T'; }
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
CT=intarsia-serve-l5
cmd=${1:?}; shift
case "$cmd" in
  up)
    who=$1; label=$2; shift 2
    case "$who" in
      dev) tree=$HOME/intarsiaml-l5-${NAME:?}; vol=intarsia-cargo-target-l5-zdev ;;
      base) tree=$HOME/intarsiaml-l5-zbase; vol=intarsia-cargo-target-l5-zbase ;;
      *) echo "unknown $who"; exit 2 ;;
    esac
    docker rm -f $CT >/dev/null 2>&1
    if systemctl --user is-active radiance >/dev/null 2>&1; then echo "[serve] stopping radiance $(now)"; systemctl --user stop radiance; sleep 3; fi
    for i in $(seq 1 30); do
      u=$(cat /sys/bus/pci/devices/0000:07:00.0/mem_info_vram_used 2>/dev/null || echo -1)
      echo "[vram] card1 used=$u B ($((u / 1048576)) MiB) $(now)"
      if [ "$u" -ge 0 ] && [ "$u" -lt 1000000000 ]; then break; fi
      sleep 2
    done
    envs=(); for kv in "$@"; do envs+=(-e "$kv"); done
    docker run -d --name $CT --network host --device /dev/kfd --device /dev/dri \
      --group-add "$(stat -c %g /dev/kfd)" --group-add "$(stat -c %g /dev/dri/renderD128)" --group-add video \
      --ipc host --security-opt seccomp=unconfined --user "$(id -u):$(id -g)" -e HOME=/tmp -e GPU_MAX_HW_QUEUES=1 \
      -e HIP_VISIBLE_DEVICES=1 -e INTARSIA_HOSTNAME=tatooine "${envs[@]}" \
      -v "$tree":/work -v /data/models/intarsia:/models:ro -v "$vol":/target -w /work "$IMG" \
      /target/release/intarsia serve --backend hip --model /models/Qwen3.6-35B-A3B-NVFP4 --host 0.0.0.0 --port 8296 \
      --served-model-name intarsia --max-seq 20480 --max-batch 1 --mem-budget 29 ${SERVE_ARGS:-} >/dev/null
    for i in $(seq 1 60); do
      if curl -s -m 3 http://localhost:8296/v1/models | grep -q '"data"'; then echo "[serve] $label ($who) serving after $((i * 5)) s $(now)"; exit 0; fi
      if [ "$(docker inspect -f '{{.State.Running}}' $CT 2>/dev/null)" != true ]; then echo "[serve] server exited"; docker logs --tail 30 $CT; exit 1; fi
      sleep 5
    done
    echo "[serve] server did not answer in 300 s"; exit 1
    ;;
  down)
    label=$1
    docker logs $CT > "$LOGDIR/$label.log" 2>&1
    docker rm -f $CT >/dev/null 2>&1
    echo "[serve] saved $LOGDIR/$label.log ($(wc -l < "$LOGDIR/$label.log") lines)"
    ;;
  restore)
    docker rm -f $CT >/dev/null 2>&1
    echo "[serve] starting radiance $(now)"
    systemctl --user start radiance
    for i in $(seq 1 36); do
      if curl -s -m 3 http://localhost:8080/v1/models | grep -q '"data":\['; then
        echo "[serve] radiance answers $(now) (after $((i * 5)) s); $(systemctl --user is-active radiance)"; exit 0
      fi
      sleep 5
    done
    echo "[serve] RADIANCE DID NOT ANSWER within 180 s $(now); $(systemctl --user is-active radiance)"; exit 1
    ;;
esac

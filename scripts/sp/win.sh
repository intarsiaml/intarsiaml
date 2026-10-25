#!/usr/bin/env bash
# Lane sp: serve helper that runs ON tatooine inside a fleetq job that takes radiance down (one 35B server on card 0).
#   win.sh up <tree> <label> [KEY=VAL ...]   stop radiance, start `intarsia serve` from <tree> (~/intarsiaml-<tree>) on :8294
#   win.sh snap <outfile>                    per-thread CPU (utime stime, ticks) of the server, plus /metrics
#   win.sh down <label>                      save the server log to ~/sp-logs/<label>.log and remove the container
#   win.sh restore                           start radiance, wait until it answers (<= 3 min), print its state
# Never leaves radiance down: `restore` is idempotent and is the last step of every window.
set -uo pipefail
LOGDIR=$HOME/sp-logs
mkdir -p "$LOGDIR"
now() { date '+%F %T'; }
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
cmd=${1:?}; shift
case "$cmd" in
  up)
    tree=$1; label=$2; shift 2
    case "$tree" in sp) vol=intarsia-cargo-target-sp ;; sp-base) vol=intarsia-cargo-target-spbase ;; *) echo "unknown tree $tree"; exit 2 ;; esac
    docker rm -f intarsia-serve >/dev/null 2>&1
    echo "[win] stopping radiance $(now)"
    systemctl --user stop radiance
    sleep 3
    envs=(); for kv in "$@"; do envs+=(-e "$kv"); done
    docker run -d --name intarsia-serve --network host --device /dev/kfd --device /dev/dri \
      --group-add "$(stat -c %g /dev/kfd)" --group-add "$(stat -c %g /dev/dri/renderD128)" --group-add video \
      --ipc host --security-opt seccomp=unconfined --user "$(id -u):$(id -g)" -e HOME=/tmp -e GPU_MAX_HW_QUEUES=1 \
      -e HIP_VISIBLE_DEVICES=0 -e INTARSIA_HOSTNAME=tatooine "${envs[@]}" \
      -v "$HOME/intarsiaml-$tree":/work -v /data/models/intarsia:/models:ro -v "$vol":/target -w /work "$IMG" \
      /target/release/intarsia serve --backend hip --model ${MODEL:-/models/Qwen3.6-35B-A3B-NVFP4} --host 0.0.0.0 --port 8294 \
      --served-model-name intarsia --max-seq ${MAX_SEQ:-20480} --max-batch ${MAX_BATCH:-1} ${SERVE_ARGS:-} >/dev/null
    for i in $(seq 1 60); do
      if curl -s -m 3 http://localhost:8294/v1/models | grep -q '"data"'; then echo "[win] $label serving after $((i * 5)) s $(now)"; exit 0; fi
      if [ "$(docker inspect -f '{{.State.Running}}' intarsia-serve 2>/dev/null)" != true ]; then echo "[win] server exited"; docker logs --tail 30 intarsia-serve; exit 1; fi
      sleep 5
    done
    echo "[win] server did not answer in 300 s"; exit 1
    ;;
  snap)
    out=$1
    pid=$(docker inspect -f '{{.State.Pid}}' intarsia-serve)
    {
      echo "## $(now) pid $pid clk_tck $(getconf CLK_TCK)"
      for t in /proc/"$pid"/task/*; do
        # comm, utime(14) stime(15)
        awk -v c="$(cat "$t/comm")" '{print $1, c, $14, $15}' "$t/stat" 2>/dev/null
      done
      echo "## metrics"
      curl -s -m 5 http://localhost:8294/metrics
    } >> "$out"
    ;;
  down)
    label=$1
    docker logs intarsia-serve > "$LOGDIR/$label.log" 2>&1
    docker rm -f intarsia-serve >/dev/null 2>&1
    echo "[win] saved $LOGDIR/$label.log ($(wc -l < "$LOGDIR/$label.log") lines)"
    ;;
  restore)
    docker rm -f intarsia-serve >/dev/null 2>&1
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

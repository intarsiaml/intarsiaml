#!/usr/bin/env bash
# Lane T2 bench window (ON tatooine, radiance down): 27B decode / prefill speed, `intarsia bench speed --at-most`, 1k / 4k / 16k,
# gen 128, 1 rep + 1 warmup per arm and round, each arm a fresh `serve`. Arms (labelled by env, not "default"):
#   tp1    one R9700, graphs + fusion (P1's shipped config)          --require-graphs
#   tp2g   both R9700s, --tp 2, graphs (no INTARSIA_TP_FUSION)       --require-graphs
#   tp2gf  both R9700s, --tp 2, graphs + INTARSIA_TP_FUSION=1        --require-graphs
# ROUNDS (default "1 2"); odd rounds in that order, even rounds reversed.
set -uo pipefail
export O=${1:?out dir}
source "$HOME/t2/lib.sh"
PORT=8091; NAME=qwen38-27b-t2; R=$(T head)
start() { # start <arm>
  local env=() gpus=0 extra=""
  case $1 in
    tp1) ;;
    tp2g) gpus=0,1; extra="$TP2" ;;
    tp2gf) gpus=0,1; extra="$TP2"; env=(-e INTARSIA_TP_FUSION=1) ;;
  esac
  mkdir -p "$R/t2-out/bench"
  docker rm -f t2-serve >/dev/null 2>&1
  # shellcheck disable=SC2086
  docker run -d --name t2-serve --network host --device /dev/kfd --device /dev/dri \
    --group-add "$(stat -c %g /dev/kfd)" --group-add "$(stat -c %g /dev/dri/renderD128)" --group-add video \
    --ipc host --security-opt seccomp=unconfined --user "$(id -u):$(id -g)" -e HOME=/tmp -e GPU_MAX_HW_QUEUES=1 \
    -e HIP_VISIBLE_DEVICES=$gpus "${env[@]}" -v "$R":/work -v "$HUB":/models:ro -w /work intarsia-hip-dev:rocm7.2.4-rust1.90.0 \
    /work/t2-intarsia serve --backend hip --model $M27 $extra --profile-every 0 --mem-budget 29 --host 0.0.0.0 --port $PORT \
    --served-model-name $NAME --max-seq 20480 --no-ui < /dev/null > /dev/null
  local i
  for i in $(seq 1 100); do
    curl -sf -m 2 http://localhost:$PORT/health >/dev/null 2>&1 < /dev/null && return 0
    docker ps -q -f name=t2-serve | grep -q . || break
    sleep 3
  done
  docker logs --tail 30 t2-serve > "$O/serve-$1-fail.log" 2>&1; return 1
}
bench() { # bench <arm> <round>
  local l=t2-$1-r$2
  start "$1" || { log "$l: serve did not come up"; return 1; }
  docker exec t2-serve /work/t2-intarsia bench speed --url http://localhost:$PORT --model $NAME --sizes ${SIZES:-1024,4096,16384} \
    --gen 128 --reps 1 --warmup 1 --at-most --thinking off --require-graphs --label $l --out-dir /work/t2-out/bench \
    > "$O/bench-$l.log" 2>&1 < /dev/null
  log "$l exit $?: $(grep -aE '^ *(1024|4096|16384) |require-graphs' "$O/bench-$l.log" | tr -s ' ' | tr '\n' '|')"
  curl -s -m 3 http://localhost:$PORT/metrics < /dev/null | grep -E '^intarsia_graph_(replays|captures)_total' > "$O/metrics-$l.txt"
  docker logs t2-serve > "$O/serve-$l.log" 2>&1
  docker rm -f t2-serve >/dev/null 2>&1
  cp "$R/t2-out/bench/$l.json" "$O/" 2>/dev/null
}
trap 'docker rm -f t2-serve >/dev/null 2>&1' EXIT
for r in ${ROUNDS:-1 2}; do
  arms="tp1 tp2g tp2gf"; (( r % 2 == 0 )) && arms="tp2gf tp2g tp1"
  for a in $arms; do bench $a $r; done
done
log done

#!/usr/bin/env bash
# Lane G2b window 3 (ON tatooine, radiance down, called by window.sh): 35B speed, dev tree.
#   Arms, interleaved x3 (round r: tp1, tp1-eager-unfused, tp2), `intarsia bench speed --at-most` 1k/4k/16k, gen 128:
#     tp1       one R9700, default (HIP graphs + fusions)
#     tp1eu     one R9700, --no-graphs and INTARSIA_NO_FUSION=1: what tp=2 runs per rank (core turns graphs and fusion
#               off at tp > 1), the like-for-like baseline
#     tp2       both R9700s, --tp 2 --devices 0,1 (HIP P2P all-reduce)
#   Then the tp2 server stays up for localeval from the Mac (localeval-sidecar.sh): READY file, waits for DONE (<= 600 s).
set -uo pipefail
O=${1:?out dir}
R=$HOME/intarsiaml-g2b-${TREE:-dev}
log() { echo "$(date +%T) $*" | tee -a "$O/steps.log"; }
PORT=8090; NAME=qwen36-35b-g2b
mkdir -p $R/g2b-out/bench
start() { # start <arm>
  local env=() gpus=0 extra=""
  case $1 in
    tp1) ;;
    tp1eu) env=(-e INTARSIA_NO_FUSION=1); extra="--no-graphs" ;;
    tp2) gpus=0,1; extra="--tp 2 --devices 0,1" ;;
  esac
  docker rm -f g2b-serve >/dev/null 2>&1
  docker run -d --name g2b-serve --network host --device /dev/kfd --device /dev/dri \
    --group-add "$(stat -c %g /dev/kfd)" --group-add "$(stat -c %g /dev/dri/renderD128)" --group-add video \
    --ipc host --security-opt seccomp=unconfined --user "$(id -u):$(id -g)" -e HOME=/tmp -e GPU_MAX_HW_QUEUES=1 \
    -e HIP_VISIBLE_DEVICES=$gpus "${env[@]}" -v $R:/work -v /data/models/intarsia:/models:ro -w /work \
    intarsia-hip-dev:rocm7.2.4-rust1.90.0 /work/g2b-intarsia serve --backend hip --model /models/Qwen3.6-35B-A3B-NVFP4 \
    --profile-every 0 --mem-budget 29 --host 0.0.0.0 --port $PORT --served-model-name $NAME --max-seq 17408 $extra > /dev/null
  for i in $(seq 1 80); do curl -sf -m 2 http://localhost:$PORT/health >/dev/null 2>&1 && return 0; sleep 3; done
  docker logs --tail 30 g2b-serve > "$O/serve-$1-fail.log" 2>&1; return 1
}
bench() { # bench <arm> <round>
  local l=g2b-$1-r$2
  start $1 || { log "$l: server did not come up"; return 1; }
  docker exec g2b-serve /work/g2b-intarsia bench speed --url http://localhost:$PORT --model $NAME --sizes 1024,4096,16384 \
    --gen 128 --reps 1 --warmup 1 --at-most --thinking off --label $l --out-dir /work/g2b-out/bench > "$O/bench-$l.log" 2>&1
  log "$l exit $?: $(grep -aE '^ *(1024|4096|16384) ' "$O/bench-$l.log" | tr -s ' ' | tr '\n' '|')"
  docker logs g2b-serve > "$O/serve-$l.log" 2>&1
  docker rm -f g2b-serve >/dev/null 2>&1
}
# tp=1 fusion-off reference for the short-set KL (accuracy dumps + 16k ids), before the speed arms
bash $HOME/g2b/stages-nf.sh "$O"
for r in 1 2 3; do for a in tp1 tp1eu tp2; do bench $a $r; done; done
cp $R/g2b-out/bench/g2b-*.json "$O/" 2>/dev/null
# localeval on tp2 (run from the Mac by localeval-sidecar.sh)
if start tp2; then
  log "tp2 server up for localeval"; touch "$O/READY"
  for i in $(seq 1 200); do [[ -f "$O/DONE" ]] && break; sleep 3; done
  log "localeval sidecar $( [[ -f $O/DONE ]] && echo done || echo 'timed out')"
  docker logs g2b-serve > "$O/serve-localeval.log" 2>&1
fi
docker rm -f g2b-serve >/dev/null 2>&1
log "done"

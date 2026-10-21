#!/usr/bin/env bash
# Lane H1 window 2 (ON tatooine, radiance down, called by window.sh): 35B decode speed, interleaved A/B x3.
#   `intarsia bench speed --at-most` 1k / 4k / 16k, gen 128, 1 rep + 1 warmup per round; each arm a fresh `serve`.
#   tp1    base, one R9700, default (graphs + fusion)                        --require-graphs
#   tp2e   base, both R9700s, --tp 2 (eager decode, unfused: main today)
#   tp2g   dev, both R9700s, --tp 2 (decode graphs, the new default)         --require-graphs
#   tp2gf  dev, both R9700s, --tp 2, INTARSIA_TP_FUSION=1 (graphs + fusion)  --require-graphs
#   Round order: r1 and r3 as listed, r2 reversed.
set -uo pipefail
export O=${1:?out dir}
source "$HOME/h1/lib.sh"
PORT=8091; NAME=qwen36-35b-h1
start() { # start <arm>
  local env=() gpus=0 extra="" tree=base
  case $1 in
    tp1) ;;
    tp2e) gpus=0,1; extra="$TP2" ;;
    tp2g) tree=dev; gpus=0,1; extra="$TP2" ;;
    tp2gf) tree=dev; gpus=0,1; extra="$TP2"; env=(-e INTARSIA_TP_FUSION=1) ;;
  esac
  local R; R=$(T $tree); mkdir -p "$R/h1-out/bench"
  docker rm -f h1-serve >/dev/null 2>&1
  docker run -d --name h1-serve --network host --device /dev/kfd --device /dev/dri \
    --group-add "$(stat -c %g /dev/kfd)" --group-add "$(stat -c %g /dev/dri/renderD128)" --group-add video \
    --ipc host --security-opt seccomp=unconfined --user "$(id -u):$(id -g)" -e HOME=/tmp -e GPU_MAX_HW_QUEUES=1 \
    -e HIP_VISIBLE_DEVICES=$gpus "${env[@]}" -v "$R":/work -v /data/models/intarsia:/models:ro -w /work \
    intarsia-hip-dev:rocm7.2.4-rust1.90.0 /work/h1-intarsia serve --backend hip --model $M35 \
    --profile-every 0 --mem-budget 29 --host 0.0.0.0 --port $PORT --served-model-name $NAME --max-seq 17408 $extra > /dev/null < /dev/null
  for i in $(seq 1 80); do curl -sf -m 2 http://localhost:$PORT/health >/dev/null 2>&1 < /dev/null && return 0; sleep 3; done
  docker logs --tail 30 h1-serve > "$O/serve-$1-fail.log" 2>&1; return 1
}
bench() { # bench <arm> <round>
  local l=h1-$1-r$2 req=""; [[ $1 != tp2e ]] && req="--require-graphs"
  start $1 || { log "$l: server did not come up"; return 1; }
  local tree=base; [[ $1 == tp2g* ]] && tree=dev
  docker exec h1-serve /work/h1-intarsia bench speed --url http://localhost:$PORT --model $NAME --sizes 1024,4096,16384 \
    --gen 128 --reps 1 --warmup 1 --at-most --thinking off $req --label $l --out-dir /work/h1-out/bench > "$O/bench-$l.log" 2>&1 < /dev/null
  log "$l exit $?: $(grep -aE '^ *(1024|4096|16384) |require-graphs' "$O/bench-$l.log" | tr -s ' ' | tr '\n' '|')"
  docker logs h1-serve > "$O/serve-$l.log" 2>&1
  docker rm -f h1-serve >/dev/null 2>&1
  cp "$(T $tree)/h1-out/bench/$l.json" "$O/" 2>/dev/null
}
for r in 1 2 3; do
  arms="tp1 tp2e tp2g tp2gf"; [[ $r == 2 ]] && arms="tp2gf tp2g tp2e tp1"
  for a in $arms; do bench $a $r; done
done
cp "$(T base)"/h1-out/bench/h1-*.json "$(T dev)"/h1-out/bench/h1-*.json "$O/" 2>/dev/null
# tp=1 calibration, short set (window 1: chunk 128 and GEMV 4,8 are bitwise on HIP, SIMT attention moves decode only):
# three knobs that may reorder prefill arithmetic, for the size of an equally correct reordering at tp=1
for kv in INTARSIA_HIP_GEMM_TILE=1 INTARSIA_HIP_MOE_WMMA=0 INTARSIA_HIP_FP8_F8=0; do
  E="$kv" acc base 0 cal-${kv#INTARSIA_HIP_}-short $M35 $SG
done
log "done"

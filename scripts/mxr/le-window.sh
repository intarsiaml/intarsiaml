#!/usr/bin/env bash
# Lane mxr, LOCAL (Mac) side of window w5, run INSIDE one fleetq slot that takes radiance down on tatooine (FLEETQ_JOB set):
#   fleetq run --host tatooine --takes-down radiance --quiet gpu,cpu --expect 150 --label "..." -- scripts/mxr/le-window.sh <rounds>
# Arms base / dev x tp1 / tp2, interleaved (round 2 reversed): a fresh `intarsia serve --max-batch 16` per arm on tatooine:8294
# (tp1 card 0, tp2 both cards), then localeval speed (1k prompt, gen 512, 1 stream + 4 / 8 / 16 streams, thinking off).
# Results: /opt/code/deepeval/results/<run_id>, names in $OUT/index.txt; server logs ~/intarsiaml-mxr-logs/w5 on tatooine.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
ROUNDS=${1:-3}
OUT=/opt/code/intarsiaml/.worktrees/mxr/.mxrlogs/w5; mkdir -p "$OUT"
URL=http://tatooine:8294/v1
rsh() { ssh -n -o BatchMode=yes tatooine "$@"; }
rsh 'systemctl --user stop radiance; sleep 3; echo "[mxr] radiance $(systemctl --user is-active radiance)"'
restore() {
  rsh 'docker rm -f intarsia-mxr-serve >/dev/null 2>&1; systemctl --user start radiance
       for i in $(seq 1 48); do curl -s -m 3 http://localhost:8080/v1/models | grep -q "\"data\":\[" && break; sleep 5; done
       echo "[mxr] radiance $(systemctl --user is-active radiance) after $((i * 5)) s: $(curl -s -m 3 http://localhost:8080/v1/models | head -c 90)"'
}
trap restore EXIT
up() { # up <tree> <tp>
  local tree=$1 tp=$2 vol repo gpus
  if [ "$tree" = dev ]; then vol=intarsia-mxr-target-dev; repo='$HOME/intarsiaml-mxr-dev'; else vol=intarsia-mxr-target-base; repo='$HOME/intarsiaml-mxr-dev/.base'; fi
  if [ "$tp" = 2 ]; then gpus=0,1; else gpus=0; fi
  rsh "docker rm -f intarsia-mxr-serve >/dev/null 2>&1; mkdir -p ~/intarsiaml-mxr-logs/w5
    docker run -d --name intarsia-mxr-serve --network host --device /dev/kfd --device /dev/dri \
      --group-add \$(stat -c %g /dev/kfd) --group-add \$(stat -c %g /dev/dri/renderD128) --group-add video \
      --ipc host --security-opt seccomp=unconfined --user \$(id -u):\$(id -g) -e HOME=/tmp -e GPU_MAX_HW_QUEUES=1 \
      -e HIP_VISIBLE_DEVICES=$gpus -e INTARSIA_HOSTNAME=tatooine \
      -v $repo:/work -v /data/models/radiance/hf-cache/hub:/models:ro -v $vol:/target -w /work intarsia-hip-dev:rocm7.2.4-rust1.90.0 \
      /target/release/intarsia serve --backend hip --tp $tp --model /models/models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/5233554c5fa56afda40150556b95573c2d7d29c0 \
      --host 0.0.0.0 --port 8294 --served-model-name intarsia --max-seq 8192 --max-batch 16 --profile-every 0 >/dev/null"
  for i in $(seq 1 90); do
    curl -s -m 3 "$URL/models" | grep -q '"data"' && { echo "[serve] $tree tp$tp up after $((i * 5)) s"; return 0; }
    [ "$(rsh 'docker inspect -f {{.State.Running}} intarsia-mxr-serve 2>/dev/null')" = true ] || { echo "[serve] $tree tp$tp EXITED"; rsh 'docker logs --tail 20 intarsia-mxr-serve 2>&1' | cut -c1-200; return 1; }
    sleep 5
  done
  echo "[serve] $tree tp$tp no answer"; return 1
}
down() { rsh "docker logs intarsia-mxr-serve > ~/intarsiaml-mxr-logs/w5/$1.log 2>&1; docker rm -f intarsia-mxr-serve >/dev/null 2>&1"; }
arm() { # arm <round> <tree> <tp>
  local name="mxr-$2-tp$3-r$1"
  up "$2" "$3" || { echo "$name: SERVER FAILED" | tee -a "$OUT/index.txt"; down "$name"; return; }
  ( cd /opt/code/deepeval && FLEETQ_JOB=$FLEETQ_JOB uv run localeval speed --url "$URL" --model intarsia --thinking off --sizes 1024 --gen 512 \
      --reps 1 --concurrency 4,8,16 --label "$name" --no-unim \
      --recipe "intarsia serve --backend hip --tp $3 --max-seq 8192 --max-batch 16 (tree $2: lane-mxr/hip-mxfp4-rows $(git -C /opt/code/intarsiaml/.worktrees/mxr rev-parse --short HEAD) or origin/main)" ) > "$OUT/$name.out" 2>&1
  echo "$name: $(grep -aoE 'results/[^ ]+' "$OUT/$name.out" | head -1)" | tee -a "$OUT/index.txt"
  grep -aiE "decode|tok/s|stream|error|missing" "$OUT/$name.out" | tail -10
  down "$name"
}
ARMS=("base 1" "dev 1" "base 2" "dev 2")
for r in $(seq 1 "$ROUNDS"); do
  if (( r % 2 == 0 )); then for ((k = ${#ARMS[@]} - 1; k >= 0; k--)); do arm $r ${ARMS[k]}; done
  else for a in "${ARMS[@]}"; do arm $r $a; done; fi
done
echo "== done $(date +%T)"

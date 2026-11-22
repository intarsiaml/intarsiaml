#!/usr/bin/env bash
# Lane rocm102: prefill + decode speed, ROCm 7.2.4 image vs 10.2.0 image (same tree, same rustc). Driven FROM THE MAC (localeval
# lives here), under one fleetq job that holds both R9700s and takes radiance down (scripts/gdn-chunk/speed-window.sh's shape):
#   fleetq run --host tatooine --takes-down radiance --expect 45 --mem 30 --quiet gpu,cpu \
#     --label "intarsiaml rocm102 speed" -- scripts/rocm102/speed-window.sh
# Arms: the same tree built in each image (own target volume each, scripts/rocm102/build.sh), rounds interleaved (r7 r10 / r10 r7 / ..).
# tp1-* = one card, `serve`; tp2-* = both cards, `serve --tp 2 --devices 0,1`; *-r7 = intarsia-hip-dev:rocm7.2.4-rust1.90.0,
# *-r10 = intarsia-hip-dev:rocm10.2.0-rust1.90.0. Each arm: a fresh server, `localeval speed` (prefill sweep + decode), then
# the server goes away. localeval runs inside this job's slot (FLEETQ_JOB is inherited).
# PHASES: "name:sizes:rounds:arm,arm" separated by spaces. CAP: seconds; arms that no longer fit are skipped.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
PHASES=${PHASES:-"tp1:1024,4096,16384:3:tp1-r7,tp1-r10 tp2:4096:3:tp2-r7,tp2-r10"}
GEN=${GEN:-128}; CAP=${CAP:-2600}; TAGP=${TAGP:-rocm102}
MODEL=${MODEL:-/models/Qwen3.6-35B-A3B-NVFP4}; NAME=${NAME:-qwen36-35b}; MODELS_DIR=${MODELS_DIR:-/data/models/intarsia}
MAXSEQ=${MAXSEQ:-18432}; PORT=8090
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
OUT=${OUT:-$REPO/docs/results/2026-11-21-hip-rocm102/raw/speed-$(date +%H%M)}; mkdir -p "$OUT"
T0=$(date +%s); left() { echo $(( CAP - ($(date +%s) - T0) )); }
say() { echo "$(date +%T) $*" | tee -a "$OUT/steps.log"; }
SSH="ssh -o BatchMode=yes tatooine"
CNAME=rocm102-serve

# guard: stops radiance now, restores it when its stdin closes (we exit or die) and reports when it answers
FIFO=$(mktemp -u "${TMPDIR:-/tmp}/rocm102guard.XXXXXX"); mkfifo "$FIFO"
$SSH 'bash ~/intarsiaml-rocm102/scripts/rocm102/guard.sh' > "$OUT/guard.log" 2>&1 < "$FIFO" &
GUARD=$!
exec 9> "$FIFO"
for i in $(seq 1 30); do grep -q "radiance stopped" "$OUT/guard.log" 2>/dev/null && break; sleep 1; done
say "guard: $(grep -h 'radiance stopped' "$OUT/guard.log")"
drop() { $SSH "docker rm -f $CNAME" >/dev/null 2>&1 < /dev/null; }
cleanup() { drop; exec 9>&-; wait $GUARD; rm -f "$FIFO"; say "guard: $(grep -h 'radiance answers\|DID NOT' "$OUT/guard.log")"; }
trap cleanup EXIT

serve() { # serve <arm>
  local arm=$1 args="" gpus=0
  local IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0 VOL=intarsia-cargo-target-rocm102-r7
  [[ $arm == *-r10 ]] && { IMG=intarsia-hip-dev:rocm10.2.0-rust1.90.0; VOL=intarsia-cargo-target-rocm102-r10; }
  [[ $arm == tp2-* ]] && { gpus=0,1; args="--tp 2 --devices 0,1"; }
  RECIPE="HIP_VISIBLE_DEVICES=$gpus intarsia serve --backend hip --model $MODEL --served-model-name $NAME-$arm $args --profile-every 0 --mem-budget 29 --max-seq $MAXSEQ --no-supervisor --host 0.0.0.0 --port $PORT (tree rocm102, docker $IMG)"
  drop
  $SSH "docker run -d --name $CNAME --network host --device /dev/kfd --device /dev/dri \
    --group-add \$(stat -c %g /dev/kfd) --group-add \$(stat -c %g /dev/dri/renderD128) --group-add video --ipc host \
    --security-opt seccomp=unconfined --user \$(id -u):\$(id -g) -e HOME=/tmp -e GPU_MAX_HW_QUEUES=1 -e HIP_VISIBLE_DEVICES=$gpus \
    -v \$HOME/intarsiaml-rocm102:/work -v $MODELS_DIR:/models:ro -v $VOL:/target -w /work $IMG /target/release/intarsia serve --backend hip --model $MODEL \
    --served-model-name $NAME-$arm $args --profile-every 0 --mem-budget 29 --max-seq $MAXSEQ --no-supervisor --host 0.0.0.0 --port $PORT" < /dev/null > "$OUT/docker-$arm.id" 2>&1
  for i in $(seq 1 140); do
    curl -sf -m 3 "http://tatooine:$PORT/health" > /dev/null 2>&1 && { say "$arm healthy after ~$((i * 3)) s"; return 0; }
    sleep 3
  done
  $SSH "docker logs --tail 40 $CNAME" > "$OUT/serve-fail-$arm.log" 2>&1 < /dev/null
  return 1
}

for ph in $PHASES; do
  IFS=: read -r PNAME SIZES ROUNDS PARMS <<<"$ph"
  parms=${PARMS//,/ }
  for r in $(seq 1 "$ROUNDS"); do
    order="$parms"
    (( r % 2 == 0 )) && order=$(echo $parms | awk '{for (i=NF;i>=1;i--) printf "%s ", $i}')
    for arm in $order; do
      if (( $(left) < 300 )); then say "SKIPPED phase $PNAME round $r arm $arm (time left $(left)s)"; continue; fi
      say "phase $PNAME round $r arm $arm: starting server"
      if ! serve "$arm"; then say "phase $PNAME round $r arm $arm: server did not come up"; drop; continue; fi
      say "phase $PNAME round $r arm $arm: server up, localeval"
      ( cd /opt/code/deepeval && uv run localeval speed --url "http://tatooine:$PORT/v1" --model "$NAME-$arm" --label "$TAGP-$PNAME-$arm-r$r" \
          --sizes "$SIZES" --gen "$GEN" --reps 1 --thinking off --wait-idle 60 --recipe "$RECIPE" ) \
        > "$OUT/localeval-$PNAME-$arm-r$r.log" 2>&1 < /dev/null
      say "phase $PNAME round $r arm $arm: localeval exit $? $(grep -aE 'run_id|results/' "$OUT/localeval-$PNAME-$arm-r$r.log" | tail -2 | tr '\n' '|')"
      $SSH "docker logs $CNAME 2>&1 | grep -aE 'prefill|tensor-parallel|admission|compiler' | tail -12" < /dev/null > "$OUT/serve-$PNAME-$arm-r$r.log" 2>&1
      drop
    done
  done
done
say "done, time left $(left)s"

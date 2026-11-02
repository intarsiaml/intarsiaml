#!/usr/bin/env bash
# Lane r9s: 35B NVFP4 multi-stream aggregate decode on ONE R9700 (tp=1, HIP), driven FROM THE MAC under one fleetq job that holds both
# R9700 cards and takes radiance down (derived from scripts/tp2-hip/speed-window.sh, tp1 arm only):
#   fleetq run --host tatooine-r9700,tatooine-r9700-2 --takes-down radiance --quiet gpu,cpu --expect 40 --mem 30 \
#     --label "R9S R9700 TP1 35B streams 1/4/8/16" -- scripts/r9s/window.sh
# Server per phase round (fresh container, card 0 only, `serve --max-batch 16 --spec off`, thinking off); then `localeval speed`
# (1k prompts, aggregate decode at CONC streams plus the single-stream row), then the server goes away. Phases (PHASES, space separated
# "name:gen:concurrency:rounds"): g512 = gen 512, 4/8/16 streams, 3 rounds; g128 = gen 128, 4/16 streams, 1 round (the 2026-10-25 setting).
# localeval runs inside this job's slot (FLEETQ_JOB is inherited). Radiance is restored by guard.sh on tatooine when this script ends
# or dies (trap), and the script waits for it and appends the downtime to OUT/downtime.log. TREE = the tree built by build.sh.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
TREE=${TREE:-r9s}; CAP=${CAP:-2400}; SIZES=${SIZES:-1024}
PHASES=${PHASES:-"g512:512:4,8,16:3 g128:128:4,16:1"}
MODEL=/models/Qwen3.6-35B-A3B-NVFP4; NAME=qwen36-35b; MODELS_DIR=/data/models/intarsia
MAXB=16; MAXSEQ=${MAXSEQ:-18432}; PORT=8090
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
OUT=${OUT:-$REPO/docs/results/2026-10-29-r9s-r9700-tp1-streams/raw/window-$(date +%H%M)}; mkdir -p "$OUT"
T0=$(date +%s); left() { echo $(( CAP - ($(date +%s) - T0) )); }
say() { echo "$(date +%T) $*" | tee -a "$OUT/steps.log"; }
SSH="ssh -o BatchMode=yes tatooine"

# guard: stops radiance now, restores it when its stdin closes (we exit or die) and reports when it answers
FIFO="$OUT/guard.fifo"; mkfifo "$FIFO"
$SSH 'mkdir -p ~/r9s' < /dev/null; scp -q "$HERE/guard.sh" tatooine:r9s/guard.sh
$SSH 'bash ~/r9s/guard.sh' > "$OUT/guard.log" 2>&1 < "$FIFO" &
GUARD=$!
exec 9> "$FIFO"
for i in $(seq 1 30); do grep -q "radiance stopped" "$OUT/guard.log" 2>/dev/null && break; sleep 1; done
say "guard: $(grep -h 'radiance stopped' "$OUT/guard.log")"
stopserver() { $SSH 'docker rm -f r9s-serve' >/dev/null 2>&1 < /dev/null; }
cleanup() {
  stopserver; exec 9>&-; wait $GUARD; rm -f "$FIFO"
  say "guard: $(grep -h 'stopping radiance\|radiance answers\|DID NOT' "$OUT/guard.log" | tr '\n' '|')"
  echo "radiance down: $(grep -h 'stopping radiance' "$OUT/guard.log" | sed 's/.*radiance //') -> $(grep -h 'radiance answers\|DID NOT' "$OUT/guard.log" | sed 's/.*\] //')  (fleetq job $FLEETQ_JOB)" >> "$OUT/downtime.log"
}
trap cleanup EXIT

D="docker run --rm -v \$HOME/intarsiaml-tp2-$TREE:/work -w /work $IMG /work/tp2-intarsia"
$SSH "$D --version" < /dev/null > "$OUT/versions.txt" 2>&1; echo "tree stamp: $(cat "$REPO/docs/results/2026-10-29-r9s-r9700-tp1-streams/raw/build.log" | grep -a '^stamp:')" >> "$OUT/versions.txt"
say "binary: $(head -c 300 "$OUT/versions.txt" | tr '\n' ' ')"

serve() { # serve <tag>
  local tag=$1
  RECIPE="HIP_VISIBLE_DEVICES=0 GPU_MAX_HW_QUEUES=1 intarsia serve --backend hip --model $MODEL --served-model-name $NAME --max-batch $MAXB --spec off --profile-every 0 --mem-budget 29 --max-seq $MAXSEQ --no-supervisor --host 0.0.0.0 --port $PORT (tree $TREE, docker $IMG)"
  stopserver
  $SSH "docker run -d --name r9s-serve --network host --device /dev/kfd --device /dev/dri \
    --group-add \$(stat -c %g /dev/kfd) --group-add \$(stat -c %g /dev/dri/renderD128) --group-add video --ipc host \
    --security-opt seccomp=unconfined --user \$(id -u):\$(id -g) -e HOME=/tmp -e GPU_MAX_HW_QUEUES=1 -e HIP_VISIBLE_DEVICES=0 \
    -v \$HOME/intarsiaml-tp2-$TREE:/work -v $MODELS_DIR:/models:ro -w /work $IMG /work/tp2-intarsia serve --backend hip --model $MODEL \
    --served-model-name $NAME --max-batch $MAXB --spec off --profile-every 0 --mem-budget 29 --max-seq $MAXSEQ --no-supervisor --host 0.0.0.0 --port $PORT" \
    < /dev/null > "$OUT/docker-$tag.id" 2>&1
  for i in $(seq 1 100); do
    curl -sf -m 3 "http://tatooine:$PORT/health" > /dev/null 2>&1 && return 0
    [[ $($SSH "docker inspect -f '{{.State.Running}}' r9s-serve" < /dev/null 2>/dev/null) == true ]] || break   # container died: no point waiting
    sleep 3
  done
  $SSH 'docker logs --tail 60 r9s-serve' > "$OUT/serve-fail-$tag.log" 2>&1 < /dev/null
  return 1
}

for ph in $PHASES; do
  IFS=: read -r PNAME GEN CONC PR <<<"$ph"
  for r in $(seq 1 "$PR"); do
    tag="$PNAME-r$r"
    if (( $(left) < 300 )); then say "SKIPPED $tag (time left $(left)s)"; continue; fi
    say "$tag: starting server"
    # the same flags fail the same way every round: stop at the first failed start so radiance comes back (trap)
    if ! serve "$tag"; then say "$tag: server did not come up, aborting the window (see serve-fail-$tag.log)"; stopserver; exit 1; fi
    say "$tag: server up, localeval (gen $GEN, streams $CONC)"
    ( cd /opt/code/deepeval && uv run localeval speed --url "http://tatooine:$PORT/v1" --model "$NAME" --label "r9s-$tag" \
        --sizes "$SIZES" --gen "$GEN" --reps 1 --concurrency "$CONC" --thinking off --wait-idle 60 --recipe "$RECIPE" ) \
      > "$OUT/localeval-$tag.log" 2>&1 < /dev/null
    say "$tag: localeval exit $? $(grep -aE 'run_id|results/' "$OUT/localeval-$tag.log" | tail -2 | tr '\n' '|')"
    $SSH 'docker logs r9s-serve 2>&1 | grep -aiE "stamp|head=|code-tree|batch|graph|spec|mtp|listening" | tail -8' < /dev/null > "$OUT/serve-$tag.log" 2>&1
    stopserver
  done
done
say "done, time left $(left)s"

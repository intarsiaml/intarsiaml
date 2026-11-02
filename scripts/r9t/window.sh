#!/usr/bin/env bash
# Lane r9t: Qwen3.6-35B-A3B NVFP4 on the R9700s (HIP): multi-stream aggregate decode at tp=2 (both cards) plus a single-stream
# prefill / decode phase at tp=2 and tp=1, driven FROM THE MAC under one fleetq job that holds both R9700 cards and takes radiance down
# (derived from scripts/r9s/window.sh and the tp2 / b1 arms of scripts/tp2-hip/speed-window.sh):
#   fleetq run --host tatooine-r9700,tatooine-r9700-2 --takes-down radiance --quiet gpu,cpu --expect 40 --mem 30 \
#     --label "R9T R9700 TP2 35B streams 1/4/8/16 + b1" -- scripts/r9t/window.sh
# Server per phase round (fresh container, `serve --spec off`, thinking off); then `localeval speed`, then the server goes away.
# localeval runs inside this job's slot (FLEETQ_JOB is inherited). Radiance is restored by guard.sh on tatooine when this script ends
# or dies (trap), and the script waits for it and appends the downtime to OUT/downtime.log. TREE = the tree on tatooine (default r9s:
# the binary the R9S lane built, reused; see the doc). PHASES: see below.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
TREE=${TREE:-r9s}; CAP=${CAP:-2400}
MODEL=/models/Qwen3.6-35B-A3B-NVFP4; NAME=qwen36-35b; MODELS_DIR=/data/models/intarsia
MAXSEQ=${MAXSEQ:-18432}; PORT=8090
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
OUT=${OUT:-$REPO/docs/results/2026-10-29-r9t-r9700-tp2-streams/raw/window-$(date +%H%M)}; mkdir -p "$OUT"
T0=$(date +%s); left() { echo $(( CAP - ($(date +%s) - T0) )); }
say() { echo "$(date +%T) $*" | tee -a "$OUT/steps.log"; }
SSH="ssh -o BatchMode=yes tatooine"

# guard: stops radiance now, restores it when its stdin closes (we exit or die) and reports when it answers
FIFO="$OUT/guard.fifo"; mkfifo "$FIFO"
$SSH 'mkdir -p ~/r9t' < /dev/null; scp -q "$HERE/guard.sh" tatooine:r9t/guard.sh
$SSH 'bash ~/r9t/guard.sh' > "$OUT/guard.log" 2>&1 < "$FIFO" &
GUARD=$!
exec 9> "$FIFO"
for i in $(seq 1 30); do grep -q "radiance stopped" "$OUT/guard.log" 2>/dev/null && break; sleep 1; done
say "guard: $(grep -h 'radiance stopped' "$OUT/guard.log")"
stopserver() { $SSH 'docker rm -f r9t-serve' >/dev/null 2>&1 < /dev/null; }
cleanup() {
  stopserver; exec 9>&-; wait $GUARD; rm -f "$FIFO"
  say "guard: $(grep -h 'stopping radiance\|radiance answers\|DID NOT' "$OUT/guard.log" | tr '\n' '|')"
  echo "radiance down: $(grep -h 'stopping radiance' "$OUT/guard.log" | sed 's/.*radiance //') -> $(grep -h 'radiance answers\|DID NOT' "$OUT/guard.log" | sed 's/.*\] //')  (fleetq job $FLEETQ_JOB)" >> "$OUT/downtime.log"
}
trap cleanup EXIT

D="docker run --rm -v \$HOME/intarsiaml-tp2-$TREE:/work -w /work $IMG /work/tp2-intarsia"
$SSH "$D --version" < /dev/null > "$OUT/versions.txt" 2>&1; echo "tree stamp (R9S build of the reused tree): $(cat "$REPO/docs/results/2026-10-29-r9s-r9700-tp1-streams/raw/build.log" | grep -a '^stamp:')" >> "$OUT/versions.txt"
say "binary: $(head -c 300 "$OUT/versions.txt" | tr '\n' ' ')"

# server <arm> <maxbatch> <tag>: arm tp2 = both cards (--tp 2 --devices 0,1), tp1 = card 0 only; maxbatch 1 = no --max-batch flag
serve() {
  local arm=$1 mb=$2 tag=$3 gpus=0 args=""
  [[ $mb -gt 1 ]] && args="--max-batch $mb"
  [[ $arm == tp2 ]] && { gpus=0,1; args="$args --tp 2 --devices 0,1"; }
  RECIPE="HIP_VISIBLE_DEVICES=$gpus GPU_MAX_HW_QUEUES=1 intarsia serve --backend hip --model $MODEL --served-model-name $NAME $args --spec off --profile-every 0 --mem-budget 29 --max-seq $MAXSEQ --no-supervisor --host 0.0.0.0 --port $PORT (tree $TREE, docker $IMG)"
  stopserver
  $SSH "docker run -d --name r9t-serve --network host --device /dev/kfd --device /dev/dri \
    --group-add \$(stat -c %g /dev/kfd) --group-add \$(stat -c %g /dev/dri/renderD128) --group-add video --ipc host \
    --security-opt seccomp=unconfined --user \$(id -u):\$(id -g) -e HOME=/tmp -e GPU_MAX_HW_QUEUES=1 -e HIP_VISIBLE_DEVICES=$gpus \
    -v \$HOME/intarsiaml-tp2-$TREE:/work -v $MODELS_DIR:/models:ro -w /work $IMG /work/tp2-intarsia serve --backend hip --model $MODEL \
    --served-model-name $NAME $args --spec off --profile-every 0 --mem-budget 29 --max-seq $MAXSEQ --no-supervisor --host 0.0.0.0 --port $PORT" \
    < /dev/null > "$OUT/docker-$tag.id" 2>&1
  for i in $(seq 1 100); do
    curl -sf -m 3 "http://tatooine:$PORT/health" > /dev/null 2>&1 && return 0
    [[ $($SSH "docker inspect -f '{{.State.Running}}' r9t-serve" < /dev/null 2>/dev/null) == true ]] || break   # container died: no point waiting
    sleep 3
  done
  $SSH 'docker logs --tail 60 r9t-serve' > "$OUT/serve-fail-$tag.log" 2>&1 < /dev/null
  return 1
}

# PHASES: "name:maxbatch:sizes:gen:concurrency:rounds:arms" separated by spaces (concurrency may be empty; arms comma separated, rounds
# interleaved, the arm order reversed on even rounds). g512 / g128 = the multi-stream phases on tp2; b1 = single stream at max-batch 1,
# 1k/4k/16k prompts, gen 512 (the 2026-10-25 b1 phase of 2026-10-25-tp2-hip.md section 10.1), tp2 and tp1 interleaved.
PHASES=${PHASES:-"g512:16:1024:512:4,8,16:3:tp2 g128:16:1024:128:4,16:1:tp2 b1:1:1024,4096,16384:512::3:tp2,tp1"}
for ph in $PHASES; do
  IFS=: read -r PNAME MB PSIZES GEN CONC PR PARMS <<<"$ph"
  parms=${PARMS//,/ }
  for r in $(seq 1 "$PR"); do
    order="$parms"
    (( r % 2 == 0 )) && order=$(echo $parms | awk '{for (i=NF;i>=1;i--) printf "%s ", $i}')
    for arm in $order; do
      tag="$PNAME-$arm-r$r"
      if (( $(left) < 300 )); then say "SKIPPED $tag (time left $(left)s)"; continue; fi
      say "$tag: starting server"
      # the same flags fail the same way every round: stop at the first failed start so radiance comes back (trap)
      if ! serve "$arm" "$MB" "$tag"; then say "$tag: server did not come up, aborting the window (see serve-fail-$tag.log)"; stopserver; exit 1; fi
      say "$tag: server up, localeval (sizes $PSIZES, gen $GEN, streams ${CONC:-1})"
      ( cd /opt/code/deepeval && uv run localeval speed --url "http://tatooine:$PORT/v1" --model "$NAME" --label "r9t-$tag" \
          --sizes "$PSIZES" --gen "$GEN" --reps 1 ${CONC:+--concurrency "$CONC"} --thinking off --wait-idle 60 --recipe "$RECIPE" ) \
        > "$OUT/localeval-$tag.log" 2>&1 < /dev/null
      say "$tag: localeval exit $? $(grep -aE 'run_id|results/' "$OUT/localeval-$tag.log" | tail -2 | tr '\n' '|')"
      $SSH 'docker logs r9t-serve 2>&1 | grep -aiE "stamp|head=|code-tree|batch|graph|spec|mtp|listening|tensor-parallel" | tail -8' < /dev/null > "$OUT/serve-$tag.log" 2>&1
      stopserver
    done
  done
done
say "done, time left $(left)s"

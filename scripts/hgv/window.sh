#!/usr/bin/env bash
# Lane hgv: A/B of main (tree hgvbase) against the hgv tree for Qwen3.6-35B-A3B NVFP4 on the R9700s (HIP), driven FROM THE MAC under one fleetq
# job that holds both R9700 cards and takes radiance down (derived from scripts/r9t/window.sh):
#   ROUND=1 MODE=perf ARMS="tp1:base tp1:hgv tp2:hgv tp2:base" fleetq run --host tatooine-r9700,tatooine-r9700-2 --takes-down radiance \
#     --quiet gpu,cpu --expect 20 --mem 30 --label "HGV A/B perf round 1" -- scripts/hgv/window.sh
# One fresh server per arm (`--max-batch 16 --spec off`, tp2 = `--tp 2 --devices 0,1`), then, by MODE:
#   perf   `localeval speed --sizes 1024 --gen 512 --concurrency 4,8,16` (the single-stream row, then 4, 8, 16 streams; decode tok/s)
#   ident  scripts/hgv/ident.py: greedy text of 1 / 8 / 16 streams, same prompt and different prompts (gen 384), saved as ident-<arm>.json
# Radiance is restored by guard.sh on tatooine when this script ends or dies (trap); the script waits for it and appends the downtime to
# OUT/downtime.log. TREE per arm: base -> ~/intarsiaml-tp2-hgvbase, hgv -> ~/intarsiaml-tp2-hgv (scripts/hgv/build.sh).
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
MODE=${MODE:-perf}; ROUND=${ROUND:-1}; CAP=${CAP:-1150}; ARMS=${ARMS:-"tp1:base tp1:hgv tp2:hgv tp2:base"}
MODEL=/models/Qwen3.6-35B-A3B-NVFP4; NAME=qwen36-35b; MODELS_DIR=/data/models/intarsia
MAXSEQ=${MAXSEQ:-18432}; PORT=8090; GEN=${GEN:-512}; IGEN=${IGEN:-384}
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
OUT=${OUT:-$REPO/docs/results/2026-10-30-hgv-hip-small-gemv-grid/raw/$MODE-r$ROUND-$(date +%H%M)}; mkdir -p "$OUT"
T0=$(date +%s); left() { echo $(( CAP - ($(date +%s) - T0) )); }
say() { echo "$(date +%T) $*" | tee -a "$OUT/steps.log"; }
SSH="ssh -o BatchMode=yes tatooine"

FIFO="$OUT/guard.fifo"; mkfifo "$FIFO"
$SSH 'mkdir -p ~/hgv' < /dev/null; scp -q "$HERE/guard.sh" tatooine:hgv/guard.sh
$SSH 'bash ~/hgv/guard.sh' > "$OUT/guard.log" 2>&1 < "$FIFO" &
GUARD=$!
exec 9> "$FIFO"
for i in $(seq 1 30); do grep -q "radiance stopped" "$OUT/guard.log" 2>/dev/null && break; sleep 1; done
say "guard: $(grep -h 'radiance stopped' "$OUT/guard.log")"
stopserver() { $SSH 'docker rm -f hgv-serve' >/dev/null 2>&1 < /dev/null; }
cleanup() {
  stopserver; exec 9>&-; wait $GUARD; rm -f "$FIFO"
  say "guard: $(grep -h 'stopping radiance\|radiance answers\|DID NOT' "$OUT/guard.log" | tr '\n' '|')"
  echo "radiance down: $(grep -h 'stopping radiance' "$OUT/guard.log" | sed 's/.*radiance //') -> $(grep -h 'radiance answers\|DID NOT' "$OUT/guard.log" | sed 's/.*\] //')  (fleetq job $FLEETQ_JOB)" >> "$OUT/downtime.log"
}
trap cleanup EXIT

# serve <arm> <tag>: arm = tp1:base | tp1:hgv | tp2:base | tp2:hgv
serve() {
  local tp=${1%%:*} tree=${1##*:} tag=$2 gpus=0 args="--max-batch 16"
  [[ $tp == tp2 ]] && { gpus=0,1; args="$args --tp 2 --devices 0,1"; }
  RECIPE="HIP_VISIBLE_DEVICES=$gpus GPU_MAX_HW_QUEUES=1 intarsia serve --backend hip --model $MODEL --served-model-name $NAME $args --spec off --profile-every 0 --mem-budget 29 --max-seq $MAXSEQ --no-supervisor --host 0.0.0.0 --port $PORT (tree hgv$([[ $tree == base ]] && echo base), docker $IMG)"
  local dir='$HOME/intarsiaml-tp2-hgv'; [[ $tree == base ]] && dir='$HOME/intarsiaml-tp2-hgvbase'
  stopserver
  $SSH "docker run -d --name hgv-serve --network host --device /dev/kfd --device /dev/dri \
    --group-add \$(stat -c %g /dev/kfd) --group-add \$(stat -c %g /dev/dri/renderD128) --group-add video --ipc host \
    --security-opt seccomp=unconfined --user \$(id -u):\$(id -g) -e HOME=/tmp -e GPU_MAX_HW_QUEUES=1 -e HIP_VISIBLE_DEVICES=$gpus \
    -v $dir:/work -v $MODELS_DIR:/models:ro -w /work $IMG /work/tp2-intarsia serve --backend hip --model $MODEL \
    --served-model-name $NAME $args --spec off --profile-every 0 --mem-budget 29 --max-seq $MAXSEQ --no-supervisor --host 0.0.0.0 --port $PORT" \
    < /dev/null > "$OUT/docker-$tag.id" 2>&1
  for i in $(seq 1 100); do
    curl -sf -m 3 "http://tatooine:$PORT/health" > /dev/null 2>&1 && return 0
    [[ $($SSH "docker inspect -f '{{.State.Running}}' hgv-serve" < /dev/null 2>/dev/null) == true ]] || break
    sleep 3
  done
  $SSH 'docker logs --tail 60 hgv-serve' > "$OUT/serve-fail-$tag.log" 2>&1 < /dev/null
  return 1
}

for arm in $ARMS; do
  tag="$MODE-r$ROUND-${arm/:/-}"
  if (( $(left) < 280 )); then say "SKIPPED $tag (time left $(left)s)"; continue; fi
  say "$tag: starting server"
  if ! serve "$arm" "$tag"; then say "$tag: server did not come up, aborting the window (see serve-fail-$tag.log)"; stopserver; exit 1; fi
  if [[ $MODE == perf ]]; then
    say "$tag: server up, localeval (1024 prompt, gen $GEN, streams 1,4,8,16)"
    ( cd /opt/code/deepeval && uv run localeval speed --url "http://tatooine:$PORT/v1" --model "$NAME" --label "hgv-$tag" \
        --sizes 1024 --gen "$GEN" --reps 1 --concurrency 4,8,16 --thinking off --wait-idle 60 --recipe "$RECIPE" ) \
      > "$OUT/localeval-$tag.log" 2>&1 < /dev/null
    say "$tag: localeval exit $? $(grep -aE 'run_id|results/' "$OUT/localeval-$tag.log" | tail -2 | tr '\n' '|')"
  else
    say "$tag: server up, ident (gen $IGEN)"
    ( cd /opt/code/deepeval && uv run python "$HERE/ident.py" --url "http://tatooine:$PORT/v1" --model "$NAME" --docs-list "$HERE/ident_docs.txt" \
        --root "$REPO" --gen "$IGEN" --out "$OUT/ident-${arm/:/-}.json" ) > "$OUT/ident-$tag.jsonl" 2> "$OUT/ident-$tag.err" < /dev/null
    say "$tag: ident exit $?: $(cut -c1-120 "$OUT/ident-$tag.jsonl" | tr '\n' '|')"
  fi
  $SSH 'docker logs hgv-serve 2>&1 | grep -aiE "stamp|head=|code-tree|batch|graph|listening|tensor-parallel" | tail -8' < /dev/null > "$OUT/serve-$tag.log" 2>&1
  stopserver
done
say "done, time left $(left)s"

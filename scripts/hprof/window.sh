#!/usr/bin/env bash
# Lane hprof: where the time goes in one batched decode step of Qwen3.6-35B-A3B NVFP4 on the R9700s (HIP), driven FROM THE MAC under
# one fleetq job that holds both R9700 cards and takes radiance down (derived from scripts/r9t/window.sh):
#   ARM=tp1 STEPS="prof route plain" fleetq run --host tatooine-r9700,tatooine-r9700-2 --takes-down radiance --quiet gpu,cpu \
#     --expect 25 --mem 30 --label "HPROF R9700 35B decode profile tp1" -- scripts/hprof/window.sh
# ARM tp1 = card 0, tp2 = both cards (--tp 2 --devices 0,1). STEPS (space separated, run in order, one fresh server each):
#   plain  the server as served (graphs on) with INTARSIA_HOST_PROFILE=1; localeval gives the unprofiled aggregate tok/s
#   prof   the same server under `rocprofv3 --kernel-trace` (graph replays show up as kernels); the server is ended through
#          INTARSIA_HIP_EXIT_FILE so the profiler flushes; the CSV is copied to OUT
#   route  eager steps (INTARSIA_BATCH_GRAPHS=0) with INTARSIA_HIP_ROUTE_STATS=1: distinct routed experts per layer per step
# Workload per server: step plain first runs `localeval speed --sizes 1024 --gen $GEN --concurrency 8,16` (single-stream row, then 8 and 16 streams);
# every step then runs scripts/hprof/drive.py PHASES (b1, same8, same16, div8, div16: identical prompts as localeval sends, and 8 / 16 different docs).
# Radiance is restored by guard.sh on tatooine when this script ends or dies (trap); the script waits for it and appends the downtime
# to OUT/downtime.log. Optional: XENV="-e VAR=val ..." (extra docker env for the prof step), TAGSUF (suffix of the step tag), PHASES, GENP. TREE = the tree on tatooine built by scripts/hprof/build.sh.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
ARM=${ARM:-tp1}; STEPS=${STEPS:-"prof route plain"}; TREE=${TREE:-hprof}; CAP=${CAP:-950}; GEN=${GEN:-384}; GENP=${GENP:-192}; CONC=${CONC:-8,16}; PHASES=${PHASES:-b1,same8,same16,div8,div16}; XENV=${XENV:-}; TAGSUF=${TAGSUF:-}
MODEL=/models/Qwen3.6-35B-A3B-NVFP4; NAME=qwen36-35b; MODELS_DIR=/data/models/intarsia
MAXB=16; MAXSEQ=${MAXSEQ:-18432}; PORT=8090
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0; PIMG=intarsia-hip-prof:rocm7.2.4-rust1.90.0
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
OUT=${OUT:-$REPO/docs/results/2026-10-30-hprof-r9700-16stream-profile/raw/window-$ARM-$(date +%H%M)}; mkdir -p "$OUT"
T0=$(date +%s); left() { echo $(( CAP - ($(date +%s) - T0) )); }
say() { echo "$(date +%T) $*" | tee -a "$OUT/steps.log"; }
SSH="ssh -o BatchMode=yes tatooine"
TREEDIR='$HOME/intarsiaml-tp2-'$TREE

FIFO="$OUT/guard.fifo"; mkfifo "$FIFO"
$SSH 'mkdir -p ~/hprof' < /dev/null; scp -q "$HERE/guard.sh" tatooine:hprof/guard.sh
$SSH 'bash ~/hprof/guard.sh' > "$OUT/guard.log" 2>&1 < "$FIFO" &
GUARD=$!
exec 9> "$FIFO"
for i in $(seq 1 30); do grep -q "radiance stopped" "$OUT/guard.log" 2>/dev/null && break; sleep 1; done
say "guard: $(grep -h 'radiance stopped' "$OUT/guard.log")"
stopserver() { $SSH 'docker rm -f hprof-serve' >/dev/null 2>&1 < /dev/null; }
cleanup() {
  stopserver; exec 9>&-; wait $GUARD; rm -f "$FIFO"
  say "guard: $(grep -h 'stopping radiance\|radiance answers\|DID NOT' "$OUT/guard.log" | tr '\n' '|')"
  echo "radiance down: $(grep -h 'stopping radiance' "$OUT/guard.log" | sed 's/.*radiance //') -> $(grep -h 'radiance answers\|DID NOT' "$OUT/guard.log" | sed 's/.*\] //')  (fleetq job $FLEETQ_JOB)" >> "$OUT/downtime.log"
}
trap cleanup EXIT

D="docker run --rm -v $TREEDIR:/work -w /work $IMG /work/tp2-intarsia"
$SSH "$D --version" < /dev/null > "$OUT/versions.txt" 2>&1
say "binary: $(head -c 300 "$OUT/versions.txt" | tr '\n' ' ')"

gpus=0; TPARGS=""
[[ $ARM == tp2 ]] && { gpus=0,1; TPARGS="--tp 2 --devices 0,1"; }
SERVE="serve --backend hip --model $MODEL --served-model-name $NAME --max-batch $MAXB $TPARGS --spec off --profile-every 0 --mem-budget 29 --max-seq $MAXSEQ --no-supervisor --host 0.0.0.0 --port $PORT"

# serve <step> <tag>: start the container for the step; returns 0 when /health answers
serve() {
  local step=$1 tag=$2 img=$IMG envs="-e INTARSIA_HOST_PROFILE=1" cmd="/work/tp2-intarsia $SERVE"
  case $step in
    prof) img=$PIMG; envs="$envs $XENV -e INTARSIA_HIP_EXIT_FILE=/work/hprof-out/$tag.exit"
          cmd="rocprofv3 --kernel-trace -f csv -d /work/hprof-out/$tag -o k -- /work/tp2-intarsia $SERVE"
          $SSH "mkdir -p ~/intarsiaml-tp2-$TREE/hprof-out && rm -rf ~/intarsiaml-tp2-$TREE/hprof-out/$tag ~/intarsiaml-tp2-$TREE/hprof-out/$tag.exit" < /dev/null ;;
    route) envs="-e INTARSIA_HIP_ROUTE_STATS=1 -e INTARSIA_BATCH_GRAPHS=0" ;;
  esac
  stopserver
  $SSH "docker run -d --name hprof-serve --network host --device /dev/kfd --device /dev/dri \
    --group-add \$(stat -c %g /dev/kfd) --group-add \$(stat -c %g /dev/dri/renderD128) --group-add video --ipc host \
    --security-opt seccomp=unconfined --user \$(id -u):\$(id -g) -e HOME=/tmp -e GPU_MAX_HW_QUEUES=1 -e HIP_VISIBLE_DEVICES=$gpus $envs \
    -v $TREEDIR:/work -v $MODELS_DIR:/models:ro -w /work $img $cmd" < /dev/null > "$OUT/docker-$tag.id" 2>&1
  for i in $(seq 1 100); do
    curl -sf -m 3 "http://tatooine:$PORT/health" > /dev/null 2>&1 && return 0
    [[ $($SSH "docker inspect -f '{{.State.Running}}' hprof-serve" < /dev/null 2>/dev/null) == true ]] || break
    sleep 3
  done
  $SSH 'docker logs --tail 80 hprof-serve' > "$OUT/serve-fail-$tag.log" 2>&1 < /dev/null
  return 1
}

for step in $STEPS; do
  tag="$ARM-$step$TAGSUF"
  if (( $(left) < 350 )); then say "SKIPPED $tag (time left $(left)s)"; continue; fi
  say "$tag: starting server"
  if ! serve "$step" "$tag"; then say "$tag: server did not come up, aborting the window (see serve-fail-$tag.log)"; stopserver; exit 1; fi
  RECIPE="HIP_VISIBLE_DEVICES=$gpus GPU_MAX_HW_QUEUES=1 [step $step] intarsia $SERVE (tree $TREE)"
  if [[ $step == plain ]]; then
    say "$tag: server up, localeval (1024 prompt, gen $GEN, streams 1,$CONC)"
    ( cd /opt/code/deepeval && uv run localeval speed --url "http://tatooine:$PORT/v1" --model "$NAME" --label "hprof-$tag" \
        --sizes 1024 --gen "$GEN" --reps 1 --concurrency "$CONC" --thinking off --wait-idle 60 --recipe "$RECIPE" ) \
      > "$OUT/localeval-$tag.log" 2>&1 < /dev/null
    say "$tag: localeval exit $? $(grep -aE 'run_id|results/' "$OUT/localeval-$tag.log" | tail -2 | tr '\n' '|')"
  fi
  G=$GEN; [[ $step == prof ]] && G=$GENP   # the profiled step writes a trace per kernel launch: keep it short
  say "$tag: driver phases $PHASES (gen $G)"
  ( cd /opt/code/deepeval && uv run python "$HERE/drive.py" --url "http://tatooine:$PORT/v1" --model "$NAME" --docs "$REPO/docs" --gen "$G" --phases "$PHASES" ) \
    > "$OUT/drive-$tag.jsonl" 2> "$OUT/drive-$tag.err" < /dev/null
  say "$tag: driver exit $?: $(tail -n +2 "$OUT/drive-$tag.jsonl" | cut -c1-200 | tr '\n' '|')"
  if [[ $step == prof ]]; then
    $SSH "touch ~/intarsiaml-tp2-$TREE/hprof-out/$tag.exit" < /dev/null
    # rocprofv3 writes the CSV at finalize: wait for the container to be gone (up to 300 s) before copying
    for i in $(seq 1 100); do
      [[ $($SSH "docker inspect -f '{{.State.Running}}' hprof-serve" < /dev/null 2>/dev/null) == true ]] || break
      sleep 3
    done
    say "$tag: server state after exit file (waited $((i * 3)) s): $($SSH "docker inspect -f '{{.State.Running}} exit={{.State.ExitCode}}' hprof-serve" < /dev/null 2>&1)"
    $SSH "find ~/intarsiaml-tp2-$TREE/hprof-out/$tag -name '*.csv' -exec gzip -f {} +" < /dev/null > /dev/null 2>&1
    $SSH "find ~/intarsiaml-tp2-$TREE/hprof-out/$tag -type f -printf '%s %p\n'" < /dev/null > "$OUT/prof-files-$tag.txt" 2>&1
    mkdir -p "$OUT/prof-$tag"
    rsync -a -e "ssh -o BatchMode=yes" "tatooine:intarsiaml-tp2-$TREE/hprof-out/$tag/" "$OUT/prof-$tag/" < /dev/null > "$OUT/rsync-$tag.log" 2>&1
    say "$tag: copied $(du -sh "$OUT/prof-$tag" | cut -f1) ($(cat "$OUT/prof-files-$tag.txt" | tr '\n' ' '))"
  fi
  $SSH 'docker logs hprof-serve 2>&1' < /dev/null > "$OUT/serve-$tag.log" 2>&1
  stopserver
done
say "done, time left $(left)s"

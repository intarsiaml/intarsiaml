#!/usr/bin/env bash
# Lane wmr (derived from scripts/r9t/window.sh): Qwen3.6-35B-A3B NVFP4 on the R9700s (HIP), HIP_WMMA_ROWS family OFF vs ON, tp1 and tp2, driven FROM THE MAC
# under one fleetq job that holds both R9700 cards and takes radiance down:
#   fleetq run --host tatooine-r9700,tatooine-r9700-2 --takes-down radiance --quiet gpu,cpu --expect 20 --mem 30 \
#     --label "WMR ..." -- env PHASES='...' scripts/wmr/window.sh
# One fresh server per (phase, round, arm) (`serve --spec off`, thinking off), then `localeval speed`, optionally scripts/wmr/ident.py (cross-stream greedy
# identity, round 1 only), then the server goes away. Radiance is restored by guard.sh on tatooine when this script ends or dies (trap).
# Arm = <tp>-<knob>: tp1 (card 0) | tp2 (--tp 2 --devices 0,1); off = main default, on = INTARSIA_HIP_WMMA_ROWS=1 INTARSIA_HIP_WMMA_TABLE=fp8a8=all,bf16=all
# (the wg "option 1" table: ROWS=1 alone is the W8A16-only default table, which is a no-op on the 35B).
# PHASES (space separated): name:maxbatch:sizes:gen:concurrency:rounds:arms:ident ; arms comma separated; the arm order alternates by round so the
# two arms of a pair are adjacent; ident 1 = run ident.py in round 1.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
TREE=${TREE:-wmr}; CAP=${CAP:-1080}
MODEL=/models/Qwen3.6-35B-A3B-NVFP4; NAME=qwen36-35b; MODELS_DIR=/data/models/intarsia
MAXSEQ=${MAXSEQ:-18432}; PORT=8090
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
KNOB="INTARSIA_HIP_WMMA_ROWS=1 INTARSIA_HIP_WMMA_TABLE=fp8a8=all,bf16=all"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
OUT=${OUT:?set OUT to the raw window folder}; mkdir -p "$OUT"
T0=$(date +%s); left() { echo $(( CAP - ($(date +%s) - T0) )); }
say() { echo "$(date +%T) $*" | tee -a "$OUT/steps.log"; }
SSH="ssh -o BatchMode=yes tatooine"

FIFO="$OUT/guard.fifo"; mkfifo "$FIFO"
$SSH 'mkdir -p ~/wmr' < /dev/null; scp -q "$HERE/guard.sh" tatooine:wmr/guard.sh
$SSH 'bash ~/wmr/guard.sh' > "$OUT/guard.log" 2>&1 < "$FIFO" &
GUARD=$!
exec 9> "$FIFO"
for i in $(seq 1 30); do grep -q "radiance stopped" "$OUT/guard.log" 2>/dev/null && break; sleep 1; done
say "guard: $(grep -h 'radiance stopped' "$OUT/guard.log")"
stopserver() { $SSH 'docker rm -f wmr-serve' >/dev/null 2>&1 < /dev/null; }
cleanup() {
  stopserver; exec 9>&-; wait $GUARD; rm -f "$FIFO"
  say "guard: $(grep -h 'stopping radiance\|radiance answers\|DID NOT' "$OUT/guard.log" | tr '\n' '|')"
  echo "radiance down: $(grep -h 'stopping radiance' "$OUT/guard.log" | sed 's/.*radiance //') -> $(grep -h 'radiance answers\|DID NOT' "$OUT/guard.log" | sed 's/.*\] //')  (fleetq job $FLEETQ_JOB)" >> "$OUT/downtime.log"
}
trap cleanup EXIT

D="docker run --rm -v \$HOME/intarsiaml-tp2-$TREE:/work -w /work $IMG /work/tp2-intarsia"
$SSH "$D --version" < /dev/null > "$OUT/versions.txt" 2>&1
say "binary: $(head -c 300 "$OUT/versions.txt" | tr '\n' ' ')"

# wmr2: PLAN=1 runs `plan show` (card 0; plan show has no --max-batch) for knob off / on / fp8 / bf16 first, to OUT/plan-<arm>.txt
if [[ ${PLAN:-0} == 1 ]]; then
  for pa in off on fp8 bf16; do
    case $pa in off) pe="";; on) pe="-e INTARSIA_HIP_WMMA_ROWS=1 -e INTARSIA_HIP_WMMA_TABLE=fp8a8=all,bf16=all";;
      fp8) pe="-e INTARSIA_HIP_WMMA_ROWS=1 -e INTARSIA_HIP_WMMA_TABLE=fp8a8=all";; bf16) pe="-e INTARSIA_HIP_WMMA_ROWS=1 -e INTARSIA_HIP_WMMA_TABLE=bf16=all";; esac
    $SSH "docker run --rm --device /dev/kfd --device /dev/dri --group-add \$(stat -c %g /dev/kfd) --group-add \$(stat -c %g /dev/dri/renderD128) --group-add video --ipc host \
      --security-opt seccomp=unconfined --user \$(id -u):\$(id -g) -e HOME=/tmp -e HIP_VISIBLE_DEVICES=0 $pe \
      -v \$HOME/intarsiaml-tp2-$TREE:/work -v $MODELS_DIR:/models:ro -w /work $IMG /work/tp2-intarsia plan show --backend hip --model $MODEL --max-seq 8192" \
      < /dev/null > "$OUT/plan-$pa.txt" 2> "$OUT/plan-$pa.err"
    say "plan show $pa exit=$? $(wc -c < "$OUT/plan-$pa.txt") B"
  done
fi

# serve <arm> <maxbatch> <tag>
serve() {
  local arm=$1 mb=$2 tag=$3 gpus=0 args="" envs="" tp=${1%%-*} knob=${1##*-}
  [[ $mb -gt 1 ]] && args="--max-batch $mb"
  [[ $tp == tp2 ]] && { gpus=0,1; args="$args --tp 2 --devices 0,1"; }
  [[ $knob == on ]] && envs="-e INTARSIA_HIP_WMMA_ROWS=1 -e INTARSIA_HIP_WMMA_TABLE=fp8a8=all,bf16=all"
  [[ $knob == fp8 ]] && envs="-e INTARSIA_HIP_WMMA_ROWS=1 -e INTARSIA_HIP_WMMA_TABLE=fp8a8=all"   # wmr2: only the fp8 half of the pair
  [[ $knob == bf16 ]] && envs="-e INTARSIA_HIP_WMMA_ROWS=1 -e INTARSIA_HIP_WMMA_TABLE=bf16=all"   # wmr2: only the bf16 half of the pair
  [[ $knob == bare ]] && envs="-e INTARSIA_HIP_WMMA_ROWS=1"   # the flag alone: the default table (W8A16 only; on the 35B it moves the three fp8 decode folds)
  RECIPE="${envs//-e /} HIP_VISIBLE_DEVICES=$gpus GPU_MAX_HW_QUEUES=1 intarsia serve --backend hip --model $MODEL --served-model-name $NAME $args --spec off --profile-every 0 --mem-budget 29 --max-seq $MAXSEQ --no-supervisor --host 0.0.0.0 --port $PORT (tree $TREE, docker $IMG, arm $arm)"
  stopserver
  $SSH "docker run -d --name wmr-serve --network host --device /dev/kfd --device /dev/dri \
    --group-add \$(stat -c %g /dev/kfd) --group-add \$(stat -c %g /dev/dri/renderD128) --group-add video --ipc host \
    --security-opt seccomp=unconfined --user \$(id -u):\$(id -g) -e HOME=/tmp -e GPU_MAX_HW_QUEUES=1 -e HIP_VISIBLE_DEVICES=$gpus $envs \
    -v \$HOME/intarsiaml-tp2-$TREE:/work -v $MODELS_DIR:/models:ro -w /work $IMG /work/tp2-intarsia serve --backend hip --model $MODEL \
    --served-model-name $NAME $args --spec off --profile-every 0 --mem-budget 29 --max-seq $MAXSEQ --no-supervisor --host 0.0.0.0 --port $PORT" \
    < /dev/null > "$OUT/docker-$tag.id" 2>&1
  for i in $(seq 1 100); do
    curl -sf -m 3 "http://tatooine:$PORT/health" > /dev/null 2>&1 && return 0
    [[ $($SSH "docker inspect -f '{{.State.Running}}' wmr-serve" < /dev/null 2>/dev/null) == true ]] || break
    sleep 3
  done
  $SSH 'docker logs --tail 60 wmr-serve' > "$OUT/serve-fail-$tag.log" 2>&1 < /dev/null
  return 1
}

for ph in $PHASES; do
  IFS=: read -r PNAME MB PSIZES GEN CONC PR PARMS PIDENT <<<"$ph"
  parms=${PARMS//,/ }
  for r in $(seq 1 "$PR"); do
    order="$parms"
    (( r % 2 == 0 )) && order=$(echo $parms | awk '{for (i=NF;i>=1;i--) printf "%s ", $i}')
    for arm in $order; do
      tag="$PNAME-$arm-r$r"
      if (( $(left) < 300 )); then say "SKIPPED $tag (time left $(left)s)"; continue; fi
      say "$tag: starting server"
      if ! serve "$arm" "$MB" "$tag"; then say "$tag: server did not come up, aborting the window (see serve-fail-$tag.log)"; stopserver; exit 1; fi
      say "$tag: server up ($(( $(date +%s) - T0 )) s into the window), localeval (sizes $PSIZES, gen $GEN, streams ${CONC:-1})"
      ( cd /opt/code/deepeval && uv run localeval speed --url "http://tatooine:$PORT/v1" --model "$NAME" --label "wmr-$tag" \
          --sizes "$PSIZES" --gen "$GEN" --reps 1 ${CONC:+--concurrency "$CONC"} --thinking off --wait-idle 60 --recipe "$RECIPE" ) \
        > "$OUT/localeval-$tag.log" 2>&1 < /dev/null
      say "$tag: localeval exit $? $(grep -aE 'run_id|results/' "$OUT/localeval-$tag.log" | tail -2 | tr '\n' '|')"
      if [[ ${PIDENT:-0} == 1 && $r == 1 ]]; then
        python3 -I "$HERE/ident.py" --url "http://tatooine:$PORT/v1" --model "$NAME" --out "$OUT/ident-$tag.json" > "$OUT/ident-$tag.log" 2>&1 < /dev/null
        say "$tag: ident exit $? $(tail -3 "$OUT/ident-$tag.log" | tr '\n' '|')"
      fi
      $SSH 'docker logs wmr-serve 2>&1 | grep -aiE "stamp|head=|code-tree|batch|graph|spec|mtp|listening|tensor-parallel|wmma" | tail -12' < /dev/null > "$OUT/serve-$tag.log" 2>&1
      stopserver
    done
  done
done
say "done, time left $(left)s"

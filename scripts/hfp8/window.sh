#!/usr/bin/env bash
# Lane hfp8: end-to-end A/B of the one-pass fp8 rows kernel on Qwen3.6-35B-A3B NVFP4 (HIP, R9700), driven FROM THE MAC under one fleetq job
# that holds both R9700 cards and takes radiance down (derived from scripts/r9t/window.sh, same guard):
#   ARMS="tp1:base tp1:new" ROUNDS=3 fleetq run --host tatooine-r9700,tatooine-r9700-2 --takes-down radiance --quiet gpu --load cpu \
#     --expect 20 --mem 30 --label "HFP8 e2e tp1 streams" -- scripts/hfp8/window.sh
# ARMS: space separated <tp1|tp2>:<base|new>; base = tree hfp8base2 (a real build of main 05451fe66, HGV included), new = tree hfp8new2 (this branch rebased on it). Each
# round starts a fresh server per arm (`serve --spec off --max-batch $MB`, thinking off), arms interleaved, the order reversed on even rounds.
# MODE=speed (default): `localeval speed --sizes $SIZES --gen $GEN --concurrency $CONC` per server; MODE=greedy: scripts/hfp8/greedy.py
# (16 concurrent different prompts at temperature 0, outputs to OUT/greedy-<tag>.jsonl). MB = --max-batch (default 16; 1 = flag omitted).
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
ARMS=${ARMS:?ARMS}; ROUNDS=${ROUNDS:-3}; MB=${MB:-16}; SIZES=${SIZES:-1024}; GEN=${GEN:-512}; CONC=${CONC-4,8,16}; MODE=${MODE:-speed}; CAP=${CAP:-1150}
GGEN=${GGEN:-192}; GSTREAMS=${GSTREAMS:-16}
MODEL=/models/Qwen3.6-35B-A3B-NVFP4; NAME=qwen36-35b; MODELS_DIR=/data/models/intarsia
MAXSEQ=${MAXSEQ:-18432}; PORT=8090
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
OUT=${OUT:?OUT dir}; mkdir -p "$OUT"
T0=$(date +%s); left() { echo $(( CAP - ($(date +%s) - T0) )); }
say() { echo "$(date +%T) $*" | tee -a "$OUT/steps.log"; }
SSH="ssh -o BatchMode=yes tatooine"

FIFO="$OUT/guard.fifo"; mkfifo "$FIFO"
$SSH 'mkdir -p ~/hprof' < /dev/null; scp -q "$REPO/scripts/hprof/guard.sh" tatooine:hprof/guard.sh
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

treeof() { case $1 in base) echo hfp8base2;; new) echo hfp8new2;; *) echo "$1";; esac; }
for v in base new; do
  $SSH "docker run --rm -v \$HOME/intarsiaml-tp2-$(treeof $v):/work -w /work $IMG /work/tp2-intarsia --version" < /dev/null > "$OUT/versions-$v.txt" 2>&1
  say "binary $v: $(head -c 250 "$OUT/versions-$v.txt" | tr '\n' ' ')"
done

# serve <tp1|tp2> <base|new> <tag>
serve() {
  local arm=$1 var=$2 tag=$3 gpus=0 args="" tree
  tree=$(treeof "$var")
  [[ $MB -gt 1 ]] && args="--max-batch $MB"
  [[ $arm == tp2 ]] && { gpus=0,1; args="$args --tp 2 --devices 0,1"; }
  RECIPE="HIP_VISIBLE_DEVICES=$gpus GPU_MAX_HW_QUEUES=1 intarsia serve --backend hip --model $MODEL --served-model-name $NAME $args --spec off --profile-every 0 --mem-budget 29 --max-seq $MAXSEQ --no-supervisor --host 0.0.0.0 --port $PORT (tree $tree = $var, docker $IMG)"
  stopserver
  $SSH "docker run -d --name hprof-serve --network host --device /dev/kfd --device /dev/dri \
    --group-add \$(stat -c %g /dev/kfd) --group-add \$(stat -c %g /dev/dri/renderD128) --group-add video --ipc host \
    --security-opt seccomp=unconfined --user \$(id -u):\$(id -g) -e HOME=/tmp -e GPU_MAX_HW_QUEUES=1 -e HIP_VISIBLE_DEVICES=$gpus \
    -v \$HOME/intarsiaml-tp2-$tree:/work -v $MODELS_DIR:/models:ro -w /work $IMG /work/tp2-intarsia serve --backend hip --model $MODEL \
    --served-model-name $NAME $args --spec off --profile-every 0 --mem-budget 29 --max-seq $MAXSEQ --no-supervisor --host 0.0.0.0 --port $PORT" \
    < /dev/null > "$OUT/docker-$tag.id" 2>&1
  for i in $(seq 1 100); do
    curl -sf -m 3 "http://tatooine:$PORT/health" > /dev/null 2>&1 && return 0
    [[ $($SSH "docker inspect -f '{{.State.Running}}' hprof-serve" < /dev/null 2>/dev/null) == true ]] || break
    sleep 3
  done
  $SSH 'docker logs --tail 60 hprof-serve' > "$OUT/serve-fail-$tag.log" 2>&1 < /dev/null
  return 1
}

for r in $(seq 1 "$ROUNDS"); do
  order="$ARMS"
  (( r % 2 == 0 )) && order=$(echo $ARMS | awk '{for (i=NF;i>=1;i--) printf "%s ", $i}')
  for arm in $order; do
    tp=${arm%%:*}; var=${arm##*:}; tag="$MODE-$tp-$var-r$r"
    if (( $(left) < 240 )); then say "SKIPPED $tag (time left $(left)s)"; continue; fi
    say "$tag: starting server"
    if ! serve "$tp" "$var" "$tag"; then say "$tag: server did not come up, aborting the window (see serve-fail-$tag.log)"; stopserver; exit 1; fi
    if [[ $MODE == greedy ]]; then
      say "$tag: server up, greedy ($GSTREAMS streams, gen $GGEN)"
      python3 "$HERE/greedy.py" --url "http://tatooine:$PORT/v1" --model "$NAME" --docs "$REPO/docs" --streams "$GSTREAMS" --gen "$GGEN" --out "$OUT/greedy-$tag.jsonl" > "$OUT/greedy-$tag.log" 2>&1 < /dev/null
      say "$tag: greedy exit $? $(tail -c 200 "$OUT/greedy-$tag.log" | tr '\n' ' ')"
    else
      say "$tag: server up, localeval (sizes $SIZES, gen $GEN, streams 1,$CONC)"
      ( cd /opt/code/deepeval && uv run localeval speed --url "http://tatooine:$PORT/v1" --model "$NAME" --label "hfp8-$tag" \
          --sizes "$SIZES" --gen "$GEN" --reps 1 ${CONC:+--concurrency "$CONC"} --thinking off --wait-idle 60 --recipe "$RECIPE" ) \
        > "$OUT/localeval-$tag.log" 2>&1 < /dev/null
      say "$tag: localeval exit $? $(grep -aE 'run_id|results/' "$OUT/localeval-$tag.log" | tail -2 | tr '\n' '|')"
    fi
    $SSH 'docker logs hprof-serve 2>&1 | grep -aiE "stamp|head=|code-tree|batch|graph|listening|tensor-parallel" | tail -6' < /dev/null > "$OUT/serve-$tag.log" 2>&1
    stopserver
  done
done
say "done, time left $(left)s"

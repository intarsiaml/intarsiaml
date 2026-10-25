#!/usr/bin/env bash
# Lane tp2: speed window, driven FROM THE MAC (localeval lives here), under one fleetq job that holds both R9700s and takes
# radiance down:
#   fleetq run --host tatooine-r9700,tatooine-r9700-2 --takes-down radiance --expect 45 --mem 30 --quiet gpu,cpu \
#     --label "tp2 speed" -- env TREE=batch scripts/tp2-hip/speed-window.sh
# Arms (same binary, rounds interleaved ABAB..): tp1 = one card, `serve --max-batch 16`; tp2 = both cards,
# `serve --tp 2 --devices 0,1 --max-batch 16` (HIP defaults: graphs, fusion, vocab-parallel head). Each arm: a fresh server,
# `localeval speed` (prefill sweep 1k/4k/16k, decode at 1 stream, aggregate decode at 4 and 16 streams, gen 512), then the
# server goes away. localeval runs inside this job's slot (FLEETQ_JOB is inherited), so it does not queue behind it.
# Radiance is restored by a guard session on tatooine when this script ends or dies, and the script waits for it.
# Variables: TREE (default batch), ROUNDS (3), SIZES, CONC, GEN, ARMS ("tp1 tp2"), EXTRA_TP2 (env for the tp2 arm, e.g.
# "INTARSIA_TP_VOCAB=0"), TAGP (label prefix), CAP (seconds, default 2400: arms that no longer fit are skipped).
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
TREE=${TREE:-batch}; ROUNDS=${ROUNDS:-3}; GEN=${GEN:-512}; SIZES=${SIZES:-1024,4096,16384}; CONC=${CONC-4,16}
ARMS=${ARMS:-"tp1 tp2"}; TAGP=${TAGP:-tp2speed}; CAP=${CAP:-2400}
MODEL=${MODEL:-/models/Qwen3.6-35B-A3B-NVFP4}; NAME=${NAME:-qwen36-35b}; MODELS_DIR=${MODELS_DIR:-/data/models/intarsia}
MAXB=${MAXB:-16}; MAXSEQ=${MAXSEQ:-18432}; PORT=8090
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
OUT=${OUT:-$REPO/docs/results/2026-10-25-tp2-hip/raw/speed-$(date +%H%M)}; mkdir -p "$OUT"
T0=$(date +%s); left() { echo $(( CAP - ($(date +%s) - T0) )); }
say() { echo "$(date +%T) $*" | tee -a "$OUT/steps.log"; }
SSH="ssh -o BatchMode=yes tatooine"

# guard: stops radiance now, restores it when its stdin closes (we exit or die) and reports when it answers
FIFO=$(mktemp -u /tmp/tp2guard.XXXXXX); mkfifo "$FIFO"
scp -q "$HERE/guard.sh" tatooine:tp2/guard.sh
$SSH 'bash ~/tp2/guard.sh' > "$OUT/guard.log" 2>&1 < "$FIFO" &
GUARD=$!
exec 9> "$FIFO"
for i in $(seq 1 30); do grep -q "radiance stopped" "$OUT/guard.log" 2>/dev/null && break; sleep 1; done
say "guard: $(grep -h 'radiance stopped' "$OUT/guard.log")"
cleanup() { $SSH 'docker rm -f tp2-speed' >/dev/null 2>&1 < /dev/null; exec 9>&-; wait $GUARD; rm -f "$FIFO"; say "guard: $(grep -h 'radiance answers\|DID NOT' "$OUT/guard.log")"; }
trap cleanup EXIT

serve() { # serve <arm>
  local arm=$1 envs="" args="" gpus=0; [[ $MAXB -gt 1 ]] && args="--max-batch $MAXB"
  if [[ $arm == tp2 ]]; then gpus=0,1; args="$args --tp 2 --devices 0,1"; envs="${EXTRA_TP2:-}"; fi
  [[ $arm == tp1 ]] && envs="${EXTRA_TP1:-}"
  RECIPE="HIP_VISIBLE_DEVICES=$gpus ${envs:+$envs }tp2-intarsia serve --backend hip --model $MODEL --served-model-name $NAME-$arm $args --profile-every 0 --mem-budget 29 --max-seq $MAXSEQ --no-supervisor --host 0.0.0.0 --port $PORT (tree $TREE, docker $IMG)"
  local envflags=""; for e in $envs; do envflags="$envflags -e $e"; done
  $SSH "docker rm -f tp2-speed >/dev/null 2>&1; docker run -d --name tp2-speed --network host --device /dev/kfd --device /dev/dri \
    --group-add \$(stat -c %g /dev/kfd) --group-add \$(stat -c %g /dev/dri/renderD128) --group-add video --ipc host \
    --security-opt seccomp=unconfined --user \$(id -u):\$(id -g) -e HOME=/tmp -e GPU_MAX_HW_QUEUES=1 -e HIP_VISIBLE_DEVICES=$gpus $envflags \
    -v \$HOME/intarsiaml-tp2-$TREE:/work -v $MODELS_DIR:/models:ro -w /work $IMG /work/tp2-intarsia serve --backend hip --model $MODEL \
    --served-model-name $NAME-$arm $args --profile-every 0 --mem-budget 29 --max-seq $MAXSEQ --no-supervisor --host 0.0.0.0 --port $PORT" < /dev/null > "$OUT/docker-$arm.id" 2>&1
  for i in $(seq 1 100); do
    curl -sf -m 3 "http://tatooine:$PORT/health" > /dev/null 2>&1 && return 0
    sleep 3
  done
  $SSH 'docker logs --tail 40 tp2-speed' > "$OUT/serve-fail-$arm.log" 2>&1 < /dev/null
  return 1
}

for r in $(seq 1 "$ROUNDS"); do
  order="$ARMS"
  (( r % 2 == 0 )) && order=$(echo $ARMS | tr ' ' '\n' | tac | tr '\n' ' ')
  for arm in $order; do
    if (( $(left) < 420 )); then say "SKIPPED round $r arm $arm (time left $(left)s)"; continue; fi
    say "round $r arm $arm: starting server"
    if ! serve "$arm"; then say "round $r arm $arm: server did not come up"; $SSH 'docker rm -f tp2-speed' >/dev/null 2>&1 < /dev/null; continue; fi
    say "round $r arm $arm: server up, localeval"
    ( cd /opt/code/deepeval && uv run localeval speed --url "http://tatooine:$PORT/v1" --model "$NAME-$arm" --label "$TAGP-$arm-r$r" \
        --sizes "$SIZES" --gen "$GEN" --reps 1 ${CONC:+--concurrency "$CONC"} --thinking off --wait-idle 60 --recipe "$RECIPE" ) \
      > "$OUT/localeval-$arm-r$r.log" 2>&1 < /dev/null
    say "round $r arm $arm: localeval exit $? $(grep -aE 'run_id|results/' "$OUT/localeval-$arm-r$r.log" | tail -2 | tr '\n' '|')"
    $SSH 'docker logs tp2-speed 2>&1 | grep -aE "tensor-parallel|decode graph|batched|graph" | tail -6' < /dev/null > "$OUT/serve-$arm-r$r.log" 2>&1
    $SSH 'docker rm -f tp2-speed' >/dev/null 2>&1 < /dev/null
  done
done
say "done, time left $(left)s"

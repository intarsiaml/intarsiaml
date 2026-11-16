#!/usr/bin/env bash
# Lane mtp-snap-gpu (ON THE MAC, sourced by exact-window.sh / speed-window.sh): the radiance guard, `serve <pr|main> <tag>` and `drop`.
# Needs: FLEETQ_JOB (run under `fleetq run --takes-down radiance`), OUT (raw log dir), CAP (seconds). Server flags: EXTRA, environment: SENV="K=V K=V".
# A server runs in docker container mtpsnap-gpu-serve from the tree's binary /target/intarsia-<pr|main> (scripts/mtp-snap-gpu/build.sh), one R9700,
# `serve --backend hip --spec mtp` (single-sequence engine: no --max-batch), the flags of scripts/gdn-chunk/speed-window.sh.
: "${FLEETQ_JOB:?run under fleetq run}"
MODEL=${MODEL:-/models/Qwen3.6-35B-A3B-NVFP4}; NAME=${NAME:-qwen36-35b}; MODELS_DIR=${MODELS_DIR:-/data/models/intarsia}
MAXSEQ=${MAXSEQ:-18432}; PORT=8090; VOL=${VOL:-intarsia-cargo-target-mtpsnap-gpu}
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
CNAME=mtpsnap-gpu-serve
SSH="ssh -o BatchMode=yes tatooine"
mkdir -p "$OUT"
T0=$(date +%s); left() { echo $(( CAP - ($(date +%s) - T0) )); }
say() { echo "$(date +%T) $*" | tee -a "$OUT/steps.log"; }

FIFO=$(mktemp -u "${TMPDIR:-/tmp}/mtpsnapguard.XXXXXX"); mkfifo "$FIFO"
$SSH 'bash ~/intarsiaml-mtpsnap-gpu/scripts/mtp-snap-gpu/guard.sh' > "$OUT/guard.log" 2>&1 < "$FIFO" &
GUARD=$!
exec 9> "$FIFO"
for i in $(seq 1 30); do grep -q "radiance stopped" "$OUT/guard.log" 2>/dev/null && break; sleep 1; done
say "guard: $(grep -h 'radiance stopped' "$OUT/guard.log")"
CURTAG=""
drop() {
  [[ -n "$CURTAG" ]] && $SSH "docker logs $CNAME" > "$OUT/serve-$CURTAG.log" 2>&1 < /dev/null
  CURTAG=""
  $SSH "docker rm -f $CNAME" >/dev/null 2>&1 < /dev/null
}
cleanup() { drop; exec 9>&-; wait $GUARD; rm -f "$FIFO"; say "guard: $(grep -h 'radiance answers\|DID NOT' "$OUT/guard.log")"; }
trap cleanup EXIT

serve() { # serve <pr|main> <tag>
  local bin=$1 tag=$2 envflags="" e
  for e in ${SENV:-}; do envflags="$envflags -e $e"; done
  drop
  CURTAG=$tag
  say "serve $tag: bin=$bin extra='${EXTRA:-}' env='${SENV:-}'"
  $SSH "docker run -d --name $CNAME --network host --device /dev/kfd --device /dev/dri \
    --group-add \$(stat -c %g /dev/kfd) --group-add \$(stat -c %g /dev/dri/renderD128) --group-add video --ipc host \
    --security-opt seccomp=unconfined --user \$(id -u):\$(id -g) -e HOME=/tmp -e GPU_MAX_HW_QUEUES=1 -e HIP_VISIBLE_DEVICES=0 $envflags \
    -v \$HOME/intarsiaml-mtpsnap-gpu:/work -v $MODELS_DIR:/models:ro -v $VOL:/target -w /work $IMG /target/intarsia-$bin serve --backend hip --model $MODEL \
    --served-model-name $NAME --spec mtp --profile-every 0 --mem-budget 29 --max-seq $MAXSEQ --no-supervisor --host 0.0.0.0 --port $PORT ${EXTRA:-}" < /dev/null > "$OUT/docker-$tag.id" 2>&1
  local i
  for i in $(seq 1 100); do
    if curl -sf -m 3 "http://tatooine:$PORT/health" > /dev/null 2>&1; then say "serve $tag: up after $((i * 3)) s"; return 0; fi
    sleep 3
  done
  say "serve $tag: DID NOT COME UP"
  return 1
}

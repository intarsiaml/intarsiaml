#!/usr/bin/env bash
# Lane wht6-8b, gate 3: tp2 prefill and decode speed, the lossy wht6 / wht8 all-reduce wires vs the exact wire. Driven FROM THE MAC (localeval
# lives here), under one fleetq job that holds both R9700s and takes radiance down (the shape of scripts/gdn-chunk/speed-window.sh):
#   fleetq run --host tatooine --takes-down radiance --expect 40 --mem 30 --quiet gpu,cpu \
#     --label "intarsiaml hip: wht6-8b speed" -- scripts/wht6-8b/speed-window.sh
# Arms, the same binary, rounds interleaved (order rotated each round): tp2-off, tp2-wht6, tp2-wht8 = both cards,
# `serve --tp 2 --devices 0,1`; tp2-wht* set INTARSIA_HIP_AR_WIRE=wht* (plus ON_EXTRA, e.g. INTARSIA_HIP_AR_WIRE_MIN_KB=1024). Each arm: a fresh server, `localeval
# speed` (the prefill sweep and GEN decoded tokens per size), then the server goes away. localeval runs inside this job's slot.
# PHASES: "name:sizes:rounds:arm,arm" separated by spaces. CAP: seconds; arms that no longer fit are skipped.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
PHASES=${PHASES:-"tp2:1024,4096,16384:3:tp2-off,tp2-wht6,tp2-wht8"}
ROUND0=${ROUND0:-0}   # rounds already done by an earlier window of the same measurement (keeps the interleave order and the log names)
GEN=${GEN:-128}; CAP=${CAP:-2100}; TAGP=${TAGP:-wht6ar}
MODEL=${MODEL:-/models/Qwen3.6-35B-A3B-NVFP4}; NAME=${NAME:-qwen36-35b}; MODELS_DIR=${MODELS_DIR:-/data/models/intarsia}
MAXSEQ=${MAXSEQ:-18432}; PORT=8090; VOL=${VOL:-intarsia-cargo-target-wht6-8b}
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
OUT=${OUT:-$REPO/docs/results/2026-11-16-hip-wht8-ar/raw/speed-$(date +%H%M)}; mkdir -p "$OUT"
T0=$(date +%s); left() { echo $(( CAP - ($(date +%s) - T0) )); }
say() { echo "$(date +%T) $*" | tee -a "$OUT/steps.log"; }
SSH="ssh -o BatchMode=yes tatooine"
CNAME=wht6-8b-serve

# guard: stops radiance now, restores it when its stdin closes (we exit or die) and reports when it answers
FIFO="$OUT/guard.fifo"; rm -f "$FIFO"; mkfifo "$FIFO"
$SSH 'bash ~/intarsiaml-wht6-8b/scripts/wht6-8b/guard.sh' > "$OUT/guard.log" 2>&1 < "$FIFO" &
GUARD=$!
exec 9> "$FIFO"
for i in $(seq 1 30); do grep -q "radiance stopped" "$OUT/guard.log" 2>/dev/null && break; sleep 1; done
say "guard: $(grep -h 'radiance stopped' "$OUT/guard.log")"
drop() { $SSH "docker rm -f $CNAME" >/dev/null 2>&1 < /dev/null; }
cleanup() { drop; exec 9>&-; wait $GUARD; rm -f "$FIFO"; say "guard: $(grep -h 'radiance answers\|DID NOT' "$OUT/guard.log")"; }
trap cleanup EXIT

serve() { # serve <arm>
  local arm=$1 envs="" args="" gpus=0
  [[ $arm == tp2-* ]] && { gpus=0,1; args="--tp 2 --devices 0,1"; }
  [[ $arm == *-wht* ]] && envs="INTARSIA_HIP_AR_WIRE=${arm##*-}${ON_EXTRA:+ $ON_EXTRA}"
  RECIPE="HIP_VISIBLE_DEVICES=$gpus ${envs:+$envs }intarsia serve --backend hip --model $MODEL --served-model-name $NAME-$arm $args --profile-every 0 --mem-budget 29 --max-seq $MAXSEQ --no-supervisor --host 0.0.0.0 --port $PORT (tree wht6-8b, docker $IMG)"
  local envflags=""; for e in $envs; do envflags="$envflags -e $e"; done
  drop
  $SSH "docker run -d --name $CNAME --network host --device /dev/kfd --device /dev/dri \
    --group-add \$(stat -c %g /dev/kfd) --group-add \$(stat -c %g /dev/dri/renderD128) --group-add video --ipc host \
    --security-opt seccomp=unconfined --user \$(id -u):\$(id -g) -e HOME=/tmp -e GPU_MAX_HW_QUEUES=1 -e HIP_VISIBLE_DEVICES=$gpus $envflags \
    -v \$HOME/intarsiaml-wht6-8b:/work -v $MODELS_DIR:/models:ro -v $VOL:/target -w /work $IMG /target/release/intarsia serve --backend hip --model $MODEL \
    --served-model-name $NAME-$arm $args --profile-every 0 --mem-budget 29 --max-seq $MAXSEQ --no-supervisor --host 0.0.0.0 --port $PORT" < /dev/null > "$OUT/docker-$arm.id" 2>&1
  for i in $(seq 1 100); do
    curl -sf -m 3 "http://tatooine:$PORT/health" > /dev/null 2>&1 && return 0
    sleep 3
  done
  $SSH "docker logs --tail 40 $CNAME" > "$OUT/serve-fail-$arm.log" 2>&1 < /dev/null
  return 1
}

for ph in $PHASES; do
  IFS=: read -r PNAME SIZES ROUNDS PARMS <<<"$ph"
  parms=${PARMS//,/ }
  for r in $(seq $((ROUND0 + 1)) $((ROUND0 + ROUNDS))); do
    # rotate the order each round (r1 off wht6 wht8, r2 wht6 wht8 off, r3 wht8 off wht6): every arm takes every position once per 3 rounds
    order=$(echo $parms | awk -v k=$(( (r - 1) % $(echo $parms | wc -w) )) '{for (i = 1; i <= NF; i++) printf "%s ", $(((i - 1 + k) % NF + 1))}')
    for arm in $order; do
      if (( $(left) < 300 )); then say "SKIPPED phase $PNAME round $r arm $arm (time left $(left)s)"; continue; fi
      say "phase $PNAME round $r arm $arm: starting server"
      if ! serve "$arm"; then say "phase $PNAME round $r arm $arm: server did not come up"; drop; continue; fi
      say "phase $PNAME round $r arm $arm: server up, localeval"
      ( cd /opt/code/deepeval && uv run localeval speed --url "http://tatooine:$PORT/v1" --model "$NAME-$arm" --label "$TAGP-$PNAME-$arm-r$r" \
          --sizes "$SIZES" --gen "$GEN" --reps "${REPS:-3}" --thinking off --wait-idle 60 --recipe "$RECIPE" ) \
        > "$OUT/localeval-$PNAME-$arm-r$r.log" 2>&1 < /dev/null
      say "phase $PNAME round $r arm $arm: localeval exit $? $(grep -aE 'run_id|results/' "$OUT/localeval-$PNAME-$arm-r$r.log" | tail -2 | tr '\n' '|')"
      $SSH "docker logs $CNAME 2>&1 | grep -aE 'prefill|tensor-parallel|LOSSY' | tail -12" < /dev/null > "$OUT/serve-$PNAME-$arm-r$r.log" 2>&1
      drop
    done
  done
done
say "done, time left $(left)s"

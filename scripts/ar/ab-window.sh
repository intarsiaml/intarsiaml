#!/usr/bin/env bash
# Lane ar: INTARSIA_HIP_AR_DRAIN speed A/B at tp=2, driven FROM THE MAC (localeval lives here), one fleetq job holding both R9700s and
# taking radiance down:
#   fleetq acquire --host tatooine-r9700,tatooine-r9700-2 --takes-down radiance --quiet gpu,cpu --expect 70 --mem 30 --wait --label "ar: ..."
#   FLEETQ_JOB=<id> W=a1 PHASES="speed27:4 speed35:2" scripts/ar/ab-window.sh
# Arms (same binary, rounds interleaved with a rotating order): d3 = INTARSIA_HIP_AR_DRAIN=3 (default), d1, d5. Each arm-round: a fresh
# `serve --tp 2 --devices 0,1 --max-batch 1`, one localeval call, the server goes away. Phases:
#   speed27 : 27B MXFP4, --spec off, `localeval speed` 1k/4k/16k, gen 128, 3 reps
#   cats    : 27B MXFP4, --spec mtp --spec-k 2, `localeval quality --judge none` over scripts/dfh/cats/rcp-*.jsonl, 512 forced tokens
#   speed35 : Qwen3.6-35B-A3B NVFP4 (INTARSIA_TP_FUSION default on at tp>1), `localeval speed` 1k/4k, gen 256, 3 reps
# PHASES "name:rounds" (rounds <= 4 here). Radiance is restored by guard.sh on tatooine when this script ends or dies.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}" "${W:?W=<window>}"
PHASES=${PHASES:-"speed27:4"}; CAP=${CAP:-4500}; PORT=8097; R0=${R0:-1}
M27=/models/radiance/hf-cache/hub/models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/5233554c5fa56afda40150556b95573c2d7d29c0
M35=/models/intarsia/Qwen3.6-35B-A3B-NVFP4
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
ORD=("d1 d3 d7" "d7 d3 d1" "d3 d7 d1" "d1 d7 d3")
# arms d<N> = INTARSIA_HIP_AR_DRAIN=<N> (d7 = lane ars LL handshake; every arm runs the B binary ars-intarsia-b, so only the knob differs)
# ARMS="d3 d1": two arms, order alternating per round
[[ -n ${ARMS:-} ]] && { rev=$(echo $ARMS | awk '{for (i=NF;i>=1;i--) printf "%s ", $i}'); ORD=("$ARMS" "$rev" "$ARMS" "$rev"); }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
OUT=$REPO/docs/results/2026-10-28-det-tp2-nondeterminism/raw/ars-$W; mkdir -p "$OUT"
T0=$(date +%s); left() { echo $(( CAP - ($(date +%s) - T0) )); }
say() { echo "$(date +%T) $*" | tee -a "$OUT/steps.log"; }
SSH="ssh -o BatchMode=yes tatooine"
$SSH 'mkdir -p ~/intarsiaml-ars/.ars' < /dev/null; scp -q "$HERE/guard.sh" tatooine:intarsiaml-ars/.ars/
FIFO=$(mktemp -u "$REPO/.worktrees/ars-guard.XXXXXX" 2>/dev/null || mktemp -u "$OUT/guard.XXXXXX"); mkfifo "$FIFO"
$SSH 'bash ~/intarsiaml-ars/.ars/guard.sh' > "$OUT/guard.log" 2>&1 < "$FIFO" &
GUARD=$!
exec 9> "$FIFO"
for i in $(seq 1 30); do grep -q "radiance stopped" "$OUT/guard.log" 2>/dev/null && break; sleep 1; done
say "guard: $(grep -h 'radiance stopped' "$OUT/guard.log")"
cleanup() { $SSH 'docker rm -f ars-serve' >/dev/null 2>&1 < /dev/null; exec 9>&-; wait $GUARD; rm -f "$FIFO"; say "guard: $(grep -h 'radiance answers\|DID NOT' "$OUT/guard.log")"
  start=$(grep -m1 'radiance stopped' "$OUT/guard.log" | grep -oE '[0-9]{2}:[0-9]{2}:[0-9]{2}'); back=$(grep -m1 'radiance answers' "$OUT/guard.log" | grep -oE '[0-9]{2}:[0-9]{2}:[0-9]{2}' | head -1)
  echo "$(date +%F) fleetq #$FLEETQ_JOB ars-$W radiance stopped $start answers ${back:-NEVER} (guard.log in raw/ars-$W)" >> "$OUT/../downtime.log"; }
trap cleanup EXIT

serve() { # serve <phase> <arm>
  local ph=$1 arm=$2 model name spec maxseq drain=${2#d}
  case $ph in
    speed27) model=$M27; name=qwen38-27b; spec="--spec off"; maxseq=20480 ;;
    cats)    model=$M27; name=qwen38-27b; spec="--spec mtp --spec-k 2"; maxseq=8192 ;;
    speed35) model=$M35; name=qwen36-35b; spec=""; maxseq=8192 ;;
  esac
  SERVED=$name-$arm
  RECIPE="HIP_VISIBLE_DEVICES=0,1 INTARSIA_HIP_AR_DRAIN=$drain ars-intarsia serve --backend hip --model $model $spec --tp 2 --devices 0,1 --max-batch 1 --profile-every 0 --mem-budget 29 --max-seq $maxseq (tree ~/intarsiaml-ars, docker $IMG, lane ar)"
  $SSH "docker rm -f ars-serve >/dev/null 2>&1; docker run -d --name ars-serve --network host --device /dev/kfd --device /dev/dri \
    --group-add \$(stat -c %g /dev/kfd) --group-add \$(stat -c %g /dev/dri/renderD128) --group-add video --ipc host \
    --security-opt seccomp=unconfined --user \$(id -u):\$(id -g) -e HOME=/tmp -e GPU_MAX_HW_QUEUES=1 -e HIP_VISIBLE_DEVICES=0,1 -e INTARSIA_HIP_AR_DRAIN=$drain \
    -v \$HOME/intarsiaml-ars:/work -v /data/models:/models:ro -w /work $IMG /work/${BIN:-ars-intarsia-b} serve --backend hip --model $model \
    --served-model-name $SERVED $spec --tp 2 --devices 0,1 --max-batch 1 --profile-every 0 --mem-budget 29 --max-seq $maxseq --no-supervisor --host 0.0.0.0 --port $PORT" \
    < /dev/null > "$OUT/docker-$ph-$arm.id" 2>&1
  for i in $(seq 1 100); do curl -sf -m 3 "http://tatooine:$PORT/health" > /dev/null 2>&1 && return 0; sleep 3; done
  $SSH 'docker logs --tail 40 ars-serve' > "$OUT/serve-fail-$ph-$arm.log" 2>&1 < /dev/null; return 1
}

if [[ -n ${PRESTAGES:-} ]]; then
  # stages-soak.sh stages (tests, rate, kt7, soak7, ids18 ...) inside this same radiance-down window, before the A/B phases
  mkdir -p "$OUT/pre"; scp -q "$HERE/stages-soak.sh" tatooine:intarsiaml-ars/.ars/
  say "prestages: $PRESTAGES"
  $SSH "rm -rf ~/intarsiaml-ars/.ars/out/$W; N=${N:-250} REFN=${REFN:-0} DEFN=${DEFN:-1} DEFE='${DEFE:-}' DEFBIN=${DEFBIN:-} STAGES='$PRESTAGES' timeout ${PRECAP:-3000} bash ~/intarsiaml-ars/.ars/stages-soak.sh ~/intarsiaml-ars/.ars/out/$W" < /dev/null 2>&1 | tee "$OUT/pre/stages.log"
  rsync -a "tatooine:intarsiaml-ars/.ars/out/$W/" "$OUT/pre/" 2>/dev/null
  say "prestages done, time left $(left)s"
fi

for ph in $PHASES; do
  PNAME=${ph%%:*}; PR=${ph#*:}
  if [[ $PNAME == speed35 && -n ${ARMS35:-} ]]; then # arms for the 35B phase only (e.g. a d3 reference beside d1 and d7)
    rev35=$(echo $ARMS35 | awk '{for (i=NF;i>=1;i--) printf "%s ", $i}'); ORD=("$ARMS35" "$rev35" "$ARMS35" "$rev35")
  elif [[ -n ${ARMS:-} ]]; then ORD=("$ARMS" "$rev" "$ARMS" "$rev"); fi
  for r in $(seq "$R0" $(( R0 + PR - 1 ))); do
    for arm in ${ORD[$(( r - 1 ))]}; do
      if (( $(left) < 480 )); then say "SKIPPED $PNAME round $r arm $arm (time left $(left)s)"; continue; fi
      say "$PNAME round $r arm $arm: starting server"
      if ! serve "$PNAME" "$arm"; then say "$PNAME round $r arm $arm: server did not come up"; $SSH 'docker rm -f ars-serve' >/dev/null 2>&1 < /dev/null; continue; fi
      printf '[targets.ar]\nurl = "http://tatooine:%s/v1"\nmodel = "%s"\n' "$PORT" "$SERVED" > "$OUT/targets-$PNAME-$arm.toml"
      case $PNAME in
        speed27|speed35)
          sizes=1024,4096,16384; gen=128; [[ $PNAME == speed35 ]] && { sizes=1024,4096; gen=256; }
          ( cd /opt/code/deepeval && LOCALEVAL_TARGETS="$OUT/targets-$PNAME-$arm.toml" uv run localeval speed ar --thinking off --sizes "$sizes" --gen "$gen" --reps 3 \
              --label "ars-$W-$PNAME-$arm-r$r" --recipe "$RECIPE" --wait-idle 60 ) > "$OUT/$PNAME-$arm-r$r.log" 2>&1 < /dev/null
          say "$PNAME round $r arm $arm: localeval exit $? $(grep -aoE 'results/[^ ]+' "$OUT/$PNAME-$arm-r$r.log" | tail -1)" ;;
        cats)
          printf 'extra_body = { ignore_eos = true, min_tokens = 512 }\n' >> "$OUT/targets-$PNAME-$arm.toml"
          for f in "$REPO"/scripts/dfh/cats/rcp-*.jsonl; do
            c=$(basename "$f" .jsonl)
            ( cd /opt/code/deepeval && LOCALEVAL_TARGETS="$OUT/targets-$PNAME-$arm.toml" uv run localeval quality ar --thinking off --label "ars-$W-$arm-$c-r$r" \
                --recipe "$RECIPE" --dataset "$f" --judge none --max-tokens 512 --wait-idle 60 ) > "$OUT/q-$arm-$c-r$r.log" 2>&1 < /dev/null
            say "$PNAME round $r arm $arm $c: localeval exit $? $(grep -aoE 'results/[^ ]+' "$OUT/q-$arm-$c-r$r.log" | tail -1)"
          done ;;
      esac
      $SSH 'docker rm -f ars-serve' >/dev/null 2>&1 < /dev/null
    done
  done
done
say "done, time left $(left)s"

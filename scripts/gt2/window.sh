#!/usr/bin/env bash
# Lane gt2 (docs/results/2026-11-07-spec-graph-tp2.md), from the Mac, ONE fleetq job holding both R9700s and taking radiance down (shape of scripts/ag1/cat-window.sh):
#   fleetq run --host tatooine-r9700,tatooine-r9700-2 --takes-down radiance --quiet gpu,cpu --expect 50 --label "gt2 35B tp2 mtp" -- env W=w1 scripts/gt2/window.sh
# Stages: PRE (default "ids": stages-ids.sh, gates + rule (a)), then speed arms (ARMSEQ rounds separated by ";", default "tp1m tp2off tp2m;tp2m tp2off tp1m"):
# a fresh `serve --max-batch 1` per arm (35B, --max-seq 20480, --mem-budget 29), `localeval speed --sizes 1024,4096,16384 --gen 128 --reps 3` per arm.
# Stops starting new work when less than MINLEFT s of CAP remain. Radiance is restored by guard.sh on tatooine when this script ends or dies.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}" "${W:?W=<window>}"
CAP=${CAP:-2400}; PORT=8095; MINLEFT=${MINLEFT:-200}
M35=/models/intarsia/Qwen3.6-35B-A3B-NVFP4
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
OUT=$REPO/docs/results/2026-11-07-spec-graph-tp2/raw/$W; mkdir -p "$OUT"
T0=$(date +%s); left() { echo $(( CAP - ($(date +%s) - T0) )); }
say() { echo "$(date +%T) $*" | tee -a "$OUT/steps.log"; }
SSH="ssh -o BatchMode=yes tatooine"
ARMSEQ=${ARMSEQ:-"tp1m tp2off tp2m;tp2m tp2off tp1m"}
armcfg() { # armcfg <arm>: sets GPUS TPA spec
  case $1 in
    tp1m) GPUS=0; TPA=""; spec="--spec mtp --spec-k 2" ;;
    tp1off) GPUS=0; TPA=""; spec="--spec off" ;;
    tp2off) GPUS=0,1; TPA="--tp 2 --devices 0,1"; spec="--spec off" ;;
    tp2m) GPUS=0,1; TPA="--tp 2 --devices 0,1"; spec="--spec mtp --spec-k 2" ;;
    tp2m3) GPUS=0,1; TPA="--tp 2 --devices 0,1"; spec="--spec mtp --spec-k 3" ;;
    *) echo "unknown arm $1" >&2; exit 2 ;;
  esac
}
if [[ ${DRY:-0} == 1 ]]; then
  IFS=';' read -r -a rounds <<< "$ARMSEQ"
  for i in "${!rounds[@]}"; do for arm in ${rounds[$i]}; do armcfg "$arm"; echo "DRY round $((i+1)) arm $arm: HIP_VISIBLE_DEVICES=$GPUS gt2-intarsia serve --backend hip --model $M35 $spec $TPA --max-batch 1 --mem-budget 29 --max-seq 20480"; done; done
  echo "DRY PRE=${PRE-ids} CAP=$CAP"; exit 0
fi
n=$($SSH "grep -c 'ids' ~/intarsiaml-gt2/gt2-intarsia" < /dev/null); [[ ${n:-0} == 0 ]] && { echo "no binary"; exit 4; }
mkdir -p "$REPO/target"; FIFO=$(mktemp -u "$REPO/target/gt2guard.XXXXXX"); mkfifo "$FIFO"
$SSH 'mkdir -p ~/intarsiaml-gt2/.gt2' < /dev/null; scp -q "$HERE/guard.sh" "$HERE/lib.sh" "$HERE"/stages-*.sh tatooine:intarsiaml-gt2/.gt2/
$SSH 'bash ~/intarsiaml-gt2/.gt2/guard.sh' > "$OUT/guard.log" 2>&1 < "$FIFO" &
GUARD=$!
exec 9> "$FIFO"
for i in $(seq 1 30); do grep -q "radiance stopped" "$OUT/guard.log" 2>/dev/null && break; sleep 1; done
say "guard: $(grep -h 'radiance stopped' "$OUT/guard.log")"
cleanup() { $SSH 'docker rm -f gt2-serve' >/dev/null 2>&1 < /dev/null; exec 9>&-; wait $GUARD; rm -f "$FIFO"; say "guard: $(grep -h 'radiance answers\|DID NOT' "$OUT/guard.log")"
  start=$(grep -m1 'radiance stopped' "$OUT/guard.log" | grep -oE '[0-9]{2}:[0-9]{2}:[0-9]{2}'); back=$(grep -m1 'radiance answers' "$OUT/guard.log" | grep -oE '[0-9]{2}:[0-9]{2}:[0-9]{2}' | head -1)
  echo "$(date +%F) fleetq #$FLEETQ_JOB $W radiance stopped $start answers ${back:-NEVER} (guard.log in raw/$W)" >> "$OUT/../downtime.log"; }
trap cleanup EXIT
for stage in ${PRE-ids}; do
  say "pre: stage $stage (cap ${PRE_CAP:-1500} s)"
  $SSH "rm -rf ~/intarsiaml-gt2/.gt2/out/$W-$stage; mkdir -p ~/intarsiaml-gt2/.gt2/out/$W-$stage; GATES='${GATES:-1}' IDNAMES='${IDNAMES:-}' ${PREENV:-} timeout ${PRE_CAP:-1500} bash ~/intarsiaml-gt2/.gt2/stages-$stage.sh ~/intarsiaml-gt2/.gt2/out/$W-$stage" < /dev/null > "$OUT/$stage-pre.log" 2>&1
  say "pre: $stage exit $?"
  mkdir -p "$OUT/$stage"; scp -q "tatooine:intarsiaml-gt2/.gt2/out/$W-$stage/*" "$OUT/$stage/" 2>/dev/null
  say "pre: $(grep -hE 'PASS|FAIL|summary|refusal' "$OUT/$stage/steps.log" | cut -c10- | tr '\n' '|')"
  if [[ ${STOP_ON_FAIL:-1} == 1 ]] && grep -q 'FAIL ids' "$OUT/$stage/steps.log"; then say "STOP: an ids gate failed (STOP_ON_FAIL), no speed arms"; exit 6; fi
done
serve() { # serve <arm>
  local arm=$1 spec TPA GPUS
  armcfg "$arm"
  RECIPE="HIP_VISIBLE_DEVICES=$GPUS gt2-intarsia serve --backend hip --model 35B $spec $TPA --max-batch 1 --profile-every 0 --mem-budget 29 --max-seq 20480 (docker $IMG, lane gt2)"
  $SSH "docker rm -f gt2-serve >/dev/null 2>&1; docker run -d --name gt2-serve --network host --device /dev/kfd --device /dev/dri \
    --group-add \$(stat -c %g /dev/kfd) --group-add \$(stat -c %g /dev/dri/renderD128) --group-add video --ipc host \
    --security-opt seccomp=unconfined --user \$(id -u):\$(id -g) -e HOME=/tmp -e GPU_MAX_HW_QUEUES=1 -e HIP_VISIBLE_DEVICES=$GPUS \
    -v \$HOME/intarsiaml-gt2:/work -v /data/models:/models:ro -w /work $IMG /work/gt2-intarsia serve --backend hip --model $M35 \
    --served-model-name q35-$arm $spec $TPA --max-batch 1 --profile-every 0 --mem-budget 29 --max-seq 20480 --no-supervisor --host 0.0.0.0 --port $PORT" \
    < /dev/null > "$OUT/docker-$arm.id" 2>&1
  for i in $(seq 1 100); do curl -sf -m 3 "http://tatooine:$PORT/health" > /dev/null 2>&1 && return 0; sleep 3; done
  $SSH 'docker logs --tail 40 gt2-serve' > "$OUT/serve-fail-$arm.log" 2>&1 < /dev/null; return 1
}
metric() { curl -s -m 5 "http://tatooine:$PORT/metrics" | grep -E '^intarsia_spec_(steps|proposed_tokens|accepted_tokens)_total' | sed 's/{[^}]*}//' | tr '\n' ' '; }
IFS=';' read -r -a rounds <<< "$ARMSEQ"
for i in "${!rounds[@]}"; do
  r=$((i + 1))
  for arm in ${rounds[$i]}; do
    if (( $(left) < MINLEFT )); then say "SKIPPED round $r arm $arm (time left $(left)s)"; continue; fi
    say "round $r arm $arm: starting server"
    if ! serve "$arm"; then say "round $r arm $arm: server did not come up"; $SSH 'docker rm -f gt2-serve' >/dev/null 2>&1 < /dev/null; continue; fi
    m0=$(metric)
    ( cd /opt/code/deepeval && uv run localeval speed --url "http://tatooine:$PORT/v1" --model "q35-$arm" --thinking off --sizes "${SIZES:-1024,4096,16384}" --gen 128 --reps 3 \
        --label "gt2-$W-$arm-r$r" --no-unim --recipe "$RECIPE" --wait-idle 60 ) > "$OUT/speed-$arm-r$r.log" 2>&1 < /dev/null
    say "round $r arm $arm speed: exit $? $(grep -aE 'decode' "$OUT/speed-$arm-r$r.log" | head -4 | tr '\n' '|' | cut -c1-400)"
    echo "$arm r$r before: $m0 after: $(metric)" >> "$OUT/spec-metrics.txt"
    $SSH 'docker logs gt2-serve 2>&1 | grep -aE "speculative|mtp|graph|error|admission|window|--spec" | head -12' < /dev/null > "$OUT/serve-$arm-r$r.log" 2>&1
    $SSH 'docker rm -f gt2-serve' >/dev/null 2>&1 < /dev/null
  done
done
say "done, time left $(left)s"

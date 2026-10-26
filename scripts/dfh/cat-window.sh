#!/usr/bin/env bash
# Lane dfh: per-category speed window for the 27B at tp=1 (card 0), driven FROM THE MAC (localeval lives here), one fleetq job that
# holds both R9700s and takes radiance down (after scripts/tp2-hip/speed-window.sh):
#   fleetq run --host tatooine-r9700,tatooine-r9700-2 --takes-down radiance --quiet gpu,cpu --expect 50 --mem 30 \
#     --label "dfh cat" -- env W=w5 scripts/dfh/cat-window.sh
# Arms (ARMS, default "off mtp2 dfl7"): a fresh `serve --max-batch 1` per arm, then `localeval quality --judge none` over every file
# of scripts/dfh/cats/ (categories-rcp recall/code/prose and the held-out prose/chat/code/recall, one call per file so the server's
# speculation counters can be read per category), max 512 tokens forced (ignore_eos + min_tokens through a lane-local
# LOCALEVAL_TARGETS file). Rounds (ROUNDS, default 1) alternate the arm order. PROBE=1 first runs tests/dflash2_real.rs on card 0.
# Radiance is restored by guard.sh on tatooine when this script ends or dies.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}" "${W:?W=<window>}"
TREE=${TREE:-dev}; ROUNDS=${ROUNDS:-1}; ARMS=${ARMS:-"off mtp2 dfl7"}; GEN=${GEN:-512}; CAP=${CAP:-2700}; PORT=8095
M27=/models/radiance/hf-cache/hub/models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/5233554c5fa56afda40150556b95573c2d7d29c0
DRAFT=/models/tensorfold/Qwen3.8-27B-DFlash2
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
OUT=$REPO/docs/results/2026-10-26-dfh-hip-dflash/raw/$W; mkdir -p "$OUT"
T0=$(date +%s); left() { echo $(( CAP - ($(date +%s) - T0) )); }
say() { echo "$(date +%T) $*" | tee -a "$OUT/steps.log"; }
SSH="ssh -o BatchMode=yes tatooine"
FIFO=$(mktemp -u "$REPO/target/dfhguard.XXXXXX"); mkfifo "$FIFO"
$SSH 'mkdir -p ~/dfh' < /dev/null; scp -q "$HERE/guard.sh" "$HERE/lib.sh" tatooine:dfh/
$SSH 'bash ~/dfh/guard.sh' > "$OUT/guard.log" 2>&1 < "$FIFO" &
GUARD=$!
exec 9> "$FIFO"
for i in $(seq 1 30); do grep -q "radiance stopped" "$OUT/guard.log" 2>/dev/null && break; sleep 1; done
say "guard: $(grep -h 'radiance stopped' "$OUT/guard.log")"
cleanup() { $SSH 'docker rm -f dfh-serve' >/dev/null 2>&1 < /dev/null; exec 9>&-; wait $GUARD; rm -f "$FIFO"; say "guard: $(grep -h 'radiance answers\|DID NOT' "$OUT/guard.log")"
  start=$(grep -m1 'radiance stopped' "$OUT/guard.log" | grep -oE '[0-9]{2}:[0-9]{2}:[0-9]{2}'); back=$(grep -m1 'radiance answers' "$OUT/guard.log" | grep -oE '[0-9]{2}:[0-9]{2}:[0-9]{2}' | head -1)
  echo "$(date +%F) fleetq #$FLEETQ_JOB $W radiance stopped $start answers ${back:-NEVER} (guard.log in raw/$W)" >> "$OUT/../downtime.log"; }
trap cleanup EXIT

if [[ ${PROBE:-0} == 1 ]]; then
  say "probe: tests/dflash2_real.rs on card 0"
  $SSH "export O=\$HOME/dfh/out/$W; mkdir -p \$O; source ~/dfh/lib.sh; E='INTARSIA_DFLASH2_TARGET=$M27 INTARSIA_DFLASH2_DRAFT=$DRAFT INTARSIA_DFLASH2_LEN=3000' mtest $TREE probe27 0 600 --test dflash2_real -- --nocapture; cat \$O/mtest-probe27.log" < /dev/null > "$OUT/mtest-probe27.log" 2>&1
  say "probe: $(grep -aE 'PROBE_TIME|PROBE_DRAFT_PROFILE' "$OUT/mtest-probe27.log" | tr '\n' '|')"
fi

serve() { # serve <arm>
  local arm=$1 spec
  case $arm in
    off) spec="--spec off" ;;
    mtp2) spec="--spec mtp --spec-k 2" ;;
    mtp*) spec="--spec mtp --spec-k ${arm#mtp}" ;;
    dfl*) spec="--spec dflash --spec-draft $DRAFT --spec-k ${arm#dfl}" ;;
  esac
  RECIPE="HIP_VISIBLE_DEVICES=0 dfh-intarsia serve --backend hip --model 27B $spec --max-batch 1 --profile-every 0 --mem-budget 29 --max-seq 8192 (tree $TREE, docker $IMG, lane dfh)"
  $SSH "docker rm -f dfh-serve >/dev/null 2>&1; docker run -d --name dfh-serve --network host --device /dev/kfd --device /dev/dri \
    --group-add \$(stat -c %g /dev/kfd) --group-add \$(stat -c %g /dev/dri/renderD128) --group-add video --ipc host \
    --security-opt seccomp=unconfined --user \$(id -u):\$(id -g) -e HOME=/tmp -e GPU_MAX_HW_QUEUES=1 -e HIP_VISIBLE_DEVICES=0 \
    -v \$HOME/intarsiaml-dfh-$TREE:/work -v /data/models:/models:ro -w /work $IMG /work/dfh-intarsia serve --backend hip --model $M27 \
    --served-model-name qwen38-27b-$arm $spec --max-batch 1 --profile-every 0 --mem-budget 29 --max-seq 8192 --no-supervisor --host 0.0.0.0 --port $PORT" \
    < /dev/null > "$OUT/docker-$arm.id" 2>&1
  for i in $(seq 1 100); do curl -sf -m 3 "http://tatooine:$PORT/health" > /dev/null 2>&1 && return 0; sleep 3; done
  $SSH 'docker logs --tail 40 dfh-serve' > "$OUT/serve-fail-$arm.log" 2>&1 < /dev/null; return 1
}
metric() { curl -s -m 5 "http://tatooine:$PORT/metrics" | grep -E '^intarsia_spec_(steps|proposed_tokens|accepted_tokens|costgate_[a-z_]+)_total' | sed 's/{[^}]*}//' | tr '\n' ' '; }

for r in $(seq 1 "$ROUNDS"); do
  order="$ARMS"; (( r % 2 == 0 )) && order=$(echo $ARMS | awk '{for (i=NF;i>=1;i--) printf "%s ", $i}')
  for arm in $order; do
    if (( $(left) < 600 )); then say "SKIPPED round $r arm $arm (time left $(left)s)"; continue; fi
    say "round $r arm $arm: starting server"
    if ! serve "$arm"; then say "round $r arm $arm: server did not come up"; continue; fi
    printf '[targets.dfh-%s]\nurl = "http://tatooine:%s/v1"\nmodel = "qwen38-27b-%s"\nextra_body = { ignore_eos = true, min_tokens = %s }\n' "$arm" "$PORT" "$arm" "$GEN" > "$OUT/targets-$arm.toml"
    for f in "$HERE"/cats/*.jsonl; do
      c=$(basename "$f" .jsonl); m0=$(metric)
      ( cd /opt/code/deepeval && LOCALEVAL_TARGETS="$OUT/targets-$arm.toml" uv run localeval quality "dfh-$arm" --thinking off --label "dfh-$W-$arm-$c-r$r" \
          --recipe "$RECIPE" --dataset "$f" --judge none --max-tokens "$GEN" --wait-idle 60 ) > "$OUT/q-$arm-$c-r$r.log" 2>&1 < /dev/null
      rc=$?; m1=$(metric)
      echo "$arm $c r$r before: $m0" >> "$OUT/spec-metrics.txt"; echo "$arm $c r$r after:  $m1" >> "$OUT/spec-metrics.txt"
      say "round $r arm $arm $c: localeval exit $rc $(grep -aoE 'results/[^ ]+' "$OUT/q-$arm-$c-r$r.log" | tail -1)"
    done
    $SSH 'docker logs dfh-serve 2>&1 | grep -aE "speculative|dflash|mtp|graph|error" | tail -12' < /dev/null > "$OUT/serve-$arm-r$r.log" 2>&1
    $SSH 'docker rm -f dfh-serve' >/dev/null 2>&1 < /dev/null
  done
done
say "done, time left $(left)s"

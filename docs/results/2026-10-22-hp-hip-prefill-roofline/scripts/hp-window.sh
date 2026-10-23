#!/usr/bin/env bash
# Lane hp: ONE radiance-down window on tatooine, started from endor as
#   fleetq run --host tatooine --exclusive --takes-down radiance --mem 30 --expect <min> --label "hp: ..." -- \
#     env OUT=<raw dir> STAGES="ceiling prof:base ..." BUDGET=<s> bash hp-window.sh
# Preflight (radiance untouched if it fails), stop radiance, run hp-stages.sh on tatooine (BUDGET s), copy its output dir
# back to $OUT, restart radiance in an EXIT trap and poll /v1/models (<= 180 s). Downtime goes to $OUT/downtime.log.
# LOC=<rounds>: after the stages, localeval A/B rounds (base, dev) at 1k/4k/16k, LOC_ENV = env of the dev arm.
set -uo pipefail
: "${OUT:?set OUT to the raw output dir}"
mkdir -p "$OUT"
T=tatooine
LOG=$OUT/downtime.log
now() { date '+%F %T'; }
say() { echo "[window] $* $(now)" | tee -a "$LOG"; }
STAGES=${STAGES:?set STAGES}

restore() {
  ssh $T 'docker rm -f intarsia-serve >/dev/null 2>&1' < /dev/null
  say "starting radiance"
  ssh $T 'systemctl --user start radiance' < /dev/null
  local i
  for i in $(seq 1 36); do
    if ssh $T 'curl -s -m 3 http://localhost:8080/v1/models' < /dev/null | grep -q '"data":\['; then
      say "radiance answers (after $((i * 5)) s)"
      return 0
    fi
    sleep 5
  done
  say "RADIANCE DID NOT ANSWER within 180 s"
  return 1
}

loc_stop() { ssh $T 'docker rm -f intarsia-serve >/dev/null 2>&1; true' < /dev/null; }
loc_arm() { # loc_arm <label> <tree> <volume> [docker -e args]
  local label=$1 tr=$2 vol=$3 envs=${4:-} i
  loc_stop
  ssh $T "docker run -d --name intarsia-serve --network host --device /dev/kfd --device /dev/dri --group-add \$(stat -c %g /dev/kfd) --group-add \$(stat -c %g /dev/dri/renderD128) --group-add video --ipc host --security-opt seccomp=unconfined --user \$(id -u):\$(id -g) -e HOME=/tmp -e GPU_MAX_HW_QUEUES=1 -e HIP_VISIBLE_DEVICES=0 $envs -v \$HOME/$tr:/work -v /data/models/intarsia:/models:ro -v $vol:/target -w /work intarsia-hip-dev:rocm7.2.4-rust1.90.0 /target/release/intarsia serve --backend hip --model /models/Qwen3.6-35B-A3B-NVFP4 --profile-every 0 --mem-budget 29 --host 0.0.0.0 --port 8090 --served-model-name qwen36-35b-intarsia-hip --max-seq 17408" < /dev/null > "$OUT/loc-$label-start.out" 2>&1
  for ((i = 0; i < 240; i += 3)); do curl -sf -m 2 "http://$T:8090/health" > /dev/null 2>&1 && break; sleep 3; done
  say "localeval arm $label up"
  (cd /opt/code/deepeval && uv run localeval speed --url "http://$T:8090/v1" --model qwen36-35b-intarsia-hip --sizes 1024,4096,16384 --gen 128 --reps 3 --thinking off --label "hp-$label") < /dev/null > "$OUT/loc-$label.txt" 2>&1
  say "localeval arm $label exit $?"
  loc_stop
}

# preflight: nothing below stops radiance until these pass
ssh $T "test -x ~/intarsiaml-hp/docs/results/2026-10-22-hp-hip-prefill-roofline/scripts/hp-stages.sh && test -s ~/hp-prompts/prompt16384.txt && docker run --rm -v intarsia-cargo-target-hp-base:/t:ro intarsia-hip-dev:rocm7.2.4-rust1.90.0 test -x /t/release/intarsia" < /dev/null || { echo "preflight failed, radiance untouched"; exit 3; }
ssh $T 'systemctl --user is-active radiance' < /dev/null | tee -a "$LOG"

say "job ${FLEETQ_JOB:-none}; stages: $STAGES; stopping radiance"
trap restore EXIT
ssh $T 'systemctl --user stop radiance' < /dev/null
sleep 2
say "radiance stopped"
ssh $T "PROF_NS='${PROF_NS:-4096 16384}' BUDGET=${BUDGET:-1500} bash ~/intarsiaml-hp/docs/results/2026-10-22-hp-hip-prefill-roofline/scripts/hp-stages.sh $STAGES" < /dev/null 2>&1 | tee "$OUT/window.log"
if [ "${LOC:-0}" -gt 0 ] && ssh $T 'test -e ~/intarsiaml-hp/bench-out/hp/out/ABORT' < /dev/null; then
  say "stages aborted (a test failed): localeval skipped"
elif [ "${LOC:-0}" -gt 0 ]; then
  for r in $(seq 1 "$LOC"); do
    loc_arm "base-r$r" intarsiaml-hpbase intarsia-cargo-target-hp-base
    loc_arm "dev-r$r" intarsiaml-hp intarsia-cargo-target-hp-dev "${LOC_ENV:-}"
  done
fi
say "stages done"
rsync -a $T:intarsiaml-hp/bench-out/hp/out/ "$OUT/" < /dev/null
ssh $T 'rm -rf ~/intarsiaml-hp/bench-out/hp/out' < /dev/null

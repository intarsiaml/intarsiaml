#!/usr/bin/env bash
# Lane pf: ONE radiance-down window on tatooine, started from endor as
#   fleetq run --host tatooine --exclusive --takes-down radiance --mem 30 --label "pf: ..." -- \
#     env OUT=<raw dir> STAGES="prof opvp bitwise ab ids chunk" bash scripts/hip-decode/pf-window.sh
# Stops radiance, runs scripts/hip-decode/pf-stages.sh on tatooine (BUDGET s, default 2700), copies bench-out/pf/out back to
# $OUT (env: LOC = localeval rounds (base, dev) after the stages, BUDGET, VARMS = INTARSIA_HIP_FLASH_VP values for the vpm / ab stages, IDV = the value the ids stage checks), and restarts radiance in an EXIT trap (polls /v1/models <= 180 s). The downtime goes to $OUT/downtime.log.
set -uo pipefail
: "${OUT:?set OUT to the raw output dir}"
mkdir -p "$OUT"
T=tatooine
LOG=$OUT/downtime.log
now() { date '+%F %T'; }
say() { echo "[window] $* $(now)" | tee -a "$LOG"; }
STAGES=${STAGES:-prof opvp bitwise ab ids chunk}
HERE=$(cd "$(dirname "$0")" && pwd)

restore() {
  say "starting radiance"
  ssh $T 'docker rm -f intarsia-serve >/dev/null 2>&1; systemctl --user start radiance' < /dev/null
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


# localeval A/B (the standings' metric): LOC rounds of {base binary, dev binary with the build's default knobs}, each arm = a server
# on GPU 0 + `localeval run --sizes 4096,16384 --reps 3`. Runs on endor (needs /opt/code/deepeval); FLEETQ_JOB makes localeval run
# inside this window's slot.
loc_stop() { ssh $T 'docker rm -f intarsia-serve >/dev/null 2>&1; true' < /dev/null; }
loc_arm() { # loc_arm <label> <cargo target volume>
  local label=$1 vol=$2 i
  loc_stop
  ssh $T "cd ~/intarsiaml-pf && docker run -d --name intarsia-serve --network host --device /dev/kfd --device /dev/dri --group-add \$(stat -c %g /dev/kfd) --group-add \$(stat -c %g /dev/dri/renderD128) --group-add video --ipc host --security-opt seccomp=unconfined --user \$(id -u):\$(id -g) -e HOME=/tmp -e GPU_MAX_HW_QUEUES=1 -e HIP_VISIBLE_DEVICES=0 -v \$HOME/intarsiaml-pf:/work -v /data/models/intarsia:/models:ro -v $vol:/target -w /work intarsia-hip-dev:rocm7.2.4-rust1.90.0 /target/release/intarsia serve --backend hip --model /models/Qwen3.6-35B-A3B-NVFP4 --profile-every 0 --mem-budget 29 --host 0.0.0.0 --port 8090 --served-model-name qwen36-35b-intarsia-hip --max-seq 17408" < /dev/null > "$OUT/loc-$label-start.out" 2>&1
  for ((i = 0; i < 240; i += 3)); do curl -sf -m 2 "http://$T:8090/health" > /dev/null 2>&1 && break; sleep 3; done
  say "localeval arm $label up"
  (cd /opt/code/deepeval && uv run localeval run --url "http://$T:8090/v1" --model qwen36-35b-intarsia-hip --sizes 4096,16384 --gen 128 --reps 3 --thinking off --label "pf-$label") > "$OUT/loc-$label.txt" 2>&1
  say "localeval arm $label exit $?"
  loc_stop
}

# preflight: nothing below stops radiance until these pass
ssh $T "test -x ~/intarsiaml-pf/scripts/hip-decode/pf-stages.sh && test -s ~/intarsiaml-pf/bench-out/pf/prompt16384.txt && docker run --rm -v intarsia-cargo-target-pf:/t:ro intarsia-hip-dev:rocm7.2.4-rust1.90.0 test -x /t/release/intarsia && docker run --rm -v intarsia-cargo-target-pf-dev:/t:ro intarsia-hip-dev:rocm7.2.4-rust1.90.0 test -x /t/release/intarsia" < /dev/null || { echo "preflight failed, radiance untouched"; exit 3; }
ssh $T 'systemctl --user is-active radiance' < /dev/null | tee -a "$LOG"

say "job ${FLEETQ_JOB:-none}; stages: $STAGES; stopping radiance"
trap restore EXIT
ssh $T 'systemctl --user stop radiance' < /dev/null
sleep 2
say "radiance stopped"
ssh $T "BUDGET=${BUDGET:-2700} VARMS='${VARMS:-}' IDV='${IDV:-1}' bash ~/intarsiaml-pf/scripts/hip-decode/pf-stages.sh $STAGES" < /dev/null 2>&1 | tee "$OUT/window.log"
if [ "${LOC:-0}" -gt 0 ]; then
  for r in $(seq 1 "$LOC"); do
    loc_arm "base-r$r" intarsia-cargo-target-pf
    loc_arm "dev-r$r" intarsia-cargo-target-pf-dev
  done
fi
say "stages done"
rsync -a $T:intarsiaml-pf/bench-out/pf/out/ "$OUT/" < /dev/null

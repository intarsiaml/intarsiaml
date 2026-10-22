#!/usr/bin/env bash
# Lane tc: ONE radiance-down window on tatooine (at most 20 min), started from endor as
#   fleetq run --host tatooine --exclusive --takes-down radiance --mem 30 --label "tc: ..." -- \
#     env OUT=<raw dir> STAGES="ids speed" TM=1024 BUDGET=1000 bash scripts/hip-decode/tc-window.sh
# Stops radiance, runs scripts/hip-decode/tc-stages.sh on tatooine (stages ids / acc), then the localeval speed arms (stage
# `speed`, on endor, inside this slot via FLEETQ_JOB), and restarts radiance in an EXIT trap (polls /v1/models <= 180 s).
# The downtime goes to $OUT/downtime.log. Arms: base = origin/main, off = lane branch with the knob unset, on = knob TM.
set -uo pipefail
: "${OUT:?set OUT to the raw output dir}"
mkdir -p "$OUT"
T=tatooine
LOG=$OUT/downtime.log
TM=${TM:-1024}
BUDGET=${BUDGET:-1000}
STAGES=${STAGES:-ids speed}
SPEEDARMS=${SPEEDARMS:-base on off}
now() { date '+%F %T'; }
say() { echo "[window] $* $(now)" | tee -a "$LOG"; }
T0=$(date +%s)
left() { echo $((BUDGET - ($(date +%s) - T0))); }

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

# localeval speed arm: a server on GPU 0 (knob exported through docker -e for `on`), then `localeval run --sizes 4096,16384 --reps 3`
loc_stop() { ssh $T 'docker rm -f intarsia-serve >/dev/null 2>&1; true' < /dev/null; }
loc_arm() { # loc_arm <arm> <round>
  local arm=$1 r=$2 repo vol envx="" i
  case $arm in
    base) repo=intarsiaml-tcbase; vol=intarsia-cargo-target-tcbase ;;
    off) repo=intarsiaml-tc; vol=intarsia-cargo-target-tc ;;
    on) repo=intarsiaml-tc; vol=intarsia-cargo-target-tc; envx="-e INTARSIA_PREFILL_TAIL_MERGE=$TM" ;;
  esac
  local label="$arm-r$r"
  loc_stop
  ssh $T "docker run -d --name intarsia-serve --network host --device /dev/kfd --device /dev/dri --group-add \$(stat -c %g /dev/kfd) --group-add \$(stat -c %g /dev/dri/renderD128) --group-add video --ipc host --security-opt seccomp=unconfined --user \$(id -u):\$(id -g) -e HOME=/tmp -e GPU_MAX_HW_QUEUES=1 -e HIP_VISIBLE_DEVICES=0 $envx -v \$HOME/$repo:/work -v /data/models/intarsia:/models:ro -v $vol:/target -w /work intarsia-hip-dev:rocm7.2.4-rust1.90.0 /target/release/intarsia serve --backend hip --model /models/Qwen3.6-35B-A3B-NVFP4 --profile-every 0 --mem-budget 29 --host 0.0.0.0 --port 8090 --served-model-name qwen36-35b-intarsia-hip --max-seq 17408" < /dev/null > "$OUT/loc-$label-start.out" 2>&1
  for ((i = 0; i < 240; i += 3)); do curl -sf -m 2 "http://$T:8090/health" > /dev/null 2>&1 && break; sleep 3; done
  say "localeval arm $label up"
  ssh $T 'docker logs intarsia-serve 2>&1 | head -80' < /dev/null > "$OUT/loc-$label-serve.log" 2>&1
  (cd /opt/code/deepeval && uv run localeval run --url "http://$T:8090/v1" --model qwen36-35b-intarsia-hip --sizes 4096,16384 --gen 128 --reps 3 --thinking off --label "tc-$label") > "$OUT/loc-$label.txt" 2>&1
  say "localeval arm $label exit $?"
  loc_stop
}

# preflight: nothing below stops radiance until these pass
ssh $T "test -x ~/intarsiaml-tc/scripts/hip-decode/tc-stages.sh && test -s ~/intarsiaml-tc/bench-out/tc/prompt16654.txt && test -s ~/intarsiaml-tcbase/bench-out/tc/prompt16654.txt && docker run --rm -v intarsia-cargo-target-tc:/t:ro intarsia-hip-dev:rocm7.2.4-rust1.90.0 test -x /t/release/intarsia && docker run --rm -v intarsia-cargo-target-tcbase:/t:ro intarsia-hip-dev:rocm7.2.4-rust1.90.0 test -x /t/release/intarsia" < /dev/null || { echo "preflight failed, radiance untouched"; exit 3; }
ssh $T 'systemctl --user is-active radiance' < /dev/null | tee -a "$LOG"

say "job ${FLEETQ_JOB:-none}; stages: $STAGES; stopping radiance"
trap restore EXIT
ssh $T 'systemctl --user stop radiance' < /dev/null
sleep 2
say "radiance stopped"
T0=$(date +%s)
GPU_STAGES=$(for s in $STAGES; do [ "$s" != speed ] && echo -n "$s "; done)
if [ -n "$GPU_STAGES" ]; then
  ssh $T "BUDGET=$((BUDGET - 60)) TM=$TM bash ~/intarsiaml-tc/scripts/hip-decode/tc-stages.sh $GPU_STAGES" < /dev/null 2>&1 | tee "$OUT/window.log"
fi
if [[ " $STAGES " == *" speed "* ]]; then
  r=1
  while :; do
    for a in $SPEEDARMS; do
      if [ "$(left)" -lt 240 ]; then say "SKIPPED speed arm $a-r$r: $(left) s left"; continue; fi
      loc_arm "$a" "$r"
    done
    r=$((r + 1))
    [ "$(left)" -ge 800 ] || break
  done
fi
say "stages done"
rsync -a $T:intarsiaml-tc/bench-out/tc/out/ "$OUT/" < /dev/null

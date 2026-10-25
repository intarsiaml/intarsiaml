#!/usr/bin/env bash
# Lane mx: ONE radiance-down localeval window on tatooine, from the Mac (after lane tc's tc-window.sh). Per round, every arm
# starts `intarsia serve` on card 0 from its tree, runs `localeval speed` against it inside this fleetq slot (FLEETQ_JOB),
# and stops it; rounds alternate the arm order. radiance comes back in an EXIT trap (polls /v1/models <= 200 s).
#   fleetq run --host tatooine-r9700,tatooine-r9700-2 --takes-down radiance --quiet gpu,cpu --mem 30 --expect 45 \
#     --label "mx loc ..." -- env W=loc1 ARMS="base27 dev27 base35 dev35" ROUNDS=3 scripts/mx/loc-window.sh
# An arm is <tree><model>[:<extra serve args, commas for spaces>]: tree base|dev (~/intarsiaml-mx-<tree>), model 27|35.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}" "${W:?W=<window name>}"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$S/../.." && pwd)"
OUT=$REPO/docs/results/2026-10-25-mx-hip-27b-prefill/raw/$W
mkdir -p "$OUT"
LOG=$OUT/window.log
T=tatooine
ARMS=${ARMS:-base27 dev27}
ROUNDS=${ROUNDS:-3}
SIZES=${SIZES:-1024,4096,16384}
REPS=${REPS:-2}
BUDGET=${BUDGET:-2500}
M27=/models/models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/5233554c5fa56afda40150556b95573c2d7d29c0
now() { date '+%F %T'; }
say() { echo "[window] $* $(now)" | tee -a "$LOG"; }
T0=$(date +%s)
left() { echo $((BUDGET - ($(date +%s) - T0))); }
restore() {
  say "starting radiance"
  ssh $T 'docker rm -f intarsia-serve >/dev/null 2>&1; systemctl --user start radiance' < /dev/null
  local i
  for i in $(seq 1 40); do
    if ssh $T 'curl -s -m 3 http://localhost:8080/v1/models' < /dev/null | grep -q '"data":\['; then
      say "radiance answers (after $((i * 5)) s); $(ssh $T 'systemctl --user is-active radiance' < /dev/null)"
      echo "$(now) job $FLEETQ_JOB window $W: $(grep -E 'stopping radiance|radiance answers|DID NOT' "$LOG" | tr '\n' ' ')" >> "$OUT/../downtime.log"
      return 0
    fi
    sleep 5
  done
  say "RADIANCE DID NOT ANSWER within 200 s"
  return 1
}
arm() {  # arm <spec> <round>
  local spec=$1 r=$2 tree model extra name mdir mpath
  local base=${spec%%:*}; extra=""; [[ $spec == *:* ]] && extra=${spec#*:} && extra=${extra//,/ }
  tree=${base%??}; model=${base: -2}
  if [[ $model == 27 ]]; then mdir=/data/models/radiance/hf-cache/hub; mpath=$M27; name=qwen38-27b-intarsia-hip
  else mdir=/data/models/intarsia; mpath=/models/Qwen3.6-35B-A3B-NVFP4; name=qwen36-35b-intarsia-hip; fi
  local label="mx-$W-$base${extra:+-$(echo "$extra" | tr -c 'A-Za-z0-9=' '_')}-r$r"
  ssh $T "docker rm -f intarsia-serve >/dev/null 2>&1; docker run -d --name intarsia-serve --network host --device /dev/kfd --device /dev/dri \
    --group-add \$(stat -c %g /dev/kfd) --group-add \$(stat -c %g /dev/dri/renderD128) --group-add video --ipc host \
    --security-opt seccomp=unconfined --user \$(id -u):\$(id -g) -e HOME=/tmp -e GPU_MAX_HW_QUEUES=1 -e HIP_VISIBLE_DEVICES=0 \
    ${ENVX:-} -v \$HOME/intarsiaml-mx-$tree:/work -v $mdir:/models:ro -w /work intarsia-hip-dev:rocm7.2.4-rust1.90.0 \
    /work/mx-intarsia serve --backend hip --model $mpath --profile-every 0 --mem-budget 29 --host 0.0.0.0 --port 8090 \
    --served-model-name $name --max-seq 17408 $extra" < /dev/null > "$OUT/$label-start.out" 2>&1
  local i up=0
  for ((i = 0; i < 300; i += 3)); do curl -sf -m 2 "http://$T:8090/health" > /dev/null 2>&1 && up=1 && break; sleep 3; done
  ssh $T 'docker logs intarsia-serve 2>&1 | head -60' < /dev/null > "$OUT/$label-serve.log" 2>&1
  if ((up == 0)); then say "arm $label DID NOT COME UP"; ssh $T 'docker rm -f intarsia-serve' < /dev/null > /dev/null 2>&1; return 1; fi
  say "arm $label up"
  (cd /opt/code/deepeval && uv run localeval speed --url "http://$T:8090/v1" --model $name --sizes "$SIZES" --gen 128 --reps "$REPS" \
    --thinking off --label "$label" --recipe "intarsia serve --backend hip --model $mpath --mem-budget 29 --max-seq 17408 $extra (tree $tree, $(ssh $T "cat ~/intarsiaml-mx-$tree/.stamp 2>/dev/null" < /dev/null))") > "$OUT/$label.txt" 2>&1
  say "arm $label localeval exit $?: $(grep -aE 'run id|run_id|results/' "$OUT/$label.txt" | head -2 | tr '\n' ' ')"
  ssh $T 'docker logs intarsia-serve > /dev/null 2>&1; docker rm -f intarsia-serve' < /dev/null > /dev/null 2>&1
}
# preflight: nothing below stops radiance until it passes
for spec in $ARMS; do b=${spec%%:*}; t=${b%??}
  ssh $T "test -x ~/intarsiaml-mx-$t/mx-intarsia" < /dev/null || { echo "preflight failed ($t), radiance untouched"; exit 3; }
done
trap restore EXIT
say "job $FLEETQ_JOB $W arms [$ARMS] rounds $ROUNDS; stopping radiance"
ssh $T 'systemctl --user stop radiance' < /dev/null
sleep 3
for ((r = 1; r <= ROUNDS; r++)); do
  order=$ARMS; ((r % 2 == 0)) && order=$(echo $ARMS | tr ' ' '\n' | tail -r | tr '\n' ' ')
  for a in $order; do
    if (($(left) < 240)); then say "SKIPPED $a r$r: $(left) s left"; continue; fi
    arm "$a" "$r"
  done
done
say "rounds done"

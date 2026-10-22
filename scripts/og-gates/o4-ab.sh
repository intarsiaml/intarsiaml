#!/usr/bin/env bash
# O4: is the 35B 4096-token HIP prefill delta of round 9 (-0.30 % in W9, -0.22 % in W7) noise? Runs ON tatooine, inside a
# radiance-down window (og-burst.sh), one R9700 (HIP_VISIBLE_DEVICES=0), one binary (tree ~/og-hip, current main).
# Arms (same binary, env only):
#   dflt   defaults                                         (levers 1 and 2 of round 9 on)
#   off    INTARSIA_HIP_STAGE=0 INTARSIA_HIP_SPLIT_MIN_US=0 (lever 1 page-locked staging off, lever 2 split trigger = round 8's rule)
# Each sample is a FRESH `serve` (35B, --max-seq 8192, --mem-budget 29, graphs on, --profile-every 0) and `bench speed
# --at-most --sizes 4096,3968 --gen 16 --reps 5` (the median of 5 is the sample). --at-most keeps every prompt <= its size, so
# 4096 stays in ONE 4096-token prefill chunk (round 7 saw 4079..4120-token prompts and a second chunk past 4096). ABBA order:
# round 1 dflt off, round 2 off dflt, ... so drift cancels in the pairs. 3968 is the control that sits well below the chunk.
#   o4-ab.sh <outdir under ~/og-hip> [rounds]
set -uo pipefail
REPO=${INTARSIA_REPO:-$HOME/og-hip}
OUT=${1:?outdir under $REPO}; ROUNDS=${2:-5}
export INTARSIA_REPO=$REPO INTARSIA_TARGET_VOL=${INTARSIA_TARGET_VOL:-intarsia-cargo-target-og}
D=deploy/tatooine/dev.sh
cd "$REPO"
mkdir -p "$OUT"
RELOUT=${OUT#$REPO/}
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
B=/target/release/intarsia; M35=/models/Qwen3.6-35B-A3B-NVFP4
NAME=og-serve; PORT=8090
ts() { date +%T; }
trap 'docker rm -f $NAME >/dev/null 2>&1' EXIT
sample() { # sample <arm> <round>
  local arm=$1 r=$2 envs=""
  [[ $arm == off ]] && envs="-e INTARSIA_HIP_STAGE=0 -e INTARSIA_HIP_SPLIT_MIN_US=0"
  docker rm -f $NAME >/dev/null 2>&1
  echo "--- $arm r$r serve [$(ts)]"
  docker run -d --name $NAME --network host \
    --device /dev/kfd --device /dev/dri --group-add "$(stat -c %g /dev/kfd)" --group-add "$(stat -c %g /dev/dri/renderD128)" --group-add video \
    --ipc host --security-opt seccomp=unconfined --user "$(id -u):$(id -g)" -e HOME=/tmp -e GPU_MAX_HW_QUEUES=1 -e HIP_VISIBLE_DEVICES=${RS_GPU:-0} \
    $envs \
    -v "$REPO":/work -v /data/models/intarsia:/models:ro -v "$($D --target-vol)":/target -w /work \
    $IMG $B serve --backend hip --model $M35 --profile-every 0 --mem-budget 29 --host 0.0.0.0 --port $PORT --served-model-name og-35b --max-seq 8192 >/dev/null
  local up=0 i
  for i in $(seq 1 200); do
    curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { up=1; break; }
    docker ps -q -f name=$NAME | grep -q . || break
    sleep 3
  done
  if [[ $up != 1 ]]; then echo "serve did not come up"; docker logs $NAME 2>&1 | tail -20; docker rm -f $NAME >/dev/null; return 1; fi
  echo "up after ~$((i * 3)) s"
  docker logs $NAME 2>&1 | grep -aiE "prefill chunk|graph" | head -3 > "$OUT/serve-$arm-r$r.txt"
  docker exec $NAME $B bench speed --url http://127.0.0.1:$PORT --model og-35b --sizes 4096,3968 --gen 16 --reps 5 --at-most \
    --gb-per-token 2.2 --bandwidth-gbs 627 --label "o4-$arm-r$r" --out-dir /work/$RELOUT --thinking off 2>&1 | tail -8
  docker rm -f $NAME >/dev/null
}
echo "== o4-ab start $(ts) rounds $ROUNDS host $(hostname)"
rocm-smi --showclocks --showtemp 2>/dev/null | head -12 > "$OUT/gpu-before.txt"
for r in $(seq 1 "$ROUNDS"); do
  if (( r % 2 )); then sample dflt "$r"; sample off "$r"; else sample off "$r"; sample dflt "$r"; fi
done
echo "== o4-ab done $(ts)"

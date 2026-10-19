#!/usr/bin/env bash
# P2' (MoeBlock) max-batch-1 serve soak on tatooine, one radiance-down window, both R9700s in parallel
# (docs/results/2026-10-19-p2prime-soak.md). Runs ON tatooine inside a fleetq lease the caller took:
#   fleetq acquire --host tatooine --takes-down radiance --mem 30 --label "..." --wait
#   FLEETQ_JOB=<id> ssh tatooine 'FLEETQ_JOB=<id> bash ~/intarsiaml-p2p/scripts/hip-decode/p2prime-soak.sh window <tag> <arm-gpu0> <arm-gpu1>'
#   fleetq release <id>      (only after this printed "radiance answers" and "only radiance is running")
# arm = blk (INTARSIA_HIP_MOE_BLOCK=1, P2') or off (INTARSIA_HIP_MOE_BLOCK=0, the unfused chain: the control). Both are set
# explicitly so the window means the same on the pre-flip and the post-flip binary. Each lane serves the 35B with the
# default --max-batch 1 on its own GPU/port/container, then tools/p2prime_soak.py drives it (see its docstring).
# The trap removes both containers, kills the host-side clients, restarts radiance and polls /v1/models (<= 3 min).
# Env: SOAK_S (600), SAMPLE (40), BURST_CAP (lane hard cap, s, default 1020), SOAK_RATE (0.3).
set -uo pipefail
export INTARSIA_REPO="${INTARSIA_REPO:-$HOME/intarsiaml-p2p}"
TARGET_VOL="${INTARSIA_TARGET_VOL:-intarsia-cargo-target-p2s0}"
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
M35=/models/Qwen3.6-35B-A3B-NVFP4
BIN=/target/release/intarsia
OUTROOT=$HOME/p2p-soak
now() { date '+%F %T'; }

lane() {  # lane <arm> <gpu> <port> <outdir>
  local arm=$1 gpu=$2 port=$3 out=$4 val name
  [ "$arm" = blk ] && val=1 || val=0
  name=soak-$arm-gpu$gpu
  mkdir -p "$out"
  local start; start=$(now)
  echo "[lane $arm gpu$gpu] starting $name INTARSIA_HIP_MOE_BLOCK=$val at $start" | tee "$out/lane.txt"
  docker rm -f "$name" >/dev/null 2>&1
  docker run -d --name "$name" --network host \
    --device /dev/kfd --device /dev/dri --group-add "$(stat -c %g /dev/kfd)" --group-add "$(stat -c %g /dev/dri/renderD128)" --group-add video \
    --ipc host --security-opt seccomp=unconfined --user "$(id -u):$(id -g)" -e HOME=/tmp -e GPU_MAX_HW_QUEUES=1 \
    -e HIP_VISIBLE_DEVICES="$gpu" -e INTARSIA_HIP_MOE_BLOCK="$val" \
    -v "$INTARSIA_REPO":/work -v /data/models/intarsia:/models:ro -v "$TARGET_VOL":/target -w /work \
    "$IMG" $BIN serve --backend hip --model $M35 --profile-every 0 --mem-budget 29 --no-supervisor \
    --host 0.0.0.0 --port "$port" --served-model-name qwen36-35b-intarsia-hip --max-seq 8192 >/dev/null
  timeout "${BURST_CAP:-1020}" python3 "$INTARSIA_REPO/tools/p2prime_soak.py" lane --url "http://127.0.0.1:$port" --arm "$arm" \
    --out "$out" --soak-s "${SOAK_S:-600}" --sample "${SAMPLE:-40}" --soak-rate "${SOAK_RATE:-0.3}" > "$out/lane.log" 2>&1
  echo $? > "$out/lane.rc"
  docker logs "$name" > "$out/serve.log" 2>&1
  docker inspect -f '{{.State.Status}} exit={{.State.ExitCode}} restarts={{.RestartCount}} oomkilled={{.State.OOMKilled}}' "$name" > "$out/container_state.txt" 2>&1
  journalctl -k --since "$start" --no-pager 2>&1 | grep -iE 'amdgpu|kfd|gpu|reset|timeout|fault|hang|ring' > "$out/kernel_gpu.log"
  docker rm -f "$name" >/dev/null 2>&1
  echo "[lane $arm gpu$gpu] done $(now) rc=$(cat "$out/lane.rc")" | tee -a "$out/lane.txt"
}

restart_radiance() {
  docker rm -f soak-blk-gpu0 soak-off-gpu0 soak-blk-gpu1 soak-off-gpu1 >/dev/null 2>&1
  pkill -f 'p2prime_soak.py|cbidentity.py|cbsoak.py' 2>/dev/null
  echo "[window] starting radiance $(now)" | tee -a "$LOG"
  systemctl --user start radiance
  local i
  for i in $(seq 1 36); do
    if curl -s -m 3 http://localhost:8080/v1/models | grep -q '"data":\['; then
      echo "[window] radiance answers $(now) (after $((i * 5)) s)" | tee -a "$LOG"
      echo "[window] containers now: $(docker ps --format '{{.Names}}' | tr '\n' ' ')" | tee -a "$LOG"
      return 0
    fi
    sleep 5
  done
  echo "[window] RADIANCE DID NOT ANSWER within 180 s $(now)" | tee -a "$LOG"
  return 1
}

if [ "${1:-}" = window ]; then
  tag=${2:?tag}; arm0=${3:?arm gpu0}; arm1=${4:?arm gpu1}
  OUT=$OUTROOT/$tag
  mkdir -p "$OUT"
  LOG=$OUT/window.log
  trap restart_radiance EXIT
  echo "[window] job ${FLEETQ_JOB:-none}; gpu0=$arm0 gpu1=$arm1 SOAK_S=${SOAK_S:-600}; stopping radiance $(now)" | tee "$LOG"
  systemctl --user stop radiance
  sleep 2
  echo "[window] radiance stopped $(now)" | tee -a "$LOG"
  lane "$arm0" 0 8090 "$OUT/gpu0-$arm0" &
  lane "$arm1" 1 8091 "$OUT/gpu1-$arm1" &
  wait
  echo "[window] lanes done $(now)" | tee -a "$LOG"
  exit 0
fi
echo "usage: $0 window <tag> <blk|off> <blk|off>"; exit 2

#!/usr/bin/env bash
# Lane fh, phase B: packed prefill default (HIP) against INTARSIA_PACK_PREFILL=0, ABAB x 2, tp=2 then tp=1, Qwen3.6-35B-A3B-NVFP4 on
# tatooine's R9700s. Runs on endor INSIDE a fleetq slot that takes radiance down (the two cards are radiance's):
#   J=$(fleetq acquire --host tatooine --takes-down radiance --mem 30 --expect 75 --label "hip pack default A/B (radiance down)" --wait)
#   FLEETQ_JOB=$J scripts/fh/window.sh <outdir>       # then: fleetq release $J
# Before the first run, build the branch on tatooine (CPU slot, no GPU): see scripts/fh/build.sh. Needs ~/intarsiaml-g2b-fh/g2b-intarsia.
# Arms: A = the build's default (packed prefill on), B = INTARSIA_PACK_PREFILL=0. Visits per tp: A B A B, a fresh server each.
# Per visit: localeval speed (1k prompt, 128 generated, 1 / 4 / 8 / 16 streams: aggregate tok/s, TTFT median) and ttft.py (per-stream
# TTFT p50 / p95 over 3 bursts per stream count; sha256 of each stream's greedy text, 1 stream = the greedy-ids check; packed-pass counters).
# Expected runtime: tp=2 ~4 min per visit, tp=1 ~3.5 min -> ~30 min of GPU time + ~10 min gates; about 50 min with radiance's stop / restore. Radiance is
# restored by an EXIT trap on every path; a deadman on tatooine starts it again after DEADMAN seconds (default 4500) if this script dies.
# Then (GATES=1, default) the 35B packed-prefill engine test and the HIP packed_rows / fp8_f8 kernel tests on card 0, ~10 min more; GATES=0 skips them.
# Knobs: TPS="2 1"  ARMS="A B A B"  SKIP_LE=1 (ttft.py only)  GATES=0.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq (see the header)}"
OUT=${1:?outdir}; mkdir -p "$OUT"
H=$(cd "$(dirname "$0")" && pwd); LE=/opt/code/deepeval
PORT=8296; NAME=intarsia; URL=http://tatooine:$PORT/v1
R='~/intarsiaml-g2b-fh'   # expanded by the remote shell
DEADMAN=${DEADMAN:-4500}
log() { echo "$(date +%T) $*" | tee -a "$OUT/steps.log"; }
rsh() { ssh -n -o BatchMode=yes tatooine "$@"; }
restore() {
  rsh "docker rm -f fh-serve >/dev/null 2>&1; [ -f ~/fh-deadman.pid ] && kill \$(cat ~/fh-deadman.pid) 2>/dev/null; rm -f ~/fh-deadman.pid"
  log "starting radiance"
  rsh "FLEETQ_JOB=$FLEETQ_JOB systemctl --user start radiance"
  for i in $(seq 1 36); do
    if rsh "curl -s -m 3 http://localhost:8080/v1/models" | grep -q '"data":\['; then log "radiance answers (after $((i * 5)) s)"; return 0; fi
    sleep 5
  done
  log "RADIANCE DID NOT ANSWER within 180 s"; return 1
}
trap restore EXIT
trap 'exit 130' INT TERM
rsh "systemctl --user status radiance --no-pager" > "$OUT/radiance-status-before.txt" 2>&1
rsh "nohup bash -c 'sleep $DEADMAN; systemctl --user start radiance' >/dev/null 2>&1 & echo \$! > ~/fh-deadman.pid"
log "job $FLEETQ_JOB: stopping radiance"
rsh "FLEETQ_JOB=$FLEETQ_JOB systemctl --user stop radiance" && sleep 3
start() { # start <tp> <arm>
  local gpus=0 extra="" env=()
  [[ $1 == 2 ]] && { gpus=0,1; extra="--tp 2 --devices 0,1"; }
  [[ $2 == B ]] && env=(-e INTARSIA_PACK_PREFILL=0)
  rsh "docker rm -f fh-serve >/dev/null 2>&1"
  rsh "docker run -d --name fh-serve --network host --device /dev/kfd --device /dev/dri \
    --group-add \$(stat -c %g /dev/kfd) --group-add \$(stat -c %g /dev/dri/renderD128) --group-add video \
    --ipc host --security-opt seccomp=unconfined --user \$(id -u):\$(id -g) -e HOME=/tmp -e GPU_MAX_HW_QUEUES=1 \
    -e HIP_VISIBLE_DEVICES=$gpus ${env[*]} -v $R:/work -v /data/models/intarsia:/models:ro -w /work \
    intarsia-hip-dev:rocm7.2.4-rust1.90.0 /work/g2b-intarsia serve --backend hip --model /models/Qwen3.6-35B-A3B-NVFP4 \
    --profile-every 0 --mem-budget 29 --host 0.0.0.0 --port $PORT --served-model-name $NAME --max-seq 8192 --max-batch 16 $extra >/dev/null"
  for i in $(seq 1 100); do rsh "curl -sf -m 2 http://localhost:$PORT/health" >/dev/null 2>&1 && return 0; sleep 3; done
  rsh "docker logs --tail 30 fh-serve" > "$OUT/serve-tp$1-$2-fail.log" 2>&1; return 1
}
v=0
for tp in ${TPS:-2 1}; do
  for arm in ${ARMS:-A B A B}; do
    v=$((v + 1)); L=tp$tp-v$v-$arm
    log "== $L"
    start $tp $arm || { log "$L: server did not come up"; continue; }
    if [[ ${SKIP_LE:-0} != 1 ]]; then
      (cd $LE && uv run localeval speed --url $URL --model $NAME --thinking off --sizes 1024 --gen 128 --reps 1 --concurrency 1,4,8,16 \
        --label fh-$L --no-unim --recipe "intarsia serve --backend hip tp=$tp arm=$arm (A default, B INTARSIA_PACK_PREFILL=0) --max-batch 16 --max-seq 8192" > "$OUT/$L-le.txt" 2>&1
       cp "$(ls -dt $LE/results/*fh-$L | head -1)/result.json" "$OUT/$L-le.json")
      grep -aE 'streams: aggregate' "$OUT/$L-le.txt" | tee -a "$OUT/steps.log"
    fi
    python3 "$H/ttft.py" --url $URL --model $NAME --out "$OUT/$L-ttft.json" 2>&1 | tee -a "$OUT/steps.log"
    rsh "docker logs fh-serve" > "$OUT/serve-$L.log" 2>&1
    rsh "docker rm -f fh-serve >/dev/null 2>&1"
  done
done
if [[ ${GATES:-1} == 1 ]]; then
  # bit-identity gates on card 0 (the binaries were compiled by build.sh, so this only runs them): the 35B engine test (24 requests, 8 and 16 slots,
  # packed prefill on, every request equal to the single-sequence engine) and the HIP kernel row-invariance tests the packed pass rests on
  log "== gates"
  rsh "cd $R && export INTARSIA_REPO=$R INTARSIA_TARGET_VOL=intarsia-cargo-target-g2b-fh HIP_VISIBLE_DEVICES=0 FLEETQ_JOB=$FLEETQ_JOB && timeout 1200 deploy/tatooine/dev.sh env INTARSIA_QWEN36_DIR=/models/Qwen3.6-35B-A3B-NVFP4 INTARSIA_BATCH_TEST_BACKEND=hip INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-models --test batch_cuda_real -- --ignored real_35b_packed_prefill_equals_the_local_engine --test-threads=1 --nocapture" > "$OUT/gate-35b-packed.log" 2>&1
  log "35b packed test exit $?: $(grep -aE '^test result|FAILED|panicked' $OUT/gate-35b-packed.log | tr '\n' ' ')"
  rsh "cd $R && export INTARSIA_REPO=$R INTARSIA_TARGET_VOL=intarsia-cargo-target-g2b-fh HIP_VISIBLE_DEVICES=0 FLEETQ_JOB=$FLEETQ_JOB && timeout 600 deploy/tatooine/dev.sh env INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-hip --test packed_rows --test fp8_f8 -- --test-threads=1" > "$OUT/gate-kernels.log" 2>&1
  log "kernel tests exit $?: $(grep -aE '^test result|FAILED|panicked' $OUT/gate-kernels.log | tr '\n' ' ')"
fi
log "all done"

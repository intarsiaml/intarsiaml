#!/usr/bin/env bash
# Phase 2b tensor-parallel gates. Runs ON tatooine, inside a fleetq lease that has radiance down and BOTH R9700s
# (see scripts/phase2b-burst.sh, which takes the lease, stops radiance, runs this, restarts radiance and waits for it).
#   scripts/phase2b-tp-remote.sh comm smoke plan accuracy bench
# Needs the release build from `deploy/tatooine/dev.sh cargo build --release -p intarsia-cli --features hip`
# and `cargo test --release -p intarsia-backend-hip --no-run` (CPU slot, no lease).
set -uo pipefail
export INTARSIA_REPO="${INTARSIA_REPO:-$HOME/intarsiaml-tp}" INTARSIA_TARGET_VOL="${INTARSIA_TARGET_VOL:-intarsia-cargo-target-tp}"
cd "$INTARSIA_REPO"
D=deploy/tatooine/dev.sh
B=/target/release/intarsia
M=/models/Qwen3-8B
G=/models/golden
ts() { date +%T; }
trap 'docker rm -f intarsia-serve >/dev/null 2>&1' EXIT

stage_comm() {
  echo "=== [$(ts)] comm: correctness (all-reduce vs host sum, exact; 9600 back-to-back launches; abort)"
  $D cargo test --release -p intarsia-backend-hip --test comm -- --nocapture --test-threads=1 2>&1 | tail -25
  echo "=== [$(ts)] comm: rate"
  $D cargo test --release -p intarsia-backend-hip --test comm -- --ignored --nocapture all_reduce_rate 2>&1 | grep -v "^warning" | tail -15
  echo "=== [$(ts)] comm: rate, drain=3"
  INTARSIA_HIP_AR_DRAIN=3 $D cargo test --release -p intarsia-backend-hip --test comm -- --ignored --nocapture all_reduce_rate 2>&1 | grep -v "^warning" | tail -15
}

gen() { # gen <label> <extra args...>
  local label=$1; shift
  echo "--- generate: $label"
  $D $B generate --backend hip --model $M -p "The capital of France is" --max-tokens 40 --show-ids "$@" 2>&1 | tail -8
}

stage_smoke() {
  echo "=== [$(ts)] smoke: greedy generation, tp=1 vs tp=2 (host comm) vs tp=2 (p2p)"
  HIP_VISIBLE_DEVICES=0 gen "tp=1"
  gen "tp=2 --comm host" --tp 2 --comm host
  gen "tp=2 (p2p)" --tp 2
}

stage_plan() {
  echo "=== [$(ts)] plan show --tp 2"
  $D $B plan show --backend hip --model $M --tp 2 2>&1 | tail -40
}

acc() { # acc <tp> <golden>
  local tp=$1 g=$2
  echo "--- accuracy tp=$tp $g"
  if [ "$tp" = 1 ]; then export HIP_VISIBLE_DEVICES=0; else unset HIP_VISIBLE_DEVICES; fi
  time $D $B accuracy --backend hip --model $M --tp "$tp" --golden "$G/$g" --mode both 2>&1 | tail -22
}

stage_accuracy() {
  echo "=== [$(ts)] accuracy tp=2"
  acc 2 qwen3-8b.safetensors
  acc 2 qwen3-8b-long.safetensors
}

stage_accuracy_bf16() {
  echo "=== [$(ts)] accuracy tp=2 with bf16 partials (INTARSIA_TP_PARTIAL=bf16: the in-place bf16 all-reduce)"
  export INTARSIA_TP_PARTIAL=bf16
  acc 2 qwen3-8b.safetensors
  acc 2 qwen3-8b-long.safetensors
  unset INTARSIA_TP_PARTIAL
}

stage_accuracy_tp1() {
  echo "=== [$(ts)] accuracy tp=1 (same build)"
  acc 1 qwen3-8b.safetensors
  acc 1 qwen3-8b-long.safetensors
}

serve_bench() { # serve_bench <tp> <gb-per-token> [extra serve args]
  local tp=$1 gb=$2; shift 2
  local name=qwen3-8b-tp$tp port=8090 vis=()
  [ "$tp" = 1 ] && vis=(-e HIP_VISIBLE_DEVICES=0)
  docker rm -f intarsia-serve >/dev/null 2>&1
  echo "--- serve tp=$tp [$(ts)]"
  docker run -d --name intarsia-serve --network host \
    --device /dev/kfd --device /dev/dri --group-add "$(stat -c %g /dev/kfd)" --group-add "$(stat -c %g /dev/dri/renderD128)" --group-add video \
    --ipc host --security-opt seccomp=unconfined --user "$(id -u):$(id -g)" -e HOME=/tmp -e GPU_MAX_HW_QUEUES=1 "${vis[@]}" \
    -v "$INTARSIA_REPO":/work -v /data/models/intarsia:/models:ro -v "$($D --target-vol)":/target -w /work \
    intarsia-hip-dev:rocm7.2.4-rust1.90.0 $B serve --backend hip --model $M --tp "$tp" "$@" \
    --host 0.0.0.0 --port $port --served-model-name $name --max-seq 8192 >/dev/null
  local up=0
  for i in $(seq 1 120); do
    if curl -s -m 2 http://127.0.0.1:$port/health 2>/dev/null | grep -q ok; then up=1; break; fi
    sleep 3
  done
  if [ "$up" != 1 ]; then echo "serve did not come up"; docker logs intarsia-serve 2>&1 | tail -30; docker rm -f intarsia-serve >/dev/null; return 1; fi
  echo "up after ~$((i*3))s"
  docker logs intarsia-serve 2>&1 | grep -i "admission\|tensor\|tp=" | head -5
  docker exec intarsia-serve $B bench speed --url http://127.0.0.1:$port --model $name \
    --sizes 128,1024,4096 --gen 128 --reps 3 --gb-per-token "$gb" --bandwidth-gbs 627 --tp "$tp" \
    --label "phase2b-qwen3-8b-tp$tp" --out-dir /work/bench-out --thinking off 2>&1 | tail -25
  docker rm -f intarsia-serve >/dev/null
}

stage_bench() {
  echo "=== [$(ts)] bench tp=1 then tp=2 (bytes read per token by the busiest rank: tp1 15.14 GB, tp2 8.19 GB)"
  serve_bench 1 15.14
  serve_bench 2 8.19
}

stage_perf() { # decode with per-op profile and the collective time, tp=2
  echo "=== [$(ts)] decode profile tp=2 (--profile-every 1)"
  $D $B generate --backend hip --model $M --tp 2 --profile-every 1 -p "The capital of France is" --max-tokens 64 2>&1 | tail -12
}

for s in "$@"; do
  echo "################ stage $s start $(ts)"
  "stage_$s"
  echo "################ stage $s end $(ts)"
done

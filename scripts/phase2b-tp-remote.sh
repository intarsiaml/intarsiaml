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
  echo "=== [$(ts)] comm: correctness and rate with the fence drain (INTARSIA_HIP_AR_DRAIN=1; the default is 3)"
  INTARSIA_HIP_AR_DRAIN=1 $D cargo test --release -p intarsia-backend-hip --test comm -- --nocapture --test-threads=1 2>&1 | grep -E "^test |test result"
  INTARSIA_HIP_AR_DRAIN=1 $D cargo test --release -p intarsia-backend-hip --test comm -- --ignored --nocapture all_reduce_rate 2>&1 | grep -v "^warning" | tail -9
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

acc() { # acc <tp> <golden> [extra accuracy args]
  local tp=$1 g=$2; shift 2
  echo "--- accuracy tp=$tp $g $*"
  if [ "$tp" = 1 ]; then
    time env HIP_VISIBLE_DEVICES=0 $D $B accuracy --backend hip --model $M --tp 1 --golden "$G/$g" --mode both "$@" 2>&1 | tail -22
  else
    time $D $B accuracy --backend hip --model $M --tp "$tp" --golden "$G/$g" --mode both "$@" 2>&1 | tail -22
  fi
}

stage_accuracy() {
  echo "=== [$(ts)] accuracy tp=2"
  acc 2 qwen3-8b.safetensors
  acc 2 qwen3-8b-long.safetensors
}

stage_accuracy_bf16() {
  echo "=== [$(ts)] accuracy tp=2 with bf16 partials (INTARSIA_TP_PARTIAL=bf16: the in-place bf16 all-reduce)"
  INTARSIA_TP_PARTIAL=bf16 acc 2 qwen3-8b.safetensors
  INTARSIA_TP_PARTIAL=bf16 acc 2 qwen3-8b-long.safetensors
}

stage_accuracy_self() { # tp=2 against tp=1's own logits (same weights, same prompts): how often do the top-1s differ
  echo "=== [$(ts)] self-golden: tp=1 writes its top-k log-probs, tp=2 is compared with them"
  mkdir -p "$INTARSIA_REPO/bench-out"
  for g in qwen3-8b qwen3-8b-long; do
    env HIP_VISIBLE_DEVICES=0 $D $B accuracy --backend hip --model $M --tp 1 --golden "$G/$g.safetensors" --mode prefill \
      --write-self-golden /work/bench-out/sg-$g-tp1.safetensors --allow-unstamped --allow-dirty 2>&1 | tail -4
    $D $B accuracy --backend hip --model $M --tp 2 --self-golden /work/bench-out/sg-$g-tp1.safetensors --mode prefill \
      --allow-unstamped --allow-dirty --allow-same-code 2>&1 | tail -16
  done
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
    if curl -sf -m 2 http://127.0.0.1:$port/health >/dev/null 2>&1; then up=1; break; fi
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

stage_perf() { # decode speed with / without the collectives, and the timed collective, tp=2
  local P="The capital of France is"
  echo "=== [$(ts)] decode, 192 tokens, no per-op profiling"
  echo "--- tp=1"; env HIP_VISIBLE_DEVICES=0 $D $B generate --backend hip --model $M -p "$P" --max-tokens 192 --ignore-eos --profile-every 0 2>&1 | grep -E "finish=|tensor parallel"
  echo "--- tp=2"; $D $B generate --backend hip --model $M --tp 2 -p "$P" --max-tokens 192 --ignore-eos --profile-every 0 2>&1 | grep -E "finish=|tensor parallel"
  echo "--- tp=2, collectives skipped (INTARSIA_TP_NO_COMM=1; garbage output, timing only)"
  INTARSIA_TP_NO_COMM=1 $D $B generate --backend hip --model $M --tp 2 -p "$P" --max-tokens 192 --ignore-eos --profile-every 0 2>&1 | grep -E "finish=|tensor parallel"
  echo "--- tp=2, per-op profile on every step"
  $D $B generate --backend hip --model $M --tp 2 -p "$P" --max-tokens 64 --ignore-eos --profile-every 1 2>&1 | grep -E "finish=|tensor parallel|steps="
}

for s in "$@"; do
  echo "################ stage $s start $(ts)"
  "stage_$s"
  echo "################ stage $s end $(ts)"
done

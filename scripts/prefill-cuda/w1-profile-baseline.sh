#!/usr/bin/env bash
# Window 1 (runs on spark1 INSIDE scripts/prefill-cuda-window.sh, vLLM is down): per-op profile of 35B prefill at ~1k and
# ~4k tokens (engine profiler + nsys), then the pre-change baselines and noise envelopes of the 35b-short and 35b-long
# accuracy cases from the base build (arch/plan tip 7e65770). Steps are skipped when the deadline is near.
#   BASE_BIN: build of 7e65770; PROF_BIN: base + `generate --op-table` (CLI-only change)
set -u
P=$HOME/intarsia-perf
BASE_BIN=$P/base/target/acc/release/intarsia
PROF_BIN=$P/p0/target/acc/release/intarsia
M=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
GD=$HOME/intarsiaml-golden
O=$P/out/w1; mkdir -p $O $P/art
DEADLINE=$(( $(date +%s) + ${W1_BUDGET_MIN:-28} * 60 ))
left() { echo $(( (DEADLINE - $(date +%s)) / 60 )); }
need() { # need <minutes> <what>
  if (( $(left) < $1 )); then echo "SKIP $2: $(left) min left, needs $1"; return 1; fi
  echo "== $2 ($(left) min left) $(date +%T)"; return 0
}
memcheck() { local gb=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1024 / 1024 )); echo "MemAvailable ${gb} GB"; (( gb >= $1 + 20 )) || { echo "ABORT: not enough memory"; exit 1; }; }
memcheck 38
nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv | tee $O/clocks.txt
GEN="--model $M --backend cuda --max-seq 8192 --mem-budget 45 --max-tokens 1 --temperature 0"

# prompt tokens | prefill chunk. 256 is the served default; the larger chunks show how much of prefill is weight traffic
# that a bigger forward pass amortises (MoE reads every expert once per chunk).
for cfg in 1024:256 4096:256 4096:1024 4096:2048 1024:1024; do
  n=${cfg%%:*}; ck=${cfg##*:}
  need 2 "engine profile prompt~$n chunk $ck" || continue
  $PROF_BIN generate $GEN --prefill-chunk $ck --profile-every 1 --op-table -p "$(cat $P/prompts/p$n.txt)" > $O/prof$n-c$ck.out 2> $O/prof$n-c$ck.err
  grep -E "finish=|op table|^[a-z_0-9]+ +[0-9.]+ " $O/prof$n-c$ck.err | head -40
  # unprofiled prefill time of the same prompt (profiling syncs after every op)
  $PROF_BIN generate $GEN --prefill-chunk $ck --profile-every 0 -p "$(cat $P/prompts/p$n.txt)" 2>&1 >/dev/null | grep -E "finish="
done

if command -v nsys >/dev/null; then
  for cfg in 4096:256 4096:2048; do
    n=${cfg%%:*}; ck=${cfg##*:}
    need 3 "nsys prompt~$n chunk $ck" || continue
    nsys profile -t cuda --force-overwrite true -o $O/nsys$n-c$ck $PROF_BIN generate $GEN --prefill-chunk $ck --profile-every 0 -p "$(cat $P/prompts/p$n.txt)" > $O/nsys$n-c$ck.out 2> $O/nsys$n-c$ck.err
    grep -E "finish=" $O/nsys$n-c$ck.err
    nsys stats -r cuda_gpu_kern_sum -f csv -o $O/nsys$n-c$ck $O/nsys$n-c$ck.nsys-rep > $O/nsys$n-c$ck.stats.log 2>&1 || tail -3 $O/nsys$n-c$ck.stats.log
    ls $O | grep nsys$n-c$ck
  done
fi

# --- pre-change baselines + envelopes (docs/accuracy.md "Noise envelope"), same flags as scripts/phase1b-acceptance.sh
acc() { # acc <name> <golden> <flags> <perturbations...>
  local name=$1 golden=$2 flags=$3; shift 3
  local A=$P/art/$name
  need $(( 3 + 3 * $# )) "accuracy $name baseline + $# envelope pass(es)" || return 0
  memcheck 60
  $BASE_BIN accuracy --model $M --backend cuda --golden $golden $flags --write-baseline $A.baseline.json --write-self-golden $A.self.safetensors > $O/$name.old.log 2>&1
  test -s $A.baseline.json || { echo "no baseline for $name"; return 0; }
  grep -E "floor verdict|golden gate|aggregate|prefill.*KL|decode.*KL" $O/$name.old.log | head -12
  local p
  for p in "$@"; do
    need 3 "envelope $name $p" || continue
    $BASE_BIN accuracy --model $M --backend cuda --golden $golden $flags --noise-ref $A.baseline.json --noise-perturb $p --write-noise-envelope $A.envelope.json > $O/$name.envelope-${p//[^a-z0-9=]/_}.log 2>&1
    test -s $A.envelope.json && echo "envelope $p written"
  done
}
acc 35b-short $GD/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors "--mode both --decode-positions 16 --max-seq 4096 --mem-budget 60" attn-simt chunk=64
echo "== w1 done $(date +%T), $(left) min left"

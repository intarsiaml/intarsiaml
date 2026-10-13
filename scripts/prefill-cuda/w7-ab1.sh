#!/usr/bin/env bash
# Window 7 (spark1, vLLM down, inside scripts/prefill-cuda-window.sh): the tests of the build under test (dev3 = e87beed),
# round 1 of the A/B ("ours" arms: old = 7e65770 at the shipped chunk 256; c2048 = this branch, default kernels, chunk 2048;
# fastc2048 = this branch with INTARSIA_CUDA_FAST_PREFILL=1, chunk 2048), then the accuracy gates of the c2048 configuration
# (default kernels), which is what the c2048 arm runs. Gates last: they are skipped when the window is nearly over.
set -u
P=$HOME/intarsia-perf
NEW=$P/dev3/target/acc/release/intarsia
OLD=$P/base/target/acc/release/intarsia
M=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
GD=$HOME/intarsiaml-golden
O=$P/out/w7; mkdir -p $O
DEADLINE=$(( $(date +%s) + ${W7_BUDGET_MIN:-21} * 60 ))
left() { echo $(( (DEADLINE - $(date +%s)) / 60 )); }
need() { if (( $(left) < $1 )); then echo "SKIP $2: $(left) min left, needs $1"; return 1; fi; echo "== $2 ($(left) min left) $(date +%T)"; return 0; }
gb=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1024 / 1024 )); echo "MemAvailable ${gb} GB"; (( gb >= 60 )) || { echo "ABORT: memory"; exit 1; }
PROV="--pre-change 7e65770 --revlist $P/revlist"

echo "== kernel tests of the build under test $(date +%T)"
( cd $P/dev3 && CARGO_TARGET_DIR=target/acc cargo test --release -p intarsia-backend-cuda --no-fail-fast --test gdn_lanes --test quantmoe --test admission -- --nocapture --test-threads=1 > $O/tests.log 2>&1; echo "exit $?" )
grep -E "^test .*(ok|FAILED)$|test result|panicked" $O/tests.log | cut -c1-170

AB_ARMS_ORDER="old:$OLD: c2048:$NEW:--prefill-chunk=2048 fastc2048:$NEW:--prefill-chunk=2048"
bash $P/scripts/ab-ours.sh 1 $AB_ARMS_ORDER

gate() { # gate <name> <golden> <flags> <min>
  local name=$1 golden=$2 flags=$3 mins=$4 A=$P/art/$1
  need $mins "gate $name, default kernels, chunk 2048" || return 0
  $NEW accuracy --model $M --backend cuda --golden $golden $flags --prefill-chunk 2048 --baseline $A.baseline.json --noise-envelope $A.envelope.json $PROV --allow-config-mismatch > $O/$name-c2048.log 2>&1
  echo "exit $?"
  grep -E "^prefill floor|^decode \(teacher-forced\) floor" $O/$name-c2048.log | sed -E 's/\| tail mass.*//; s/\(not gated for this class\) ok //' | cut -c1-120
  grep -E "(floor|baseline) verdict|prefix-reuse verdict|per-prompt gate: (FAIL|ok)|failing per-prompt" $O/$name-c2048.log | cut -c1-230
}
gate 35b-short $GD/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors "--mode both --decode-positions 16 --max-seq 4096 --mem-budget 60" 2
gate 35b-long $GD/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors "--mode all --decode-positions 64 --max-seq 16384 --mem-budget 70" 5
echo "== w7 done $(date +%T), $(left) min left"

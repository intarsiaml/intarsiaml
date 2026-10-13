#!/usr/bin/env bash
# Window 5 (spark1, vLLM down, inside scripts/prefill-cuda-window.sh). The first gate run (W4, build 9578fd9) failed the
# 35b-short per-prompt gate (chat_multiturn) and the 35b-long prefix-reuse gate at chunk 256: the new kernels served only
# part of the M / T range, so a prompt's rounding depended on how it was chunked (HEAD's gemm == mma bitwise invariant).
# d940701 makes both kernels serve every M >= 5 / every T. This window: tests + mutation checks of that build (dev2), then
# the gates; the bisect (kernels disabled one at a time on the W4 build, dev) runs only if a chunk-256 gate still fails.
set -u
P=$HOME/intarsia-perf
DEV=$P/dev2
BIN=$DEV/target/acc/release/intarsia
OLD9=$P/dev/target/acc/release/intarsia
M=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
GD=$HOME/intarsiaml-golden
O=$P/out/w5; mkdir -p $O
DEADLINE=$(( $(date +%s) + ${W5_BUDGET_MIN:-20} * 60 ))
left() { echo $(( (DEADLINE - $(date +%s)) / 60 )); }
need() { if (( $(left) < $1 )); then echo "SKIP $2: $(left) min left, needs $1"; return 1; fi; echo "== $2 ($(left) min left) $(date +%T)"; return 0; }
gb=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1024 / 1024 )); echo "MemAvailable ${gb} GB"; (( gb >= 60 )) || { echo "ABORT: memory"; exit 1; }
export CARGO_TARGET_DIR=target/acc
T="cargo test --release -p intarsia-backend-cuda"
PROV="--pre-change 7e65770 --revlist $P/revlist"

need 3 "kernel tests (dev2 = d940701)" && {
  cd $DEV
  $T --test gdn_lanes --test quantmoe --test attention --test admission -- --nocapture --test-threads=1 > $O/tests.log 2>&1
  echo "exit $?"; grep -E "^test .*(ok|FAILED)$|test result|panicked" $O/tests.log | cut -c1-170
  grep -E "gdn_delta_rule_bf16_lanes +[0-9]|matmul_fp8_gemm_f8 +[0-9]" $O/tests.log | cut -c1-170
}
need 2 "mutation checks (all three must FAIL)" && {
  (cd $P/mutg && $T --test gdn_lanes -- --nocapture --test-threads=1 > $O/mut-gdn.log 2>&1; echo "mutg (gdn lanes: no w decay) exit $? (expected non-zero)")
  grep -E "test result|panicked" $O/mut-gdn.log | cut -c1-170
  (cd $P/mutf && $T --test quantmoe matmul_fp8_gemm_f8 -- --nocapture --test-threads=1 > $O/mut-f8.log 2>&1; echo "mutf (f8 gemm: last k stage dropped) exit $? (expected non-zero)")
  grep -E "test result|panicked" $O/mut-f8.log | cut -c1-170
  (cd $P/mutr && $T --test quantmoe matmul_fp8_gemm_f8_rows -- --nocapture --test-threads=1 > $O/mut-rows.log 2>&1; echo "mutr (f8 gemm: 128-row tile rounds differently) exit $? (expected non-zero)")
  grep -E "test result|panicked" $O/mut-rows.log | cut -c1-170
}
cd $P
gate() { # gate <bin> <name> <golden> <flags> <extra> <tag>
  local bin=$1 name=$2 golden=$3 flags=$4 extra=$5 tag=$6 A=$P/art/$2
  need 4 "gate $name $tag" || return 0
  $bin accuracy --model $M --backend cuda --golden $golden $flags --baseline $A.baseline.json --noise-envelope $A.envelope.json $PROV $extra > $O/$name$tag.log 2>&1
  echo "exit $?"; grep -E "(floor|baseline) verdict|golden gate|prefix-reuse verdict|failing per-prompt" $O/$name$tag.log | cut -c1-230
}
SHORT="--mode both --decode-positions 16 --max-seq 4096 --mem-budget 60"
LONG="--mode all --decode-positions 64 --max-seq 16384 --mem-budget 70"
G_SHORT=$GD/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors
G_LONG=$GD/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors
gate $BIN 35b-short $G_SHORT "$SHORT" "--write-self-golden $P/art/35b-short.new2.safetensors" "-d2"
gate $BIN 35b-long $G_LONG "$LONG" "" "-d2"
if grep -q "baseline verdict[^:]*: FAIL\|prefix-reuse verdict: FAIL" $O/35b-short-d2.log $O/35b-long-d2.log 2>/dev/null; then
  echo "== a chunk-256 gate still fails: bisect on the W4 build (dev, 9578fd9), 35b-short, one new kernel disabled at a time"
  for dis in gdn_delta_rule_bf16_lanes matmul_fp8_gemm_f8 gdn_delta_rule_bf16_lanes,matmul_fp8_gemm_f8; do
    need 2 "bisect disable=$dis" || continue
    INTARSIA_CUDA_DISABLE=$dis $OLD9 accuracy --model $M --backend cuda --golden $G_SHORT $SHORT --baseline $P/art/35b-short.baseline.json --noise-envelope $P/art/35b-short.envelope.json $PROV > $O/bisect-${dis//,/+}.log 2>&1
    echo "exit $?"; grep -E "^prefill floor|(floor|baseline) verdict|failing per-prompt" $O/bisect-${dis//,/+}.log | sed -E 's/\| tail mass.*//' | cut -c1-200
  done
fi
gate $BIN 35b-short $G_SHORT "$SHORT --prefill-chunk 2048" "--allow-config-mismatch" "-d2-c2048"
gate $BIN 35b-long $G_LONG "$LONG --prefill-chunk 2048" "--allow-config-mismatch" "-d2-c2048"
need 2 "determinism: 35b-short dev2 vs its own dump, second process" && {
  $BIN accuracy --model $M --backend cuda --self-golden $P/art/35b-short.new2.safetensors $SHORT $PROV --allow-same-code --self-max-kl 1e-9 --self-max-pos-kl 1e-6 --self-max-flip-pct 0 --self-max-tail 1e-7 > $O/35b-short.det.log 2>&1
  echo "exit $?"; grep -E "self-golden verdict" $O/35b-short.det.log | cut -c1-200
}
echo "== w5 done $(date +%T), $(left) min left"

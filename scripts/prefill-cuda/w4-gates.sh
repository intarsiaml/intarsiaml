#!/usr/bin/env bash
# Window 4 (spark1, vLLM down, inside scripts/prefill-cuda-window.sh): the final build's kernel tests with mutation checks,
# the "after" profiles, and the 35B accuracy gates against the baselines + noise envelopes recorded by the pre-change build
# (W1/W2: ~/intarsia-perf/art, build ~/intarsia-perf/base = 7e65770). Gate flags are those of scripts/phase1b-acceptance.sh
# (stage_accuracy): the new build runs the full flags, the baseline came from the same flags with --mode both.
#   the chunk-2048 gates differ from the baseline in the prefill chunk, a hard config mismatch that --allow-config-mismatch
#   overrides; the envelopes contain chunk perturbations (35b-short chunk=64, 35b-long chunk=1024), the KL rule is absolute.
set -u
P=$HOME/intarsia-perf
DEV=$P/${DEV:-dev}
BIN=$DEV/target/acc/release/intarsia
M=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
GD=$HOME/intarsiaml-golden
O=$P/out/w4; mkdir -p $O
DEADLINE=$(( $(date +%s) + ${W4_BUDGET_MIN:-30} * 60 ))
left() { echo $(( (DEADLINE - $(date +%s)) / 60 )); }
need() { if (( $(left) < $1 )); then echo "SKIP $2: $(left) min left, needs $1"; return 1; fi; echo "== $2 ($(left) min left) $(date +%T)"; return 0; }
gb=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1024 / 1024 )); echo "MemAvailable ${gb} GB"; (( gb >= 60 )) || { echo "ABORT: memory"; exit 1; }
export CARGO_TARGET_DIR=target/acc
T="cargo test --release -p intarsia-backend-cuda"
PROV="--pre-change 7e65770 --revlist $P/revlist"

need 3 "kernel tests (final build)" && {
  cd $DEV
  $T --test gdn_lanes --test quantmoe --test attention --test admission -- --nocapture --test-threads=1 > $O/tests.log 2>&1
  echo "exit $?"; grep -E "^test .*(ok|FAILED)$|test result|panicked" $O/tests.log | cut -c1-170
  grep -E "gdn_delta_rule_bf16_lanes +[0-9]|matmul_fp8_gemm_f8 +[0-9]" $O/tests.log | cut -c1-170
}
need 2 "mutation checks (both must FAIL)" && {
  (cd $P/mutg && $T --test gdn_lanes -- --nocapture --test-threads=1 > $O/mut-gdn.log 2>&1; echo "mutg exit $? (expected non-zero)")
  grep -E "test result|panicked" $O/mut-gdn.log | cut -c1-170
  (cd $P/mutf && $T --test quantmoe matmul_fp8_gemm_f8 -- --nocapture --test-threads=1 > $O/mut-f8.log 2>&1; echo "mutf exit $? (expected non-zero)")
  grep -E "test result|panicked" $O/mut-f8.log | cut -c1-170
}
cd $P
GEN="--model $M --backend cuda --max-seq 8192 --mem-budget 45 --max-tokens 1 --temperature 0"
if command -v nsys >/dev/null; then
  for ck in 2048 256; do
    need 3 "nsys after, prompt~4096 chunk $ck" || continue
    nsys profile -t cuda --force-overwrite true -o $O/nsys4096-c$ck $BIN generate $GEN --prefill-chunk $ck --profile-every 0 -p "$(cat $P/prompts/p4096.txt)" > $O/nsys4096-c$ck.out 2> $O/nsys4096-c$ck.err
    grep -E "finish=" $O/nsys4096-c$ck.err | cut -c1-140
    nsys stats -r cuda_gpu_kern_sum -f csv -o $O/nsys4096-c$ck $O/nsys4096-c$ck.nsys-rep > $O/nsys4096-c$ck.stats.log 2>&1 || tail -3 $O/nsys4096-c$ck.stats.log
  done
fi
for cfg in 4096:2048 4096:256 1024:256; do
  n=${cfg%%:*}; ck=${cfg##*:}
  need 2 "engine op table after, prompt~$n chunk $ck" || continue
  $BIN generate $GEN --prefill-chunk $ck --profile-every 1 --op-table -p "$(cat $P/prompts/p$n.txt)" > $O/prof$n-c$ck.out 2> $O/prof$n-c$ck.err
  grep -E "finish=" $O/prof$n-c$ck.err | cut -c1-140
done

gate() { # gate <name> <golden> <flags> <extra> <tag>
  local name=$1 golden=$2 flags=$3 extra=$4 tag=$5 A=$P/art/$1
  need 4 "gate $name $tag" || return 0
  $BIN accuracy --model $M --backend cuda --golden $golden $flags --baseline $A.baseline.json --noise-envelope $A.envelope.json $PROV $extra --write-self-golden $A.new$tag.safetensors > $O/$name$tag.new.log 2>&1
  echo "exit $?"; grep -E "(floor|baseline) verdict|golden gate|prefix-reuse verdict|failing per-prompt" $O/$name$tag.new.log | cut -c1-230
  grep -E "^prefill |^decode|^prefill floor|^decode \(teacher" $O/$name$tag.new.log | head -4 | cut -c1-230
}
SHORT="--mode both --decode-positions 16 --max-seq 4096 --mem-budget 60"
LONG="--mode all --decode-positions 64 --max-seq 16384 --mem-budget 70"
G_SHORT=$GD/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors
G_LONG=$GD/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors
gate 35b-short $G_SHORT "$SHORT" "" ""
gate 35b-short $G_SHORT "$SHORT --prefill-chunk 2048" "--allow-config-mismatch" "-c2048"
need 3 "determinism: 35b-short new build vs its own dump, second process" && {
  $BIN accuracy --model $M --backend cuda --self-golden $P/art/35b-short.new.safetensors $SHORT $PROV --allow-same-code --self-max-kl 1e-9 --self-max-pos-kl 1e-6 --self-max-flip-pct 0 --self-max-tail 1e-7 > $O/35b-short.det.log 2>&1
  echo "exit $?"; grep -E "SELF-STATS|self-golden" $O/35b-short.det.log | cut -c1-200
}
gate 35b-long $G_LONG "$LONG" "" ""
gate 35b-long $G_LONG "$LONG --prefill-chunk 2048" "--allow-config-mismatch" "-c2048"
echo "== w4 done $(date +%T), $(left) min left"

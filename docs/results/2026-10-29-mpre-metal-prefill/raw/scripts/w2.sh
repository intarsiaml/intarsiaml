#!/bin/bash
# MPRE window 2: gates + speed for candidate c1 (d256 NAX flash + nvfp4 NAX GEMM) on the 35B. Inside ONE fleetq slot.
# base = c1 with INTARSIA_METAL_FLASH_NAX_D256=0 INTARSIA_METAL_NVFP4_NAX=0 (bitwise the previous build) for the self-golden;
# the speed A/B uses the real base binary (main 867b97145, Metal-identical to 7aebefb83).
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
L=/opt/code/intarsiaml/.worktrees/mpre/.lane; C=$L/intarsia-c1; B=$L/intarsia-base; OUT=$L/w2; mkdir -p $OUT
M=$HOME/models/Qwen3.6-35B-A3B-NVFP4; G=/opt/code/intarsiaml/golden
OFF="INTARSIA_METAL_FLASH_NAX_D256=0 INTARSIA_METAL_NVFP4_NAX=0"
/Users/josef/bin/macmon pipe -i 500 > $OUT/macmon.jsonl 2> /dev/null &
MM=$!
log() { echo "$(date +%T) $*" | tee -a $OUT/marks.txt; }
log "start rustc=$(pgrep -x rustc | wc -l) cargo=$(pgrep -x cargo | wc -l)"

T0=$SECONDS
late() { [ $((SECONDS - T0)) -gt 2160 ] && { log "SKIP $* (past 36 min)"; return 0; }; return 1; }
# 2. KL: base (knobs off) and candidate against the vLLM goldens, each writing a self-golden; then cand vs base self-golden
run() { local n=$1 g=$2 ms=$3 dp=$4
  env $OFF $C accuracy --model $M --backend metal --golden $G/$g --mode both --decode-positions $dp --max-seq $ms --allow-unstamped \
    --write-self-golden $OUT/$n.off.sg.safetensors --write-verdict $OUT/$n.off.verdict.json > $OUT/$n.off.log 2>&1; log "kl $n off exit $?"
  $C accuracy --model $M --backend metal --golden $G/$g --mode both --decode-positions $dp --max-seq $ms --allow-unstamped \
    --write-self-golden $OUT/$n.on.sg.safetensors --write-verdict $OUT/$n.on.verdict.json > $OUT/$n.on.log 2>&1; log "kl $n on exit $?"
  $C accuracy --model $M --backend metal --golden $G/$g --mode both --decode-positions $dp --max-seq $ms --allow-unstamped --allow-same-code \
    --self-golden $OUT/$n.off.sg.safetensors > $OUT/$n.self.log 2>&1; log "kl $n self exit $?: $(grep -a SELF-STATS $OUT/$n.self.log | tr '\n' ' ')"
}
late kl || run short qwen36-35b-a3b-nvfp4-vllm-top20.safetensors 4096 16
late kl || run long qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors 16384 16
late kl || run 4k qwen36-35b-a3b-nvfp4-vllm-4k-top20.safetensors 8192 64

# 3. greedy ids (cand): spec off == mtp k2 == prefill-chunk 512, 4 prompts, and run-to-run (repeat 2)
IP=/opt/code/intarsiaml/.worktrees/mpre/scripts/mpf/ids-prompts
for n in short-0 4k-0 4k-1 16k-0; do
  late ids && break
  case $n in short*) MS=4096;; 4k*) MS=8192;; *) MS=20480;; esac
  for arm in off mtp c512; do
    case $arm in off) A="--spec off --repeat 2";; mtp) A="--spec mtp --spec-k 2";; c512) A="--spec off --prefill-chunk 512";; esac
    $C generate -m $M --backend metal --max-seq $MS --chat --no-think --temperature 0 --profile-every 0 --ignore-eos --show-ids \
      -p "$(cat $IP/$n.txt)" --max-tokens 256 $A > /dev/null 2> $OUT/ids-$n-$arm.err
  done
  log "ids $n done"
done

# 4. model-level exactness tests (batched rows == solo, recurrent prefix reuse / packed prefill, long prompts), crate exactness,
#    PRC prefix-reuse gates on the 8B (one test per process)
D=$L/target/release/deps
bin() { ls $D/$1-* | grep -v '\.d$' | head -1; }
export INTARSIA_TEST_REAL_MODEL=1 INTARSIA_REQUIRE_GPU=1
for t in batch_recurrent batch_metal_real batch_metal; do
  late $t && continue
  $(bin $t) --test-threads=1 > $OUT/test-$t.log 2>&1; log "test $t exit $?: $(grep -E '^test result' $OUT/test-$t.log)"
done
late long || { $(bin batch_metal_real) --ignored real_35b_batched_engine_long_prompts_equal_the_local_engine > $OUT/test-long.log 2>&1; log "test long exit $?: $(grep -E '^test result' $OUT/test-long.log)"; }
for t in rows_exact row_tables; do
  late $t && continue
  $(bin $t) --test-threads=1 > $OUT/test-$t.log 2>&1; log "test $t exit $?: $(grep -E '^test result' $OUT/test-$t.log)"
done
for t in reuse_rule_resumes_with_cold_bits multi_turn_resumes_with_cold_bits engine_repeats_equal_the_cold_reference engine_multi_turn_equals_cold_reference; do
  late $t && continue
  INTARSIA_PROBE_DIR=$HOME/Models/Qwen3-8B $(bin prefix_reuse_real) --exact $t --nocapture > $OUT/prc-$t.log 2>&1; log "prc $t exit $?: $(grep -E '^test result' $OUT/prc-$t.log)"
done
unset INTARSIA_TEST_REAL_MODEL INTARSIA_REQUIRE_GPU

# 5. plan show: base vs c1 vs c1 knobs off, every Metal model on disk (G5); full kernels test on the 35B shapes (G6)
mkdir -p $OUT/plans
for m in Qwen3.6-35B-A3B-NVFP4 Qwen3-8B Llama-3.1-8B-Instruct Qwen2.5-7B-Instruct Qwen3.8-27B-Quark-AWQ-MXFP4 MiniCPM-V-4.7-35B-A3B-MLX-4bit Kolibri-1-MLX-4bit; do
  late plans && break
  $B plan show --backend metal --model $HOME/Models/$m --max-seq 20000 > $OUT/plans/$m.base.txt 2>&1
  $C plan show --backend metal --model $HOME/Models/$m --max-seq 20000 > $OUT/plans/$m.c1.txt 2>&1
  env $OFF $C plan show --backend metal --model $HOME/Models/$m --max-seq 20000 > $OUT/plans/$m.c1off.txt 2>&1
  log "plan $m: c1 $(cmp -s $OUT/plans/$m.base.txt $OUT/plans/$m.c1.txt && echo same || echo DIFF), c1off $(cmp -s $OUT/plans/$m.base.txt $OUT/plans/$m.c1off.txt && echo same || echo DIFF)"
done
late kt || { $C kernels test --backend metal --model $M > $OUT/kt-35b.txt 2>&1; log "kernels test 35B exit $?: $(grep -cE 'ADMITTED' $OUT/kt-35b.txt) admitted, $(grep -cE 'REJECT|FAIL' $OUT/kt-35b.txt) rejected"; }
# warm 1k cb trace of base (why 1k is flat): generate --repeat 2, the two runs split by the gap
late cb || { env INTARSIA_METAL_PROFILE=cb:$OUT/cb-base-1k.jsonl $B generate --model $M --backend metal --max-seq 20000 --temperature 0 --max-tokens 2 \
  --ignore-eos --profile-every 0 --repeat 2 --prompt "$(cat /opt/code/intarsiaml/.worktrees/mpre/docs/results/raw/2026-10-14-metal-phase-c/prompt-1k.txt)" > /dev/null 2> $OUT/cb-base-1k.err; log "cb 1k: $(grep -o 'prefill [0-9]* ms ([0-9.]* tok/s)' $OUT/cb-base-1k.err | tr '\n' ' ')"; }
log "end rustc=$(pgrep -x rustc | wc -l) cargo=$(pgrep -x cargo | wc -l)"
kill $MM; gzip -f $OUT/*.jsonl

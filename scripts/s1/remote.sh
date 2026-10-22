#!/usr/bin/env bash
# Lane S1, spark2 side. Run inside a fleetq slot (FLEETQ_JOB set), from the Mac:
#   ssh -n spark2 "export FLEETQ_JOB=$FLEETQ_JOB STEPS='probe tests'; bash ~/intarsia-decfuse/s1-dev/scripts/s1/remote.sh"
# Steps (STEPS, space separated; default all non-window steps in this order):
#   tests   cargo test of the new kernels (fp8_gemm_reg, nvfp4_gemm_reg, lib pins), CARGO_TARGET_DIR=target/test
#   bench   GEMM microbenchmark at the 35B prefill shapes: every matmul_fp8_gemm_reg schedule (INTARSIA_CUDA_FP8_REG_CFG
#           0 1 4 7), then the new nvfp4 / pair / GLU kernels against the calls they replace (quantmoe bench_wq_prefill_shapes)
#   ktest   intarsia kernels test --backend cuda (admission, dev binary)
#   prof    nsys over a ~4k-token 35B prefill per arm (kernel ranking + launch gaps, rank.txt)
#   ids     greedy ids per arm vs base: 35B ~1.2k-token (256 tokens) and ~3.4k-token (64 tokens) prompts, 8B the same; plans
#   acc     35B KL21 short + long goldens per arm (must reproduce base's digits), prefix-reuse gate incl. chat_reuse_4k
# Arms (ARMS, space separated name:tree[:ENV=V,ENV=V]): default 'base:s1-base dev:s1-dev cub:s1-dev:INTARSIA_CUDA_FP8_CUBLAS=1'.
# Output: ~/intarsia-decfuse/out/s1/<step>/...
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
P=$HOME/intarsia-decfuse
O=$P/out/${OUTNAME:-s1}; mkdir -p $O
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
M8=/home/nvidia/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
GD=${GOLDEN_DIR:-$HOME/intarsiaml-golden-s1}
S=$GD/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors; L=$GD/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors
CORPUS=$P/s1-dev/tools/prompts/corpus_docs.txt
STEPS=${STEPS:-tests bench ktest prof ids acc}
want() { [[ " $STEPS " == *" $1 "* ]]; }
read -r -a ARMV <<< "${ARMS:-base:s1-base dev:s1-dev cub:s1-dev:INTARSIA_CUDA_FP8_CUBLAS=1}"
arm_tree() { echo "$1" | cut -d: -f2; }
arm_env() { local e; e=$(echo "$1" | cut -s -d: -f3); echo "${e:-INTARSIA_X=0}" | tr ',' ' '; }
arm_name() { echo "$1" | cut -d: -f1; }
bin() { echo $P/$1/target/acc/release/intarsia; }
memgb() { echo $(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1024 / 1024 )); }
log() { echo "$(date +%T) $*" | tee -a $O/steps.log; }
log "== remote.sh STEPS='$STEPS' MemAvailable $(memgb) GB"
nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu --format=csv,noheader >> $O/clocks.txt 2>&1
for t in s1-base s1-dev; do log "bin $t: $($(bin $t) --version 2>&1 | head -1); pair syms $(grep -c ik_matmul_fp8_pair_gemm_reg $(bin $t)); glu syms $(grep -c ik_matmul_nvfp4_glu_gemm $(bin $t))"; done

if want tests; then
  d=$O/tests; mkdir -p $d; cd $P/s1-dev
  CT="CARGO_BUILD_JOBS=8 CARGO_TARGET_DIR=target/test INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-cuda"
  for t in "--lib" "--test fp8_gemm_reg" "--test nvfp4_gemm_reg" "--test quantmoe"; do
    eval "env $CT $t -- --test-threads=1" < /dev/null > "$d/test-${t//[ -]/}.log" 2>&1
    log "test $t exit $?: $(grep -E 'test result|panicked' "$d/test-${t//[ -]/}.log" | head -3 | tr '\n' '|')"
  done
  eval "env INTARSIA_CUDA_FP8_CUBLAS=1 $CT --test fp8_gemm_reg -- --nocapture --test-threads=1" < /dev/null > "$d/test-fp8reg-cublas.log" 2>&1
  log "test fp8_gemm_reg (cuBLAS on) exit $?: $(grep -E 'cases bitwise|test result|panicked' "$d/test-fp8reg-cublas.log" | head -3 | tr '\n' '|')"
  # every gemm_reg schedule must be bitwise matmul_fp8_gemm (and the pair two gemm_reg calls) under that schedule
  for c in ${BENCH_CFGS:-0 1 4 7}; do
    eval "env INTARSIA_CUDA_FP8_REG_CFG=$c $CT --test fp8_gemm_reg -- --test-threads=1" < /dev/null > "$d/test-fp8reg-cfg$c.log" 2>&1
    log "test fp8_gemm_reg cfg $c exit $?: $(grep -E 'test result|panicked' "$d/test-fp8reg-cfg$c.log" | head -3 | tr '\n' '|')"
  done
fi

if want bench; then
  d=$O/bench; mkdir -p $d; cd $P/s1-dev
  B="CARGO_BUILD_JOBS=8 CARGO_TARGET_DIR=target/test INTARSIA_REQUIRE_GPU=1 INTARSIA_BENCH_M=${BENCH_M:-2048,1024} INTARSIA_BENCH_REPS=${BENCH_REPS:-5} cargo test --release -p intarsia-backend-cuda --test quantmoe -- --ignored --nocapture --test-threads=1 bench_wq_prefill_shapes"
  for c in ${BENCH_CFGS:-0 1 4 7}; do
    eval "env INTARSIA_CUDA_FP8_REG_CFG=$c INTARSIA_BENCH_WQ=matmul_fp8_gemm_reg $B" < /dev/null > $d/cfg$c.log 2>&1
    log "bench cfg $c exit $?: $(grep -E 'per-chunk total' $d/cfg$c.log | tr -s ' ' | tr '\n' '|')"
  done
  eval "env INTARSIA_CUDA_FP8_CUBLAS=1 INTARSIA_BENCH_WQ=matmul_bf16,matmul_fp8_gemm_cublas,matmul_nvfp4_gemm,matmul_nvfp4_gemm_reg,matmul_fp8_pair_gemm_reg,matmul_nvfp4_glu_gemm $B" < /dev/null > $d/new.log 2>&1
  log "bench new exit $?"; grep -E 'per-chunk total|pair|glu' $d/new.log | tee -a $O/steps.log
fi

if want ktest; then
  d=$O/ktest; mkdir -p $d
  timeout 1800 $(bin s1-dev) kernels test --backend cuda < /dev/null > $d/dev.log 2>&1; rc=$?
  log "kernels test dev exit $rc: $(grep -aE 'FAIL|REJECT|kernels,|cases' $d/dev.log | tail -3 | tr '\n' '|')"
  grep -aE 'matmul_fp8_pair_gemm_reg|matmul_nvfp4_gemm_reg|matmul_nvfp4_glu_gemm' $d/dev.log | head -8 | tee -a $O/steps.log
fi

if want prof; then
  NSYS=$(ls /opt/nvidia/nsight-systems*/*/bin/nsys /usr/local/cuda/bin/nsys 2>/dev/null | head -1)
  for a in "${ARMV[@]}"; do
    t=$(arm_tree $a); n=$(arm_name $a)
    for chars in ${CHARS:-14000}; do
      d=$O/prof-$n-$chars; rm -rf $d; mkdir -p $d; head -c $chars $CORPUS > $d/prompt.txt
      (( $(memgb) >= 34 )) || { log "prof ABORT: memory $(memgb) GB"; break 2; }
      env $(arm_env $a) $NSYS profile -t cuda --force-overwrite true -o $d/nsys $(bin $t) generate -m $M35 --backend cuda --mem-budget 29 --max-seq 20480 \
        --prefill-chunk 2048 --profile-every 0 --max-tokens 1 -p "$(cat $d/prompt.txt)" < /dev/null > $d/nsys.out 2> $d/nsys.err
      log "prof $n $chars: $(grep -aE 'finish=' $d/nsys.err | head -1 | cut -c1-200)"
      $NSYS stats -r cuda_gpu_kern_sum,cuda_gpu_trace -f csv --force-export=true -o $d/kern $d/nsys.nsys-rep < /dev/null > $d/stats.log 2>&1 || tail -3 $d/stats.log
      rm -f $d/nsys.sqlite
      python3 $P/s1-dev/scripts/s1/rank.py $d/kern_cuda_gpu_kern_sum.csv $d/kern_cuda_gpu_trace.csv > $d/rank.txt 2>&1
      head -30 $d/rank.txt | tee -a $O/steps.log
    done
  done
fi

gen() { # gen <tag> <tree> <model> <prompt chars> <tokens> <env words>
  local tag=$1 tree=$2 m=$3 chars=$4 n=$5; shift 5
  env "$@" timeout 900 $(bin $tree) generate -m $m --backend cuda --mem-budget 30 --max-seq 20480 --chat --no-think --profile-every 0 --ignore-eos --show-ids \
    -p "$(head -c $chars $CORPUS)" --max-tokens $n < /dev/null > $O/ids/gen-$tag.out 2> $O/ids/gen-$tag.err
  grep -a '^ids:' $O/ids/gen-$tag.err > $O/ids/gen-$tag.ids
  log "gen $tag: $(grep -aE 'finish=' $O/ids/gen-$tag.err | head -1 | cut -c1-160)"
}
if want ids; then
  mkdir -p $O/ids
  for a in "${ARMV[@]}"; do
    t=$(arm_tree $a); n=$(arm_name $a); e=$(arm_env $a)
    gen 35s-$n $t $M35 5000 256 $e; gen 35l-$n $t $M35 14000 64 $e
    gen 8s-$n $t $M8 5000 256 $e; gen 8l-$n $t $M8 14000 64 $e
    env $e timeout 300 $(bin $t) plan show --backend cuda --model $M8 --max-seq 4096 --mem-budget 30 --json < /dev/null > $O/ids/plan8-$n.json 2> $O/ids/plan8-$n.err
    env $e timeout 300 $(bin $t) plan show --backend cuda --model $M35 --max-seq 20480 --mem-budget 30 < /dev/null > $O/ids/plan35-$n.txt 2> $O/ids/plan35-$n.err
  done
  b=$(arm_name ${ARMV[0]})
  for a in "${ARMV[@]:1}"; do
    n=$(arm_name $a)
    for p in 35s 35l 8s 8l; do
      if cmp -s $O/ids/gen-$p-$b.ids $O/ids/gen-$p-$n.ids && [[ -s $O/ids/gen-$p-$b.ids ]]; then r=identical; else r=DIFFER; fi
      log "ids $p $b vs $n: $r ($(tr ',' '\n' < $O/ids/gen-$p-$n.ids | wc -l) ids)"
    done
    cmp -s $O/ids/plan8-$b.json $O/ids/plan8-$n.json && log "8B plan $b vs $n: byte-identical ($(wc -c < $O/ids/plan8-$n.json) B)" || log "8B plan $b vs $n: DIFFER"
  done
fi

acc() { # acc <tag> <tree> <golden> <env words>
  local tag=$1 tree=$2 g=$3; shift 3; local t0=$(date +%s)
  env "$@" timeout 900 $(bin $tree) accuracy --model $M35 --backend cuda --golden $g --mode prefill --max-seq 16384 --mem-budget 40 \
    --prefill-chunk 2048 --allow-dirty < /dev/null > $O/acc/acc-$tag.log 2>&1
  log "acc $tag exit $? $(( $(date +%s) - t0 )) s: $(grep -aE '^prefill floor|KL21|top-1' $O/acc/acc-$tag.log | head -2 | cut -c1-160 | tr '\n' '|')"
}
reuse() { # reuse <tag> <tree> <env words>
  local tag=$1 tree=$2; shift 2; local t0=$(date +%s)
  env "$@" timeout 900 $(bin $tree) accuracy --model $M35 --backend cuda --golden $L --mode reuse --max-seq 16384 --mem-budget 40 --prefill-chunk 2048 --allow-dirty < /dev/null > $O/acc/reuse-$tag.log 2>&1
  log "reuse $tag exit $? $(( $(date +%s) - t0 )) s: $(grep -aE '^chat_reuse_4k|prefix-reuse gate' $O/acc/reuse-$tag.log | cut -c1-90 | tr -s ' ' | tr '\n' '|')"
}
if want acc; then
  mkdir -p $O/acc
  (( $(memgb) >= 44 )) || log "acc WARNING: MemAvailable $(memgb) GB"
  for a in "${ARMV[@]}"; do
    t=$(arm_tree $a); n=$(arm_name $a); e=$(arm_env $a)
    acc short-$n $t $S $e; acc long-$n $t $L $e; reuse $n $t $e
  done
fi
log "== remote.sh done"

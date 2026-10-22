#!/usr/bin/env bash
# Lane S1, spark2 side. Run inside a fleetq slot (FLEETQ_JOB set), from the Mac:
#   ssh -n spark2 "export FLEETQ_JOB=$FLEETQ_JOB STEPS='probe tests'; bash ~/intarsia-decfuse/s1-dev/scripts/s1/remote.sh"
# Steps (STEPS, space separated; default all non-window steps in this order):
#   probe   f32x2 packed-fp32 probe (kernels/cuda_checks/f32x2_probe.cu): SASS + throughput + bitwise
#   tests   cargo test of the new kernels (fp8_gemm_reg, nvfp4_gemm_reg, lib pins), CARGO_TARGET_DIR=target/test
#   bench   GEMM microbenchmark at the 35B prefill shapes: every matmul_fp8_gemm_reg schedule (INTARSIA_CUDA_FP8_REG_CFG
#           0..6), then the new nvfp4 / pair / GLU kernels against the calls they replace (quantmoe bench_wq_prefill_shapes)
#   ktest   intarsia kernels test --backend cuda (admission, dev binary)
#   prof    nsys over a ~4k-token 35B prefill, base and dev (kernel ranking + launch gaps, rank.txt / gaps.txt)
#   ids     greedy ids base vs dev: 35B ~1.5k-token (256 tokens) and ~4k-token (64 tokens) prompts, 8B the same; 8B plan
#   acc     35B KL21 short + long goldens (dev must reproduce base's digits), prefix-reuse gate incl. chat_reuse_4k
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
STEPS=${STEPS:-probe tests bench ktest prof ids acc}
want() { [[ " $STEPS " == *" $1 "* ]]; }
bin() { echo $P/$1/target/acc/release/intarsia; }
memgb() { echo $(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1024 / 1024 )); }
log() { echo "$(date +%T) $*" | tee -a $O/steps.log; }
log "== remote.sh STEPS='$STEPS' MemAvailable $(memgb) GB"
nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu --format=csv,noheader >> $O/clocks.txt 2>&1
for t in s1-base s1-dev; do log "bin $t: $($(bin $t) --version 2>&1 | head -1); pair syms $(grep -c ik_matmul_fp8_pair_gemm_reg $(bin $t)); glu syms $(grep -c ik_matmul_nvfp4_glu_gemm $(bin $t))"; done

if want probe; then
  d=$O/probe; mkdir -p $d; cd $P/s1-dev
  nvcc -O3 -arch=sm_121 -o $d/f32x2_probe kernels/cuda_checks/f32x2_probe.cu > $d/nvcc.log 2>&1; log "probe nvcc exit $? ($(tail -2 $d/nvcc.log | tr '\n' ' '))"
  if [[ -x $d/f32x2_probe ]]; then
    cuobjdump -sass $d/f32x2_probe > $d/sass.txt 2>&1
    log "probe SASS: FADD2/FFMA2/FMUL2 lines $(grep -cE 'FADD2|FFMA2|FMUL2|F32x2' $d/sass.txt), FADD $(grep -cE ' FADD ' $d/sass.txt), FFMA $(grep -cE ' FFMA ' $d/sass.txt)"
    timeout 120 $d/f32x2_probe < /dev/null > $d/run.txt 2>&1; log "probe run exit $?"; cat $d/run.txt | tee -a $O/steps.log
  fi
  # does nvcc already emit packed fp32 in the GDN split kernel?
  nvcc -O3 -arch=sm_121 -cubin -I kernels/cuda -o $d/gdn.cubin kernels/cuda/gdn.cu > $d/gdn-nvcc.log 2>&1 && cuobjdump -sass $d/gdn.cubin > $d/gdn-sass.txt 2>&1
  log "gdn SASS: packed lines $(grep -cE 'FADD2|FFMA2|FMUL2|F32x2' $d/gdn-sass.txt 2>/dev/null), FFMA $(grep -cE ' FFMA ' $d/gdn-sass.txt 2>/dev/null)"
fi

if want tests; then
  d=$O/tests; mkdir -p $d; cd $P/s1-dev
  CT="CARGO_BUILD_JOBS=8 CARGO_TARGET_DIR=target/test INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-cuda"
  for t in "--lib" "--test fp8_gemm_reg" "--test nvfp4_gemm_reg" "--test quantmoe"; do
    eval "env $CT $t -- --test-threads=1" < /dev/null > "$d/test-${t//[ -]/}.log" 2>&1
    log "test $t exit $?: $(grep -E 'test result|panicked' "$d/test-${t//[ -]/}.log" | head -3 | tr '\n' '|')"
  done
  # every gemm_reg schedule must be bitwise matmul_fp8_gemm (and the pair two gemm_reg calls) under that schedule
  for c in ${BENCH_CFGS:-0 1 2 3 4 5 6}; do
    eval "env INTARSIA_CUDA_FP8_REG_CFG=$c $CT --test fp8_gemm_reg -- --test-threads=1" < /dev/null > "$d/test-fp8reg-cfg$c.log" 2>&1
    log "test fp8_gemm_reg cfg $c exit $?: $(grep -E 'test result|panicked' "$d/test-fp8reg-cfg$c.log" | head -3 | tr '\n' '|')"
  done
fi

if want bench; then
  d=$O/bench; mkdir -p $d; cd $P/s1-dev
  B="CARGO_BUILD_JOBS=8 CARGO_TARGET_DIR=target/test INTARSIA_REQUIRE_GPU=1 INTARSIA_BENCH_M=${BENCH_M:-2048,1024} INTARSIA_BENCH_REPS=${BENCH_REPS:-5} cargo test --release -p intarsia-backend-cuda --test quantmoe -- --ignored --nocapture --test-threads=1 bench_wq_prefill_shapes"
  for c in ${BENCH_CFGS:-0 1 2 3 4 5 6}; do
    eval "env INTARSIA_CUDA_FP8_REG_CFG=$c INTARSIA_BENCH_WQ=matmul_fp8_gemm_reg $B" < /dev/null > $d/cfg$c.log 2>&1
    log "bench cfg $c exit $?: $(grep -E 'per-chunk total' $d/cfg$c.log | tr -s ' ' | tr '\n' '|')"
  done
  eval "env INTARSIA_BENCH_WQ=matmul_nvfp4_gemm,matmul_nvfp4_gemm_reg,matmul_fp8_pair_gemm_reg,matmul_nvfp4_glu_gemm $B" < /dev/null > $d/new.log 2>&1
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
  for t in ${PROF_TREES:-s1-base s1-dev}; do
    for chars in ${CHARS:-14000}; do
      d=$O/prof-$t-$chars; rm -rf $d; mkdir -p $d; head -c $chars $CORPUS > $d/prompt.txt
      (( $(memgb) >= 34 )) || { log "prof ABORT: memory $(memgb) GB"; break 2; }
      $NSYS profile -t cuda --force-overwrite true -o $d/nsys $(bin $t) generate -m $M35 --backend cuda --mem-budget 29 --max-seq 20480 \
        --prefill-chunk 2048 --profile-every 0 --max-tokens 1 -p "$(cat $d/prompt.txt)" < /dev/null > $d/nsys.out 2> $d/nsys.err
      log "prof $t $chars: $(grep -aE 'finish=|prompt' $d/nsys.err | head -1 | cut -c1-200)"
      $NSYS stats -r cuda_gpu_kern_sum,cuda_gpu_trace -f csv --force-export=true -o $d/kern $d/nsys.nsys-rep < /dev/null > $d/stats.log 2>&1 || tail -3 $d/stats.log
      rm -f $d/nsys.sqlite
      python3 $P/s1-dev/scripts/s1/rank.py $d/kern_cuda_gpu_kern_sum.csv $d/kern_cuda_gpu_trace.csv > $d/rank.txt 2>&1
      head -28 $d/rank.txt | tee -a $O/steps.log
    done
  done
fi

gen() { # gen <tag> <tree> <model> <prompt chars> <tokens> [env]
  local tag=$1 tree=$2 m=$3 chars=$4 n=$5 e=${6:-INTARSIA_X=0}
  env $e timeout 900 $(bin $tree) generate -m $m --backend cuda --mem-budget 30 --max-seq 20480 --chat --no-think --profile-every 0 --ignore-eos --show-ids \
    -p "$(head -c $chars $CORPUS)" --max-tokens $n < /dev/null > $O/ids/gen-$tag.out 2> $O/ids/gen-$tag.err
  grep -a '^ids:' $O/ids/gen-$tag.err > $O/ids/gen-$tag.ids
  log "gen $tag: $(grep -aE 'finish=' $O/ids/gen-$tag.err | head -1 | cut -c1-160)"
}
if want ids; then
  mkdir -p $O/ids
  for t in s1-base s1-dev; do
    gen 35s-$t $t $M35 5000 256; gen 35l-$t $t $M35 14000 64
    gen 8s-$t $t $M8 5000 256; gen 8l-$t $t $M8 14000 64
    timeout 300 $(bin $t) plan show --backend cuda --model $M8 --max-seq 4096 --mem-budget 30 --json < /dev/null > $O/ids/plan8-$t.json 2> $O/ids/plan8-$t.err
    timeout 300 $(bin $t) plan show --backend cuda --model $M35 --max-seq 20480 --mem-budget 30 < /dev/null > $O/ids/plan35-$t.txt 2> $O/ids/plan35-$t.err
  done
  [[ -n "${IDS_EXTRA_ENV:-}" ]] && { gen 35s-dev-off s1-dev $M35 5000 256 "$IDS_EXTRA_ENV"; gen 35l-dev-off s1-dev $M35 14000 64 "$IDS_EXTRA_ENV"; }
  for p in 35s 35l 8s 8l; do
    if cmp -s $O/ids/gen-$p-s1-base.ids $O/ids/gen-$p-s1-dev.ids && [[ -s $O/ids/gen-$p-s1-base.ids ]]; then r=identical; else r=DIFFER; fi
    log "ids $p base vs dev: $r ($(tr ',' '\n' < $O/ids/gen-$p-s1-dev.ids | wc -l) ids)"
  done
  cmp -s $O/ids/plan8-s1-base.json $O/ids/plan8-s1-dev.json && log "8B plan: byte-identical ($(wc -c < $O/ids/plan8-s1-dev.json) B)" || log "8B plan: DIFFER"
fi

acc() { # acc <tag> <tree> <golden> [env]
  local tag=$1 tree=$2 g=$3 e=${4:-INTARSIA_X=0}; local t0=$(date +%s)
  env $e timeout 900 $(bin $tree) accuracy --model $M35 --backend cuda --golden $g --mode prefill --max-seq 16384 --mem-budget 40 \
    --prefill-chunk 2048 --allow-dirty < /dev/null > $O/acc/acc-$tag.log 2>&1
  log "acc $tag exit $? $(( $(date +%s) - t0 )) s: $(grep -aE '^prefill floor|KL21|top-1' $O/acc/acc-$tag.log | head -2 | cut -c1-160 | tr '\n' '|')"
}
reuse() { # reuse <tag> <tree> [env]
  local tag=$1 tree=$2 e=${3:-INTARSIA_X=0}; local t0=$(date +%s)
  env $e timeout 900 $(bin $tree) accuracy --model $M35 --backend cuda --golden $L --mode reuse --max-seq 16384 --mem-budget 40 --prefill-chunk 2048 --allow-dirty < /dev/null > $O/acc/reuse-$tag.log 2>&1
  log "reuse $tag exit $? $(( $(date +%s) - t0 )) s: $(grep -aE '^chat_reuse_4k|prefix-reuse gate' $O/acc/reuse-$tag.log | cut -c1-90 | tr -s ' ' | tr '\n' '|')"
}
if want acc; then
  mkdir -p $O/acc
  (( $(memgb) >= 44 )) || log "acc WARNING: MemAvailable $(memgb) GB"
  for t in ${ACC_TREES:-s1-dev s1-base}; do acc short-$t $t $S; acc long-$t $t $L; reuse $t $t; done
fi
log "== remote.sh done"

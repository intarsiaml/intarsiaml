#!/usr/bin/env bash
# One small-GPU slot on spark1 (vLLM UP and idle; relative numbers only): sync + build this working tree as
# ~/intarsia-decfuse/$TREE (default r5-dev) with the CUDA test binaries, then run the steps named in STEPS (comma separated):
#   res     registers / shared memory / spills of the split-KV kernels (cuobjdump of attention_splitkv.o)
#   sass    SASS op counts per split-KV instantiation (FFMA FADD FMUL SHFL MUFU LDG LDS LDGSTS)
#   probe   kernels/cuda_checks/kv_stream.cu (DRAM read rate: plain stream vs the split-KV load pattern) at CAPS
#   att     tests/attention.rs attention_throughput (--ignored; one cache, repeated)
#   skv     tests/splitkv_pipe.rs (bitwise vs the base kernel), then its timing test at the 35B shape (10 rotating layers)
#   race    the racecheck wiring: gpu-test.sh $TREE --test gdn_split (tests + racecheck) and the mutation harness on
#           MUTS_RACE (default: no-loop-sync and one test-killed mutant)
#   unit    the backend's lib tests (registry pins) under each env in UNIT_ENVS
#   fi      read-only: FlashInfer decode planning arguments in the vLLM container's attention backend
#   fleetq run --host spark1 --mem 12 --label "cuda-round5 dev" --wait-timeout 60 -- scripts/cuda-round5/dev.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$S/../.." && pwd)"
HOST=${HOST:-spark1}
TREE=${TREE:-r5-dev}
STEPS=${STEPS:-res,probe,att}
R='~/intarsia-decfuse'
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
has() { [[ ",$STEPS," == *",$1,"* ]]; }
CT="CARGO_BUILD_JOBS=6 CARGO_TARGET_DIR=target/acc INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-cuda"
OBJ="d=\$(ls -td target/acc/release/build/intarsia-backend-cuda-*/out | head -1)"
if [[ "${NOBUILD:-0}" != 1 ]]; then "$REPO/scripts/cuda-decode/build.sh" "$TREE" --tests 2>&1 | tail -6; fi
rsh "nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu --format=csv"
if has res; then
  echo "== resources (attention_splitkv.o)"
  rsh "cd $R/$TREE && $OBJ && cuobjdump --dump-resource-usage \$d/attention_splitkv.o 2>&1 | grep -E 'Function|REG' | sed -E 's/_ZN[0-9A-Za-z_]*attention_splitkv_cu_[0-9a-f]+[0-9]+//' | paste - - | grep -E \"${RES_FILTER:-.}\" | sed -E 's/EEEvPK13__nv_bfloat16[^:]*:/:/; s/ (STACK|CONSTANT|TEXTURE|SURFACE|SAMPLER)[^ ]*//g' | cut -c1-200"
fi
if has sass; then
  echo "== SASS op counts (attention_splitkv.o)"
  rsh "cd $R/$TREE && $OBJ && cuobjdump -sass \$d/attention_splitkv.o | awk '/Function :/{if(f)print f, c[\"FFMA\"]+0, c[\"FADD\"]+0, c[\"FMUL\"]+0, c[\"SHFL\"]+0, c[\"MUFU\"]+0, c[\"LDG\"]+0, c[\"LDS\"]+0, c[\"LDGSTS\"]+0, n; f=\$3; delete c; n=0} /\\/\\*[0-9a-f]{4}\\*\\//{n++; if(match(\$0,/(FFMA|FADD|FMUL|SHFL|MUFU|LDGSTS|LDG|LDS)/)) c[substr(\$0,RSTART,RLENGTH)]++} END{print f, c[\"FFMA\"]+0, c[\"FADD\"]+0, c[\"FMUL\"]+0, c[\"SHFL\"]+0, c[\"MUFU\"]+0, c[\"LDG\"]+0, c[\"LDS\"]+0, c[\"LDGSTS\"]+0, n}' | sed -E 's/_ZN[0-9A-Za-z_]*attention_splitkv_cu_[0-9a-f]+[0-9]+//' | cut -c1-150; echo '(columns: FFMA FADD FMUL SHFL MUFU LDG LDS LDGSTS all)'"
fi
if has probe; then
  echo "== kv_stream probe"
  rsh "cd $R/$TREE && mkdir -p target/probe && nvcc -O3 -gencode arch=compute_121,code=sm_121 kernels/cuda_checks/kv_stream.cu -o target/probe/kv_stream && for c in ${CAPS:-1024 4096 16384 32768}; do target/probe/kv_stream \$c; done"
fi
if has att; then
  echo "== attention_throughput (one cache, repeated)"
  rsh "cd $R/$TREE && $CT --test attention -- --ignored --nocapture --test-threads=1 attention_throughput 2>&1 | grep -E 'decode|verify|panicked'"
fi
if has skv; then
  # SKV_ENVS: space-separated env sets for the pipelined kernels ("d" = default; "A=1,B=2" = several), e.g.
  #   "d INTARSIA_SPLITKV_STAGES=2 INTARSIA_SPLITKV_HS=1"
  for e in ${SKV_ENVS:-d}; do
    [[ $e == d ]] && e="INTARSIA_X=0"
    echo "== splitkv_pipe tests, ${e//,/ }"
    rsh "cd $R/$TREE && env ${e//,/ } $CT --test splitkv_pipe --test graph_pos --test attention -- --nocapture --test-threads=1 2>&1 | grep -E '^test |panicked|differ|cases|result:' | tail -24"
  done
  for round in $(seq 1 "${ROUNDS:-1}"); do
    for e in ${SKV_ENVS:-d}; do
      [[ $e == d ]] && e="INTARSIA_X=0"
      echo "== splitkv timing round $round, ${e//,/ }"
      rsh "cd $R/$TREE && env ${e//,/ } $CT --test splitkv_pipe -- --ignored --nocapture --test-threads=1 splitkv_timing 2>&1 | grep -E ' us |panicked'"
    done
  done
fi
if has race; then
  echo "== gpu-test.sh $TREE --test gdn_split (+ racecheck)"
  "$REPO/scripts/cuda-decode/gpu-test.sh" "$TREE" --test gdn_split -- --test-threads=1 2>&1 | grep -E '^test |result:|RACE|racecheck|hazard' | tail -12
  echo "gpu-test.sh exit ${PIPESTATUS[0]}"
  echo "== mutation harness (scripts/cuda-prefill4/mutcheck.sh), racecheck step"
  BASE_TREE=$TREE MUT_TREE=r5-mut NOBUILD=1 "$REPO/scripts/cuda-prefill4/mutcheck.sh" ${MUTS_RACE:-no-loop-sync v-zero}
  echo "mutcheck exit $?"
fi
if has unit; then
  for e in ${UNIT_ENVS:-INTARSIA_X=0}; do
    echo "== lib tests, env ${e//,/ }"
    rsh "cd $R/$TREE && env ${e//,/ } CARGO_BUILD_JOBS=6 CARGO_TARGET_DIR=target/acc cargo test --release -p intarsia-backend-cuda --lib 2>&1 | grep -E '^test |panicked|result:' | tail -16"
  done
fi
if has kq; then
  # `intarsia kernels test --backend cuda --quick` on the 35B and 8B shapes: the exit status and the summary lines
  M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
  M8=/home/nvidia/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
  for m in $M35 $M8; do
    echo "== kernels test --backend cuda --quick -m $(basename $(dirname $(dirname $m)))"
    rsh "cd $R/$TREE && target/acc/release/intarsia kernels test --backend cuda --quick -m $m > /tmp/kq_\$\$.out 2>&1; echo \"exit \$?\"; grep -cE ' ADMITTED\$' /tmp/kq_\$\$.out | sed 's/^/ADMITTED: /'; grep -cE 'NO CASES' /tmp/kq_\$\$.out | sed 's/^/NO CASES: /'; grep -cE ' REJECTED\$' /tmp/kq_\$\$.out | sed 's/^/REJECTED: /'; grep -E 'incomplete:|kernels, |^Error|SKIPPED' /tmp/kq_\$\$.out | cut -c1-160; rm -f /tmp/kq_\$\$.out"
  done
fi
if has fi; then
  echo "== FlashInfer decode planning in the vLLM container (read-only)"
  rsh "C=missioncntrl-guidance; F=\$(docker exec \$C python3 -c 'import vllm.v1.attention.backends.flashinfer as m; print(m.__file__)' 2>/dev/null); echo \"file \$F\"; docker exec \$C grep -nE 'split_kv|fixed_split|disable_split|use_tensor_cores|num_splits|BatchDecodeWithPagedKVCache|decode_backend|kv_cache_dtype|page_size' \$F | head -40 | cut -c1-200; docker exec \$C python3 -c 'import flashinfer, os; print(os.path.dirname(flashinfer.__file__))'; echo '-- split-size branch:'; docker exec \$C sed -n 636,653p \$F"
fi

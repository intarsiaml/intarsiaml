#!/usr/bin/env bash
# One small-GPU slot on spark1 (vLLM UP; relative numbers only): sync + build this working tree as ~/intarsia-decfuse/$TREE
# (default p4-dev) with the CUDA test binaries, then run the steps named in STEPS (comma separated):
#   gdn    tests gdn_split (bitwise vs the rows kernel) under INTARSIA_GDN_SPLIT_LPR=2, 4, 8; gdn_lanes; gdn_timing per LPR
#   res    register / shared-memory use of the GDN kernels (cuobjdump of gdn.o)
#   sass   the split kernel's SASS op counts per instantiation (FFMA / FADD / FMUL / SHFL / LDS)
#   unit   the backend's lib tests (registry selection pins) under each gate in UNIT_ENVS
#   wqt    the fp8 kernel tests (WQ_TESTS, default: --test fp8_gemm_reg --test quantmoe)
#   wq     bench_wq_prefill_shapes for each setting in WQ_CFGS (";"-separated "ENV=v ..." sets, "" = default)
#   fleetq run --host spark1 --mem 12 --label "prefill-4 dev" --wait-timeout 60 -- scripts/cuda-prefill4/dev.sh
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST=${HOST:-spark1}
TREE=${TREE:-p4-dev}
STEPS=${STEPS:-gdn,res}
R='~/intarsia-decfuse'
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
has() { [[ ",$STEPS," == *",$1,"* ]]; }
CT="CARGO_BUILD_JOBS=6 CARGO_TARGET_DIR=target/acc INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-cuda"
if [[ "${NOBUILD:-0}" != 1 ]]; then "$S/../cuda-decode/build.sh" "$TREE" --tests 2>&1 | tail -12; fi
rsh "nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu --format=csv"
if has gdn; then
  for lpr in 2 4 8; do
    echo "== gdn_split, INTARSIA_GDN_SPLIT_LPR=$lpr"
    rsh "cd $R/$TREE && INTARSIA_GDN_SPLIT_LPR=$lpr $CT --test gdn_split -- --nocapture --test-threads=1 2>&1 | grep -E '^test |panicked|differs|result:' | tail -20"
  done
  echo "== gdn_lanes"
  rsh "cd $R/$TREE && $CT --test gdn_lanes -- --test-threads=1 2>&1 | grep -E '^test |panicked|result:' | tail -10"
  for round in $(seq 1 "${ROUNDS:-1}"); do
    for lpr in ${LPRS:-2 4 8}; do
      echo "== gdn_timing round $round, INTARSIA_GDN_SPLIT_LPR=$lpr (lanes kernel at its default LPR 4)"
      rsh "cd $R/$TREE && INTARSIA_GDN_SPLIT_LPR=$lpr $CT --test gdn_timing -- --nocapture --test-threads=1 2>&1 | grep -E 'us/launch|panicked'"
    done
  done
fi
if has res; then
  echo "== resources (gdn.o)"
  rsh "cd $R/$TREE && d=\$(ls -td target/acc/release/build/intarsia-backend-cuda-*/out | head -1) && cuobjdump --dump-resource-usage \$d/gdn.o 2>&1 | grep -E 'Function|REG' | sed -E 's/_ZN[0-9A-Za-z_]*gdn_cu_[0-9a-f]+[0-9]+//' | cut -c1-160"
fi
if has sass; then
  echo "== SASS op counts (gdn.o)"
  rsh "cd $R/$TREE && d=\$(ls -td target/acc/release/build/intarsia-backend-cuda-*/out | head -1) && cuobjdump -sass \$d/gdn.o | awk '/Function :/{if(f)print f, c[\"FFMA\"]+0, c[\"FADD\"]+0, c[\"FMUL\"]+0, c[\"SHFL\"]+0, c[\"LDS\"]+0, n; f=\$3; delete c; n=0} /\\/\\*[0-9a-f]{4}\\*\\//{n++; if(match(\$0,/(FFMA|FADD|FMUL|SHFL|LDS)/)) c[substr(\$0,RSTART,RLENGTH)]++} END{print f, c[\"FFMA\"]+0, c[\"FADD\"]+0, c[\"FMUL\"]+0, c[\"SHFL\"]+0, c[\"LDS\"]+0, n}' | sed -E 's/_ZN[0-9A-Za-z_]*gdn_cu_[0-9a-f]+[0-9]+//' | cut -c1-120; echo '(columns: FFMA FADD FMUL SHFL LDS instructions)'"
fi
if has unit; then
  for e in ${UNIT_ENVS:-INTARSIA_X=0}; do
    echo "== lib tests, env ${e//,/ }"
    rsh "cd $R/$TREE && env ${e//,/ } CARGO_BUILD_JOBS=6 CARGO_TARGET_DIR=target/acc cargo test --release -p intarsia-backend-cuda --lib 2>&1 | grep -E '^test |panicked|result:' | tail -14"
  done
fi
if has wqt; then
  for e in ${WQT_ENVS:-INTARSIA_X=0}; do
    echo "== wq tests, env ${e//,/ }"
    rsh "cd $R/$TREE && env ${e//,/ } $CT ${WQ_TESTS:---test fp8_gemm_reg --test quantmoe} -- --nocapture --test-threads=1 2>&1 | grep -E '^test |rel_l2|differ|panicked|assert|result:' | tail -60"
  done
fi
if has wq; then
  IFS=';' read -r -a cfgs <<< "${WQ_CFGS:-}"
  [[ ${#cfgs[@]} -eq 0 ]] && cfgs=("")
  for round in $(seq 1 "${ROUNDS:-1}"); do
    for cfg in "${cfgs[@]}"; do
      echo "== wq bench round $round: ${cfg:-default}"
      rsh "cd $R/$TREE && BIN=\$(ls -t target/acc/release/deps/quantmoe-* | grep -v '\\.d\$' | head -1) && env $cfg INTARSIA_REQUIRE_GPU=1 \$BIN --ignored --nocapture bench_wq_prefill_shapes 2>&1 | grep -E '^M='"
    done
  done
fi

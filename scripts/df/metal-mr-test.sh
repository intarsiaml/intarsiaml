#!/bin/zsh
# Lane df: Metal M-row MXFP4 GEMV gates + timing in one exclusive slot: the bitwise tests (tests/mxfp4_gemv.rs, all), one planted
# fault (MUT=1: the shader's t chain starts at 1e-30 instead of 0; must FAIL), and the kernel bench.
#   fleetq run --host endor --exclusive --mem 30 --expect 20 --label "..." -- zsh scripts/df/metal-mr-test.sh <out dir>
set -u
O=${1:?out dir}; mkdir -p $O; R=${0:A:h:h:h}; cd $R
K=kernels/metal/mxfp4.metal
[[ -n "${FLEETQ_JOB:-}" ]] || { echo "not inside a fleetq slot" >&2; exit 2; }
say() { echo "$(date +%T) $*" | tee -a $O/steps.log; }
cargo test --release -p intarsia-backend-metal --test mxfp4_gemv -- --nocapture > $O/test-mxfp4_gemv.log 2>&1
say "mxfp4_gemv tests exit $?: $(grep -aE '^test |test result|cases bitwise' $O/test-mxfp4_gemv.log | tr '\n' '|')"
if [[ ${MUT:-1} == 1 ]]; then
  K=kernels/metal/mxfp4.metal; cp $K $O/mxfp4.metal.orig
  # planted fault: the last weight row of each simdgroup sums a word's 8 products in reverse order (other rounding, same values)
  sed -i '' 's/for (int b = 0; b < 8; ++b) t\[q\]\[m\] = fma(wf\[q\]\[b\], xf\[b\], t\[q\]\[m\]);/for (int b = 0; b < 8; ++b) { const int bb = q == P - 1 ? 7 - b : b; t[q][m] = fma(wf[q][bb], xf[bb], t[q][m]); }/' $K
  if cmp -s $K $O/mxfp4.metal.orig; then say "MUT not applied"; else
    cargo test --release -p intarsia-backend-metal --test mxfp4_gemv m_row_shaders -- --nocapture > $O/test-mut1.log 2>&1
    rc=$?; say "planted fault 1 (last weight row of a simdgroup sums each word in reverse order): exit $rc ($([[ $rc != 0 ]] && echo CAUGHT || echo MISSED)) $(grep -aoE 'mr p[0-9] [^:]*: N [0-9]+ K [0-9]+ M [0-9]+' $O/test-mut1.log | head -1)"
  fi
  cp $O/mxfp4.metal.orig $K
  # planted fault 2: the M-row shader's activation row m reads row m - 1 for m >= 2 (a row-stride slip)
  sed -i '' 's|((device const uint4\*)(xsrc + (ulong)m \* K + (ulong)u \* 32))\[c\]|((device const uint4*)(xsrc + (ulong)(m >= 2 ? m - 1 : m) * K + (ulong)u * 32))[c]|' $K
  if cmp -s $K $O/mxfp4.metal.orig; then say "MUT2 not applied"; else
    cargo test --release -p intarsia-backend-metal --test mxfp4_gemv m_row_shaders -- --nocapture > $O/test-mut2.log 2>&1
    rc=$?; say "planted fault 2 (row stride slip from row 2): exit $rc ($([[ $rc != 0 ]] && echo CAUGHT || echo MISSED))"
  fi
  cp $O/mxfp4.metal.orig $K; touch $K
  cargo build --release -p intarsia-backend-metal > /dev/null 2>&1
fi
DFBENCH_NO_LOGITS=1 cargo test --release -p intarsia-backend-metal --test df_verify_bench -- --ignored --nocapture > $O/bench.log 2>&1
say "bench exit $?"; grep DFBENCH $O/bench.log | cut -c1-260 >> $O/steps.log
say done

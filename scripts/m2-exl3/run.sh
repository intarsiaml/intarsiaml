#!/usr/bin/env bash
# M2 (EXL3) CUDA checks on spark1 with vLLM UP, inside one fleetq slot:
#   fleetq run --host spark1 --mem 8 --label "M2 exl3 cuda build+test (vLLM up)" -- scripts/m2-exl3/run.sh
# STEPS (comma list, default build,admit,tests,bench): build (rsync + cargo build of the CLI and the cuda test binaries
# via scripts/cuda-decode/build.sh), admit (`intarsia kernels test --backend cuda --op matmul_exl3`, quick and full),
# tests (cargo test --test exl3 / quantmoe / admission), bench (the ignored decode microbenchmark of tests/exl3.rs),
# prefill (the ignored prefill microbenchmark bench_prefill_shapes, GEMV vs GEMM vs bf16), race (compute-sanitizer racecheck on one exl3 test). TREE (default m2-exl3) is ~/intarsia-decfuse/<TREE> on the box.
# Raw output goes to docs/results/2026-10-21-m2-exl3/raw/cuda-*.log. Starts no server and takes no role down.
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$S/../.." && pwd)"
HOST=${HOST:-spark1}
PFX=${PFX:-cuda}
TREE=${TREE:-m2-exl3}
STEPS=${1:-${STEPS:-build,admit,tests,bench}}
OUT="$REPO/docs/results/2026-10-21-m2-exl3/raw"
R='~/intarsia-decfuse'
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
mkdir -p "$OUT"
rsh() { ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
has() { [[ ",$STEPS," == *",$1,"* ]]; }
CARGO="cd $R/$TREE && CARGO_BUILD_JOBS=6 CARGO_TARGET_DIR=target/acc INTARSIA_REQUIRE_GPU=1"

if has build; then
  echo "== build ($TREE)"
  "$S/../cuda-decode/build.sh" "$TREE" --tests 2>&1 | tee "$OUT/${PFX}-build.log" | tail -15
fi
if has admit; then
  echo "== kernels test --backend cuda --op matmul_exl3 (quick, full)"
  rsh "cd $R/$TREE && nvidia-smi --query-gpu=name,clocks.sm,temperature.gpu --format=csv,noheader; target/acc/release/intarsia kernels test --backend cuda --op matmul_exl3 --quick -v" 2>&1 | tee "$OUT/${PFX}-admission-quick.log" | tail -25 || true
  rsh "cd $R/$TREE && target/acc/release/intarsia kernels test --backend cuda --op matmul_exl3 -v" 2>&1 | tee "$OUT/${PFX}-admission-full.log" | tail -25 || true
fi
if has tests; then
  echo "== cargo test: exl3, quantmoe, admission"
  set +e
  rsh "$CARGO cargo test --release -p intarsia-backend-cuda --test exl3 --test quantmoe --test admission --no-fail-fast -- --nocapture --test-threads=1" 2>&1 | tee "$OUT/${PFX}-tests.log" | grep -E "^(test |running|test result|worst|[0-9]+ cases|x=|bits=|error|SKIPPED)|panicked|FAILED" | tail -80
  echo "(cargo test exit: ${PIPESTATUS[0]}; the full output is in cuda-tests.log)"
  set -e
fi
if has bench; then
  echo "== bench_decode_shapes"
  rsh "$CARGO cargo test --release -p intarsia-backend-cuda --test exl3 -- --ignored --nocapture --test-threads=1 bench_decode_shapes" 2>&1 | tee "$OUT/${PFX}-bench.log" | grep -E "^(exl3|bf16|nvfp4)"
fi
if has prefill; then
  echo "== bench_prefill_shapes"
  rsh "$CARGO EXL3_BENCH_STAT=${EXL3_BENCH_STAT:-min} ${PREFILL_ENV:-} cargo test --release -p intarsia-backend-cuda --test exl3 -- --ignored --nocapture --test-threads=1 bench_prefill_shapes" 2>&1 | tee "$OUT/${PFX}-prefill.log" | grep -E "^(\(cold|matmul_)"
fi
if has one; then
  # profile_one_shape (tests/exl3.rs) once per entry of VARIANTS (";"-separated env assignments, e.g. "EXL3_PROF_M=512;EXL3_PROF_M=1024"),
  # fastest of 15 cold replays each; the test binary is the one `build` made
  echo "== profile_one_shape variants"
  IFS=';' read -ra VS <<< "${VARIANTS:-EXL3_PROF_M=1024}"
  for v in "${VS[@]}"; do
    rsh "cd $R/$TREE; B=\$(ls -t target/acc/release/deps/exl3-* | grep -v '\\.d\$' | head -1); env INTARSIA_REQUIRE_GPU=1 EXL3_BENCH_STAT=min $v ${NSYS:+/usr/local/bin/nsys profile -t cuda --stats=true -f true -o /tmp/m2nsys} \$B --ignored --exact profile_one_shape --test-threads=1 --nocapture 2>&1 | grep -E '^(matmul|.*us\$)|rot_in|gemm_kernel|linear_kernel|PROF'" 2>&1 | tee -a "$OUT/${PFX}-one.log"
  done
fi
if has race; then
  echo "== racecheck (exl3 f16_activations_and_outputs)"
  rsh "cd $R/$TREE; B=\$(ls -t target/acc/release/deps/exl3-* | grep -v '\\.d\$' | head -1); INTARSIA_REQUIRE_GPU=1 timeout 900 compute-sanitizer --tool racecheck --racecheck-report all \$B --test-threads=1 --exact f16_activations_and_outputs 2>&1 | tail -15" 2>&1 | tee "$OUT/${PFX}-racecheck.log"
fi
if has memcheck; then
  # out-of-bounds / misaligned accesses: the clamped rows and the cp.async of the GEMM (f16 test: both kernels, K = 384)
  for t in f16_activations_and_outputs a_bad_codebook_gives_nan; do
    echo "== memcheck ($t)"
    rsh "cd $R/$TREE; B=\$(ls -t target/acc/release/deps/exl3-* | grep -v '\\.d\$' | head -1); INTARSIA_REQUIRE_GPU=1 timeout 900 compute-sanitizer --tool memcheck \$B --test-threads=1 --exact $t 2>&1 | tail -8" 2>&1 | tee -a "$OUT/${PFX}-memcheck.log"
  done
fi
echo "== done"

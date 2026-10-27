#!/usr/bin/env bash
# Lane W4A8, window W1, ON tatooine inside ONE fleetq slot that takes radiance down (--takes-down radiance --quiet gpu,cpu):
# correctness / admission / row-exactness / mutants / first kernel timings of the W4A8 kernels. Builds come first (build.sh,
# mutants.sh: cpu jobs). Logs: ~/intarsiaml-w4a8-logs/w1. The trap starts radiance again and waits until it answers.
#   card 0: mxfp4_a8 (clean, then INTARSIA_HIP_MXFP4_A8_PERGROUP=1), rows_exact (mxfp4), quick_coverage, admission; then timings
#   card 1: kernels test --quick (the whole HIP registry), then the mutants (each MUST fail)
#   usage: w1.sh [sections: A (card 0 tests) K (card 1 kernels test) M (mutants) T (timings)]  default all
set -uo pipefail
SECTIONS=${*:-A K M T}
want() { [[ " $SECTIONS " == *" $1 "* ]]; }
O=$HOME/intarsiaml-w4a8-logs/w1; mkdir -p "$O"
R=$HOME/intarsiaml-w4a8; V=intarsia-cargo-w4a8; D=$R/deploy/tatooine/dev.sh
now() { date '+%F %T'; }
restore() {
  systemctl --user start radiance
  for i in $(seq 1 48); do curl -s -m 3 http://localhost:8080/v1/models | grep -q '"data":\[' && break; sleep 5; done
  echo "[w4a8] radiance $(systemctl --user is-active radiance) after $((i * 5)) s; models: $(curl -s -m 3 http://localhost:8080/v1/models | head -c 100)"
}
trap restore EXIT
systemctl --user stop radiance; sleep 3; echo "[w4a8] radiance $(systemctl --user is-active radiance) $(now)"
dev() { # dev <gpu> <cmd...>
  local g=$1; shift
  INTARSIA_REPO=$R INTARSIA_TARGET_VOL=$V HIP_VISIBLE_DEVICES=$g $D "$@" < /dev/null
}
F='^test |^test result|FAILED|panicked|^error|G-B|SKIPPED|rel_l2|differ'
if want A || want K || want M; then
  if want A; then (
    echo "-- mxfp4_a8 (fold path)"; dev 0 env INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-hip --test mxfp4_a8 -- --test-threads=1 2>&1 | grep -aE "$F"
    echo "-- mxfp4_a8 INTARSIA_HIP_MXFP4_A8_PERGROUP=1"; dev 0 env INTARSIA_REQUIRE_GPU=1 INTARSIA_HIP_MXFP4_A8_PERGROUP=1 cargo test --release -p intarsia-backend-hip --test mxfp4_a8 -- --test-threads=1 2>&1 | grep -aE "$F"
    echo "-- rows_exact mxfp4"; dev 0 env INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-hip --test rows_exact mxfp4 -- --test-threads=1 2>&1 | grep -aE "$F"
    echo "-- quick_coverage + admission"; dev 0 env INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-hip --test quick_coverage --test admission -- --test-threads=1 2>&1 | grep -aE "$F"
  ) > "$O/a-card0.log" 2>&1 & fi
  if want K || want M; then (
    if want K; then echo "-- kernels test --quick (hip)"; dev 1 /target/release/intarsia kernels test --backend hip --quick 2>&1 | tail -25; fi
    if want M; then for m in m1-table m2-noscale m3-row1 m4-noclamp m5-exponent; do
      echo "-- mutant $m (MUST FAIL)"
      dev 1 env INTARSIA_REQUIRE_GPU=1 /target/mutants/$m/mxfp4_a8 --test-threads=1 2>&1 | grep -aE "$F" | head -12; echo "exit ${PIPESTATUS[0]}"
    done; echo "-- clean (MUST PASS)"; dev 1 env INTARSIA_REQUIRE_GPU=1 /target/mutants/clean/mxfp4_a8 --test-threads=1 2>&1 | grep -aE "^test result"; fi
  ) > "$O/k-card1.log" 2>&1 & fi
  wait
  cat "$O/a-card0.log" "$O/k-card1.log" 2>/dev/null
fi
if want T; then
  echo "-- timing: decode rows (rows_exact timing_mxfp4_rows, W4A4 vs W4A8)"; dev 0 env INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-hip --test rows_exact timing_mxfp4_rows -- --ignored --nocapture 2>&1 | grep -aE "^TIME" | tee "$O/t-decode.log"
  echo "-- timing: prefill GEMMs"; dev 0 env INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-hip --test mxfp4_a8 timing_mxfp4_prefill -- --ignored --nocapture 2>&1 | grep -aE "^TIME" | tee "$O/t-prefill.log"
fi

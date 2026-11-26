#!/usr/bin/env bash
# Window stage (lane ink-hip-bpf), inside window.sh: the packed != solo diagnosis. Card 0: inkling2_packed (incl. the per-op dump
# diff) at the defaults, then with INTARSIA_HIP_INK_PF2=0 (the coordinator's heads-up: PF2 mask 151 varies run to run on tiny
# shapes). Card 1: inkling2_batch (branch, hip tiny), main's inkling2 tiny suite (the g1b baseline), the branch's with PF2=0.
# Full test output: logs/diag-*.log; the summary lines to stdout.
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-bpf
S=$REPO/.scratch
L=$REPO/logs
now() { date '+%F %T'; }
T="cargo test --release -j8 -p intarsia-models --no-fail-fast"
sum() { grep -aE "^test |test result|panicked|DIFF|differ|refused|compared|packed == solo|moe_split_rows|left:|right:" "$1" | cut -c1-260 | head -${2:-60}; }
card0() {
  for arm in default pf2off; do
    e=""; [ $arm = pf2off ] && e="INTARSIA_HIP_INK_PF2=0"
    echo "[stage] card0 inkling2_packed $arm $(now)"
    HIP_VISIBLE_DEVICES=0 timeout 120 bash "$S/run.sh" env $e INTARSIA_REQUIRE_GPU=1 INTARSIA_INKLING2_BACKEND=hip \
      INTARSIA_INK_PACK_REAL=/models/Inkling-2-Mini $T --test inkling2_packed -- --nocapture --test-threads=1 > "$L/diag-packed-$arm.log" 2>&1
    sum "$L/diag-packed-$arm.log" 70 | sed "s/^/[card0 $arm] /"
  done
  echo "[stage] card0 done $(now)"
}
card1() {
  echo "[stage] card1 inkling2_batch (branch) $(now)"
  HIP_VISIBLE_DEVICES=1 INK_CONTAINER=ink-hip-bpf-run2 timeout 80 bash "$S/run.sh" env INTARSIA_REQUIRE_GPU=1 INTARSIA_INKLING2_BACKEND=hip \
    $T --test inkling2_batch -- --test-threads=1 > "$L/diag-batch.log" 2>&1
  sum "$L/diag-batch.log" 20 | sed 's/^/[card1 batch] /'
  echo "[stage] card1 inkling2 main $(now)"
  HIP_VISIBLE_DEVICES=1 INK_CONTAINER=ink-hip-bpf-run2 INK_REPO=$S/main-src INK_VOL=intarsia-cargo-target-ink-hip-bpf-main timeout 50 bash "$S/run.sh" \
    env INTARSIA_REQUIRE_GPU=1 INTARSIA_INKLING2_BACKEND=hip $T --test inkling2 -- --test-threads=1 > "$L/diag-inkling2-main.log" 2>&1
  grep -aE "FAILED|test result" "$L/diag-inkling2-main.log" | head -8 | sed 's/^/[card1 main inkling2] /'
  echo "[stage] card1 inkling2 branch PF2=0 $(now)"
  HIP_VISIBLE_DEVICES=1 INK_CONTAINER=ink-hip-bpf-run2 timeout 50 bash "$S/run.sh" env INTARSIA_HIP_INK_PF2=0 INTARSIA_REQUIRE_GPU=1 INTARSIA_INKLING2_BACKEND=hip \
    $T --test inkling2 -- --test-threads=1 > "$L/diag-inkling2-pf2off.log" 2>&1
  grep -aE "FAILED|test result" "$L/diag-inkling2-pf2off.log" | head -8 | sed 's/^/[card1 branch pf2off inkling2] /'
  echo "[stage] card1 done $(now)"
}
card0 &
card1 &
wait
echo "[stage] done $(now)"

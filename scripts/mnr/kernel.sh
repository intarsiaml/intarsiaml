#!/bin/zsh
# Lane mnr kernel job (one fleetq slot, --quiet gpu): tests/mxfp4_naxr.rs with the knob unset and set, the K-order probes, and the
# microbench at the 27B shapes (GEMV rows of today against every form of the row kernel), twice, second pass in reverse order.
#   fleetq run --host endor --expect 12 --mem 8 --quiet gpu --label "mnr: kernel tests + bench" -- scripts/mnr/kernel.sh
set -u
[[ -n "${FLEETQ_JOB:-}" ]] || { echo "not inside a fleetq slot" >&2; exit 2; }
cd ${0:A:h}/../..
R=docs/results/2026-11-15-mnr-metal-nax-rows/raw
cargo build --release --locked -p intarsia-backend-metal --example mxfp4_bw --example naxr_order --tests 2>&1 | grep -E "^error|warning: intarsia-backend-metal|Finished" -A20
for knob in unset 1; do
  E=(); [[ $knob == 1 ]] && E=(INTARSIA_METAL_NAX_ROWS=1)
  env X=1 $E perl -e 'alarm 220; exec @ARGV' cargo test --release --locked -p intarsia-backend-metal --test mxfp4_naxr -- --nocapture > $R/test-knob-$knob.log 2>&1
  echo "tests knob $knob: exit $? $(grep -E '^test result' $R/test-knob-$knob.log)"
done
target/release/examples/naxr_order > $R/order-random.log 2>&1; grep -A1 "^spread 1" $R/order-random.log | grep -v "^--" | paste - - | cut -c1-110
target/release/examples/naxr_order tree > $R/order-tree.log 2>&1
K1="matmul_mxfp4_gemv,matmul_mxfp4_naxr_rows@0,matmul_mxfp4_naxr_rows@2,matmul_mxfp4_naxr_rows@3,matmul_mxfp4_naxr_rows@4,matmul_mxfp4_naxr_rows"
K2="matmul_mxfp4_naxr_rows,matmul_mxfp4_naxr_rows@4,matmul_mxfp4_naxr_rows@3,matmul_mxfp4_naxr_rows@2,matmul_mxfp4_naxr_rows@0,matmul_mxfp4_gemv"
scripts/mnr/bw.sh $R/bw-final-a.log $K1 1,2,3,4,8,16
scripts/mnr/bw.sh $R/bw-final-b.log $K2 16,8,4,3,2,1
echo "bench done: $(grep -c median $R/bw-final-a.log) + $(grep -c median $R/bw-final-b.log) rows"

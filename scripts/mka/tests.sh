#!/bin/zsh
# Lane mka gate 1 / existing attention tests (GPU, not timing). Under fleetq. usage: tests.sh
set -u
W=/opt/code/intarsiaml/.worktrees/mka; R=$W/docs/results/2026-11-15-mka-metal-kv-once/raw/gates; mkdir -p $R
cd $W
t() { # name env... -- runs the kv_once test binary with the env
  local name=$1; shift
  env "$@" perl -e 'alarm 600; exec @ARGV' cargo test --release --locked -p intarsia-backend-metal --test kv_once -- --nocapture > $R/kvonce-$name.log 2>&1
  echo "kv_once $name: exit $? | $(grep -c 'bit-identical' $R/kvonce-$name.log) shapes | $(grep 'test result' $R/kvonce-$name.log)"
}
t all-packed X=1
t ipt1 INTARSIA_METAL_ATTN_KVO_IPT=1
t ipt2 INTARSIA_METAL_ATTN_KVO_IPT=2
t ipt3 INTARSIA_METAL_ATTN_KVO_IPT=3
t notrim INTARSIA_METAL_ATTN_KVO_TRIM=0
t tgf7680 INTARSIA_METAL_ATTN_KVO_TGF=7680
t shipped KVO_TEST_SHIPPED_POLICY=1
# existing attention tests, knob unset and knob set
for k in 0 1; do
  for tn in hs_bitwise g0b_attn row_tables rows_exact dflash2 attention multirow; do
    INTARSIA_METAL_ATTN_KV_ONCE=$k perl -e 'alarm 900; exec @ARGV' cargo test --release --locked -p intarsia-backend-metal --test $tn > $R/test-$tn-kvo$k.log 2>&1
    echo "$tn kvo=$k: exit $? | $(grep 'test result' $R/test-$tn-kvo$k.log | tr '\n' ' ')"
  done
done
echo "## done tests"

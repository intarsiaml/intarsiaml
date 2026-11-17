#!/bin/zsh
# Lane mfp8split: kernel-level negative controls. INTARSIA_METAL_NEGCTL=fp8_xd / fp8_mmt swap in a form of the fp8 rows kernel with one association changed;
# the split test must then FAIL (so it can see a kernel that is not row-exact), and pass with the env unset. Run under fleetq (default GPU load).
set -u
W=/opt/code/intarsiaml/.worktrees/mfp8split; B=$W/.base/bin-tests/rows_exact
for nc in "" fp8_xd fp8_mmt; do
  INTARSIA_METAL_NEGCTL=$nc $B fp8_row_splits_are_bitwise_the_one_row_gemv 2>&1 | grep -aE "^test |test result|panicked" | head -3 | sed "s/^/NEGCTL=${nc:-none}: /"
done

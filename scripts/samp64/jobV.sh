#!/usr/bin/env bash
# samp64 gate run on the fixed binary (intarsia-fix). Candidate schedule 16384:1 (k=1 above 16k) and 16384:1,40000:0 (plain steps above 40k).
#   w8  none (k=2 at every context = main): seeded, sampled and greedy 4k/32k/64k
#   w9  16384:1:                            the same cells
#   w10 16384:1,40000:0:                    seeded, sampled 32k/49k/64k
#   w11 none again:                         sampled 4k/32k/64k (drift check)
D="$(dirname "${BASH_SOURCE[0]}")"
export SAMP64_BIN='~/intarsiaml-samp64/bin/intarsia-fix'
C="0.7:4096,32768,65536:5 0:4096,32768,65536:5"
bash "$D/window.sh" w8 "" - seeded $C
bash "$D/window.sh" w9 "" "INTARSIA_SPEC_SAMPLED_CTX_K=16384:1" seeded $C
bash "$D/window.sh" w10 "" "INTARSIA_SPEC_SAMPLED_CTX_K=16384:1,40000:0" seeded 0.7:32768,49152,65536:5
bash "$D/window.sh" w11 "" - 0.7:4096,32768,65536:5

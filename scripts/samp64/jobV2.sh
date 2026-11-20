#!/usr/bin/env bash
# samp64 sweep on the fixed binary (intarsia-fix), sampled T=0.7: draft schedule none (k=2 at every context, = main), "1:1" (k=1 everywhere),
# "1:0" (plain steps everywhere; seeded identity against none). Sizes 4k..64k, 5 reps.
D="$(dirname "${BASH_SOURCE[0]}")"
export SAMP64_BIN='~/intarsiaml-samp64/bin/intarsia-fix'
S="0.7:4096,8192,16384,24576,32768,49152,65536:5"
bash "$D/window.sh" w8 "" - seeded $S
bash "$D/window.sh" w9 "" "INTARSIA_SPEC_SAMPLED_CTX_K=1:1" $S
bash "$D/window.sh" w10 "" "INTARSIA_SPEC_SAMPLED_CTX_K=1:0" seeded $S

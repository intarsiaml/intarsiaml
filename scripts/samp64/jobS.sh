#!/usr/bin/env bash
# samp64 follow-up on the fixed binary: seeded identity cells (w8's seeded cell crashed on a too-long probe prompt): no schedule (w12) and 16384:1 (w13);
# mid-context sampled cells, no schedule (w14) against k=1 everywhere (w15), which place the crossover between k=2 and k=1 and give 49k its baseline.
D="$(dirname "${BASH_SOURCE[0]}")"
export SAMP64_BIN='~/intarsiaml-samp64/bin/intarsia-fix'
M="0.7:8192,16384,24576,49152:5"
bash "$D/window.sh" w12 "" - seeded $M
bash "$D/window.sh" w13 "" "INTARSIA_SPEC_SAMPLED_CTX_K=16384:1" seeded
bash "$D/window.sh" w15 "" "INTARSIA_SPEC_SAMPLED_CTX_K=1:1" $M

#!/usr/bin/env bash
# samp64 validation of INTARSIA_SPEC_SAMPLED_MAX_CTX on the fixed binary (intarsia-fix): w8 limit 32768 (seeded identity, sampled and greedy 4k/32k/64k),
# w9 no limit (seeded identity reference, mid-context spec-on curve), w10 --spec off (mid-context curve).
D="$(dirname "${BASH_SOURCE[0]}")"
export SAMP64_BIN='~/intarsiaml-samp64/bin/intarsia-fix'
bash "$D/window.sh" w8 "" "INTARSIA_SPEC_SAMPLED_MAX_CTX=32768" seeded 0.7:4096,32768,65536:5 0:4096,32768,65536:5
bash "$D/window.sh" w9 "" - seeded 0.7:8192,16384,24576,49152:5
bash "$D/window.sh" w10 "--spec off" - 0.7:8192,16384,24576,49152:5

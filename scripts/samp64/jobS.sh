#!/usr/bin/env bash
# samp64 follow-up: seeded identity cells (w8's seeded cell crashed on a too-long probe prompt) on the fixed binary: no schedule, and 16384:1.
D="$(dirname "${BASH_SOURCE[0]}")"
export SAMP64_BIN='~/intarsiaml-samp64/bin/intarsia-fix'
bash "$D/window.sh" w12 "" - seeded
bash "$D/window.sh" w13 "" "INTARSIA_SPEC_SAMPLED_CTX_K=16384:1" seeded

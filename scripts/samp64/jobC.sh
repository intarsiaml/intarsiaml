#!/usr/bin/env bash
# samp64 phase 3: where the sampled spec-on / spec-off curves cross (mid contexts), existing binary.
D="$(dirname "${BASH_SOURCE[0]}")"
C="0.7:8192,16384,24576,49152:5"
bash "$D/window.sh" w6 "" - $C
bash "$D/window.sh" w7 "--spec off" - $C

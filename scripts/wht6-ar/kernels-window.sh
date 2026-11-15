#!/usr/bin/env bash
# Lane wht6-ar, gate 1 (both R9700s): the wht6 wire against the exact all-reduce (tests/ar_wht6.rs), the existing comm suites
# with the wire off (unchanged) and with INTARSIA_HIP_AR_WIRE=wht6 in the environment (the suites build their groups from explicit
# modes or from the environment), and the all-reduce rate, exact vs wht6. ON tatooine, inside a fleetq lease with radiance down.
set -u
cd "$(dirname "$0")/../.." && . scripts/wht6-ar/lib.sh
T="cargo test --release -p intarsia-backend-hip"
STEPS=${STEPS:-"wht6 comm rate"}
has() { [[ " $STEPS " == *" $1 "* ]]; }
if has wht6; then E="" run 0,1 600 wht6 $T --test ar_wht6 -- --nocapture --test-threads=1; fi
if has comm; then E="" run 0,1 900 comm $T --test comm --test ar_push --test ar_norm -- --test-threads=1; fi
if has rate; then E="" run 0,1 600 rate $T --test ar_wht6 wht6_rate -- --ignored --nocapture --test-threads=1; fi
log "done"

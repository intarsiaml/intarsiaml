#!/usr/bin/env bash
# Lane msd2: one --load cpu slot: base CLI (origin/main), dev CLI (working tree) and the dev rows_exact test binary on spark1.
#   fleetq run --host spark1 --load cpu --mem 16 --expect 60 --label "msd2: builds (cpu)" -- scripts/msd/cpu-build.sh [nobase]
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ "${1:-}" == nobase ]] || { date; REF=origin/main "$H/build.sh" base; }
date; "$H/build.sh" dev
date; ssh -n -o BatchMode=yes spark1 'export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH; cd ~/intarsiaml-msd2/src && CARGO_BUILD_JOBS=16 cargo test -p intarsia-backend-cuda --release --test rows_exact --no-run 2>&1 | grep -E "^(error|warning)|Finished|Executable" | tail -8'
date

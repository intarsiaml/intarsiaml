#!/usr/bin/env bash
# Lane dfh: dry-run a window's stages script on tatooine in a cpu slot (radiance stays up), before the window itself:
#   fleetq run --host tatooine-r9700 --load cpu --mem 0 --expect 10 --label "dfh dry <W>" -- env W=w4 scripts/dfh/dry.sh
# Every intarsia command is printed with its environment instead of run; the GPU tests run with INTARSIA_VERIFY_DRYRUN=1 and print
# their PROBE_CONFIG line. Read the output before taking the window (coordinator rule after window 2, 2026-10-26).
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}" "${W:?W=<window name>}"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ssh -o BatchMode=yes tatooine 'mkdir -p ~/dfh' < /dev/null
scp -q "$S"/lib.sh "$S"/stages-*.sh tatooine:dfh/
ssh -o BatchMode=yes tatooine "DRY=1 FLEETQ_JOB=$FLEETQ_JOB ${XENV:-} bash -c 'O=\$HOME/dfh/dry/$W; rm -rf \$O; mkdir -p \$O; bash \$HOME/dfh/stages-${STAGES:-$W}.sh \$O < /dev/null; echo; grep -ah PROBE_ \$O/*.log 2>/dev/null'" < /dev/null

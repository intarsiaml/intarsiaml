#!/usr/bin/env bash
# Lane vrc: dry-run a window's stages script on tatooine in a cpu slot (radiance stays up), before the window itself:
#   fleetq run --host tatooine-r9700 --load cpu --mem 0 --expect 10 --label "vrc dry <W>" -- env W=w4 scripts/vrc/dry.sh
# Every intarsia command is printed with its environment instead of run; the GPU tests run with INTARSIA_VERIFY_DRYRUN=1 and print
# their PROBE_CONFIG line. Read the output before taking the window (coordinator rule after window 2, 2026-10-26).
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}" "${W:?W=<window name>}"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ssh -o BatchMode=yes tatooine 'mkdir -p ~/intarsiaml-vrc/.vrc' < /dev/null
scp -q "$S"/lib.sh "$S"/stages-*.sh "$S"/cmp_taps.py tatooine:intarsiaml-vrc/.vrc/
ssh -o BatchMode=yes tatooine "DRY=1 FLEETQ_JOB=$FLEETQ_JOB ${XENV:-} bash -c 'O=\$HOME/intarsiaml-vrc/.vrc/dry/$W; rm -rf \$O; mkdir -p \$O; bash \$HOME/intarsiaml-vrc/.vrc/stages-${STAGES:-$W}.sh \$O < /dev/null; echo; grep -ah PROBE_ \$O/*.log 2>/dev/null'" < /dev/null

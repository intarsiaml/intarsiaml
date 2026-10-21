#!/usr/bin/env bash
# Local wrapper: one radiance-down window on tatooine (trap inside restores radiance), then fetch the outputs.
set -u
: "${FLEETQ_JOB:?run under fleetq}"
D=/private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/scratchpad/m3
ssh -o BatchMode=yes tatooine "FLEETQ_JOB=$FLEETQ_JOB bash ~/intarsiaml-m3/m3data/hip_burst_m3_w2.sh"
rc=$?
mkdir -p $D/tatooine
scp -q 'tatooine:intarsiaml-m3/m3out/*' $D/tatooine/
scp -q 'tatooine:hipburst-logs/m3-*' $D/tatooine/
exit $rc

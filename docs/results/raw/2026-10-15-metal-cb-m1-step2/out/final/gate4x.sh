#!/bin/zsh
[ -n "${FLEETQ_JOB:-}" ] || exit 2
D=/private/tmp/claude-501/-opt-code-intarsiaml/0e8026c5-6dd0-4b83-b375-5ff0a66894fc/scratchpad/step2f
G=/opt/code/intarsiaml-cb-wire/docs/results/raw/2026-10-15-metal-cb-m1-step2
GEN=256 ORDER="1 8 1 8 1 8" $G/gate4.sh $D/gate4-gen256 9ebd6f8
GEN=512 ORDER="1 8 1" $G/gate4.sh $D/gate4-gen512 9ebd6f8

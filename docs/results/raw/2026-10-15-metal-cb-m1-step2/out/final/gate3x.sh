#!/bin/zsh
[ -n "${FLEETQ_JOB:-}" ] || exit 2
REFDIR=/private/tmp/claude-501/-opt-code-intarsiaml/0e8026c5-6dd0-4b83-b375-5ff0a66894fc/scratchpad/step2f/gate3 MB=4 NOREUSE=1 BS="4 8" /opt/code/intarsiaml-cb-wire/docs/results/raw/2026-10-15-metal-cb-m1-step2/gate3.sh /private/tmp/claude-501/-opt-code-intarsiaml/0e8026c5-6dd0-4b83-b375-5ff0a66894fc/scratchpad/step2f/gate3-mb4 9ebd6f8
echo "--- reuse enabled (informational)"
REFDIR=/private/tmp/claude-501/-opt-code-intarsiaml/0e8026c5-6dd0-4b83-b375-5ff0a66894fc/scratchpad/step2f/gate3 MB=8 NOREUSE=0 BS="8" /opt/code/intarsiaml-cb-wire/docs/results/raw/2026-10-15-metal-cb-m1-step2/gate3.sh /private/tmp/claude-501/-opt-code-intarsiaml/0e8026c5-6dd0-4b83-b375-5ff0a66894fc/scratchpad/step2f/gate3-reuse 9ebd6f8
echo "--- sparse arrivals (more requests prefill alone with the full chunk)"
REFDIR=/private/tmp/claude-501/-opt-code-intarsiaml/0e8026c5-6dd0-4b83-b375-5ff0a66894fc/scratchpad/step2f/gate3 MB=8 NOREUSE=1 BS="2 4" SCALE=3 /opt/code/intarsiaml-cb-wire/docs/results/raw/2026-10-15-metal-cb-m1-step2/gate3.sh /private/tmp/claude-501/-opt-code-intarsiaml/0e8026c5-6dd0-4b83-b375-5ff0a66894fc/scratchpad/step2f/gate3-sparse 9ebd6f8

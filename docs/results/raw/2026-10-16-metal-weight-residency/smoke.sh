#!/bin/zsh
SC=/private/tmp/claude-501/-opt-code-intarsiaml/0e8026c5-6dd0-4b83-b375-5ff0a66894fc/scratchpad/resid
R=/opt/code/intarsiaml-residency/docs/results/raw/2026-10-16-metal-weight-residency
$SC/keepwarm-test --test-threads=1 > $1.keepwarm-test.txt 2>&1; echo "keepwarm test exit $?"
$R/serve-idle.sh $1 1 base kw

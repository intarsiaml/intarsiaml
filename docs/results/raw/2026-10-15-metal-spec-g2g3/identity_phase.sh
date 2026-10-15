#!/bin/zsh
set -u
R=/opt/code/intarsiaml-spec/docs/results/raw/2026-10-15-metal-spec-g2g3
$R/run_identity.sh $R/identity $R/cases-24.txt base off on4 on6

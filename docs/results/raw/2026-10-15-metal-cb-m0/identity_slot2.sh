#!/bin/zsh
# base binary with --spec ngram: the reference for the spec-on variant (spec counters and trace lines must match too)
R=${0:A:h}
$R/run_identity.sh $R/identity $R/cases-all.txt bon4
$R/serve.sh $R/serve bon4

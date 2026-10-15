#!/bin/zsh
# extra min_tokens cases that put EOS inside the min_tokens window of an accepted run (the verify "plain next" cut path)
R=${0:A:h}
$R/run_identity.sh $R/identity-extra $R/cases-extra.txt base off bon4 on4

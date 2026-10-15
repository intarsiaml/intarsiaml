#!/bin/zsh
R=${0:A:h}
$R/run_identity.sh $R/identity $R/cases-all.txt base off on4
$R/serve.sh $R/serve base off on4

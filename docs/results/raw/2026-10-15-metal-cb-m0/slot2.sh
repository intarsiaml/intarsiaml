#!/bin/zsh
R=${0:A:h}
$R/qualify_acc.sh $R/qualify-acc
$R/serve-gap/gap_run.sh

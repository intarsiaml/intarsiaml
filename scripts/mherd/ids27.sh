#!/bin/zsh
# 27B ids-only gates, scaled: herd (8 x 4160 shared + 200 own, 1 seed, A1 B1) then sjf (12000 + 1000, 1 seed, A1 B1)
source ${0:A:h}/lib.sh
export MODEL=27b
MODEL=27b SHARED=4160 OWN=200 SEEDS="1" MAXSEQ=5120 OUT=$R/herd-27b ${S}/herd.sh "A1 B1"
MODEL=27b LONG=12000 SHORT=1000 SEEDS="1" OUT=$R/sjf-27b ${S}/sjf.sh "A1 B1"

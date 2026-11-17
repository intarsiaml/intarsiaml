#!/bin/zsh
# Lane mfp8split: every gate step for one model in one slot. usage: allgates.sh <27b|35b>   (run under fleetq --quiet gpu,cpu)
set -u
G=/opt/code/intarsiaml/.worktrees/mfp8split/scripts/mfp8split/gates.sh
for s in solo served compare route mtp; do $G $1 $s; done

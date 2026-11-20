#!/usr/bin/env bash
# samp64 phase 1: shipping config (spec mtp default) and --spec off, sampled T=0.7 vs greedy, 4k/32k/64k, 5 reps. One fleetq job, two servers in turn.
D="$(dirname "${BASH_SOURCE[0]}")"
bash "$D/window.sh" w1 "" - 0.7:4096,32768,65536:5 0:4096,32768,65536:5
bash "$D/window.sh" w2 "--spec off" - 0.7:4096,32768,65536:5 0:4096,32768,65536:5

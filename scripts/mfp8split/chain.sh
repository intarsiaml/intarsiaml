#!/bin/zsh
# Lane mfp8split: the queued runs after the 35B gates (each its own fleetq slot, in order).
cd /opt/code/intarsiaml/.worktrees/mfp8split; R=docs/results/2026-11-17-mfp8split/raw
fleetq run --host endor --expect 50 --quiet gpu,cpu --label "mfp8split ABBA 35b (8 visits)" -- env COOL=120 ./scripts/mfp8split/abba.sh 35b $R/abba-35b "A1 B1 B2 A2 B3 A3 A4 B4" > $R/abba-35b.log 2>&1
fleetq run --host endor --expect 90 --quiet gpu,cpu --label "mfp8split gates 27b (solo/served/route/mtp)" -- ./scripts/mfp8split/allgates.sh 27b > $R/gates-27b.log 2>&1
fleetq run --host endor --expect 60 --quiet gpu,cpu --label "mfp8split ABBA 27b (4 visits)" -- env COOL=120 ./scripts/mfp8split/abba.sh 27b $R/abba-27b "A1 B1 B2 A2" > $R/abba-27b.log 2>&1

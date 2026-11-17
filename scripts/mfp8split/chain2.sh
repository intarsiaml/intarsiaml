#!/bin/zsh
# Lane mfp8split: 35B ABBA rerun (cells 9..12 only, 12 visits A B B A A B B A A B B A), after the 27B ABBA of chain.sh is done.
cd /opt/code/intarsiaml/.worktrees/mfp8split; R=docs/results/2026-11-17-mfp8split/raw
until grep -q "^## done" $R/abba-27b.log 2>/dev/null; do sleep 30; done
fleetq run --host endor --expect 50 --quiet gpu,cpu --label "mfp8split ABBA 35b rerun (cells 9..12, 12 visits)" -- env COOL=120 CELLS="9 10 11 12" ./scripts/mfp8split/abba.sh 35b $R/abba-35b-r2 "A1 B1 B2 A2 A3 B3 B4 A4 A5 B5 B6 A6" > $R/abba-35b-r2.log 2>&1

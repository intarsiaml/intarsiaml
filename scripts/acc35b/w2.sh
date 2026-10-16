#!/usr/bin/env bash
# Window 2: the MoE small-T sweep only (16 seeds x 12 token counts), see docs/results/2026-10-14-35b-accuracy.md.
STAGES=sweep OUT=w2 BUDGET_MIN=8 exec bash $HOME/intarsia-acc35b/scripts/w1.sh

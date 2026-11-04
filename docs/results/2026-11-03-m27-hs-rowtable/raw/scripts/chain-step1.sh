#!/bin/zsh
# Step 1: quick A/B of the existing INTARSIA_METAL_ATTN_HS=3 on the row-table path. A = default binary (env unset), C = HS=3.
# Order A4 C4 C16 A16 (one fleetq job per arm). Run from the worktree root.
export FLEETQ_OWNER=claude-m27-hsr
BIN=$PWD/.scratch/intarsia.def; O=docs/results/2026-11-03-m27-hs-rowtable/raw/ab-hs3; S=docs/results/2026-11-03-m27-hs-rowtable/raw/scripts/conc-arm.sh
fleetq run --host endor --quiet gpu,cpu --mem 45 --expect 14 --label "m27 rowtable step1 A mb4 (def)" -- zsh $S A-mb4 $BIN $O 4 4
INTARSIA_METAL_ATTN_HS=3 fleetq run --host endor --quiet gpu,cpu --mem 45 --expect 14 --label "m27 rowtable step1 C mb4 (HS=3)" -- zsh $S C-mb4 $BIN $O 4 4
INTARSIA_METAL_ATTN_HS=3 fleetq run --host endor --quiet gpu,cpu --mem 45 --expect 30 --label "m27 rowtable step1 C mb16 (HS=3)" -- zsh $S C-mb16 $BIN $O 16 16
fleetq run --host endor --quiet gpu,cpu --mem 45 --expect 30 --label "m27 rowtable step1 A mb16 (def)" -- zsh $S A-mb16 $BIN $O 16 16

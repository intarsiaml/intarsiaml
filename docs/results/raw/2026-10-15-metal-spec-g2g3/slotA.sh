#!/bin/zsh
# One exclusive slot, four phases: identity (24 cases x off/on4/on6, base outputs reused from the G1 run: same base binary
# c556008, same prompts, deterministic greedy), G2 speed ABBA, G3 realistic workload, merge-bar GPU parts.
set -u
R=/opt/code/intarsiaml-spec/docs/results/raw/2026-10-15-metal-spec-g2g3
NEW=/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/bin/intarsia-gate-final
echo "== identity $(date +%T)"
$R/run_identity.sh $R/identity $R/cases-24.txt off on4 on6
echo "== g2 $(date +%T)"
$R/g2/g2_run.sh
echo "== g3 $(date +%T)"
$R/g3/g3_run.sh $NEW $R/g3/runs 3 "off on"
echo "== merge bar $(date +%T)"
$R/merge/merge_bar.sh
echo "== done $(date +%T)"

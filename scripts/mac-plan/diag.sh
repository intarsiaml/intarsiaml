#!/bin/zsh
# mac-plan diagnostics: TTFT with and without the MTP default (35B), and the cost of one short prefill pass. Inside a fleetq slot.
D=/opt/code/intarsiaml/.worktrees/mac-plan/scripts/mac-plan
$D/ttft.sh mtp off "mtptrace:INTARSIA_SPEC_TRACE=1"
$D/shortpass.sh

#!/bin/zsh
# TTFT under a cb trace: 35B served, MTP default vs --spec off (tags ending in "cb" get INTARSIA_METAL_PROFILE=cb). Inside a fleetq slot.
D=/opt/code/intarsiaml/.worktrees/mac-plan/scripts/mac-plan
$D/ttft.sh mtpcb "offcb::--spec off"

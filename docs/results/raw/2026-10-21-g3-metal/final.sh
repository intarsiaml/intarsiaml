#!/bin/zsh
# G3 final slot (resume after the endor hold): GPU gates, interleaved A/B x3 (base vs dev), first-commit mutants,
# then the restated Metal speed gate (scripts/metal-speed-gate) on the dev binary.
D=/opt/code/intarsiaml/.claude/worktrees/agent-a34836167f72ea687/docs/results/raw/2026-10-21-g3-metal
W=/opt/code/intarsiaml/.claude/worktrees/agent-a34836167f72ea687
S=/private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/scratchpad
zsh $D/gates.sh $D/gates2 > $D/gates2.console 2>&1
echo "gates exit $?"
zsh $D/ab.sh ab > $D/ab.console 2>&1
echo "ab exit $?"
zsh $D/mutants.sh > $D/mutants.console 2>&1
echo "mutants exit $?"
zsh $W/scripts/metal-speed-gate/sameslot.sh $S/intarsia-dev $D/speedgate > $D/speedgate.console 2>&1
echo "speed gate exit $?"
echo FINAL_DONE $(date -Iseconds)

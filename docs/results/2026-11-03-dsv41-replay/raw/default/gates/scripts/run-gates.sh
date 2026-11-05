#!/bin/zsh
G=/opt/code/intarsiaml/.worktrees/replay-gates
for i in {1..80}; do [ -x $G/intarsia.new ] && [ -x $G/intarsia.base ] && break; sleep 15; done
[ -x $G/intarsia.new ] && [ -x $G/intarsia.base ] || { echo "binaries missing"; exit 9; }
cd /opt/code/intarsiaml/.worktrees/replay-gates/new && zsh $G/scripts/gates-lf.sh $G/out/metal $G/intarsia.new $G/intarsia.base

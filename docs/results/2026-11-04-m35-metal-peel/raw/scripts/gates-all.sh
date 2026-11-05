#!/bin/zsh
cd /opt/code/intarsiaml/.worktrees/m35-peel
S=$PWD/.scratch
echo "#### gates-a"; zsh $S/gates-a.sh
echo "#### mtp"; zsh $S/gates-mtp.sh "2050 2057 6146"
echo "#### 8B ids (main binary)"; BG_MAX=1200 BG_SLEEP=1 zsh $S/ids-run.sh $S/logs/ids8 $S/logs/gates/prompts-text8.json 258,260,265,266,772,1028,1088,2056 /Users/josef/models/Qwen3-8B 18482 main:$S/intarsia.main

#!/bin/zsh
cd /opt/code/intarsiaml/.worktrees/m35-peel
S=$PWD/.scratch
export BG_MAX=1200 BG_SLEEP=1
for m in Llama-3.1-8B-Instruct Qwen2.5-7B-Instruct; do
  echo "#### $m"; zsh $S/ids-run.sh $S/logs/ids-$m $S/logs/gates/prompts-text-$m.json 258,260,265,1028 /Users/josef/models/$m 18484 main:$S/intarsia.main
done
echo "#### 35B main with --busy-chunk 4096 (option b evidence)"
SERVE_EXTRA="--busy-chunk 4096" zsh $S/ids-run.sh $S/logs/ids35-busy4096 $S/logs/gates/prompts-text35.json 258,1028,2050,2052,2056,2057,6146,6153,2058 /Users/josef/models/Qwen3.6-35B-A3B-NVFP4 18485 busy4096:$S/intarsia.main

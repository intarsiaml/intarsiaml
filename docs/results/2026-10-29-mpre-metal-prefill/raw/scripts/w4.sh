#!/bin/zsh
# MPRE window 4: MMLU n=1425 on c1 (35B), then 4k self-golden at 64 decode positions. Inside ONE fleetq slot.
set -u
L=/opt/code/intarsiaml/.worktrees/mpre/.lane; OUT=$L/w4; mkdir -p $OUT
echo "$(date +%T) start rustc=$(pgrep -x rustc | wc -l)" >> $OUT/marks.txt
BIN=$L/intarsia-c1 zsh $L/mmlu-arm.sh intarsia35 $OUT/mmlu > $OUT/mmlu.log 2>&1
echo "$(date +%T) mmlu exit $?: $(grep -iE 'overall|MMLU (Overall|Accuracy:)' $OUT/mmlu/intarsia35-quality.out | tail -2 | tr '\n' ' ')" >> $OUT/marks.txt
M=$HOME/models/Qwen3.6-35B-A3B-NVFP4
$L/intarsia-c1 accuracy --model $M --backend metal --mode both --decode-positions 64 --allow-unstamped --allow-same-code --max-seq 8192 --self-golden $L/w2/4k.off.sg.safetensors > $OUT/self-4k.log 2>&1
echo "$(date +%T) self 4k exit $?: $(grep -a SELF-STATS $OUT/self-4k.log | tr '\n' ' ')" >> $OUT/marks.txt
$L/intarsia-c1 accuracy --model $M --backend metal --mode both --decode-positions 16 --allow-unstamped --allow-same-code --max-seq 16384 --self-golden $L/w2/long.off.sg.safetensors > $OUT/self-long.log 2>&1
echo "$(date +%T) self long exit $?: $(grep -a SELF-STATS $OUT/self-long.log | tr '\n' ' ')" >> $OUT/marks.txt

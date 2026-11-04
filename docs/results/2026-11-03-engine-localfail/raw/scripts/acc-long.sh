#!/bin/zsh
# rerun of the long 8B golden with the baseline's decode positions (64); first run used 16 and was refused. usage: acc-long.sh OUT NEW BASE
O=${1:A}; F=${2:A}; B=${3:A}; G=/opt/code/intarsiaml/golden; g=qwen3-8b-long-fp32; M8=/Users/josef/models/Qwen3-8B
for t in base new; do case $t in base) X=$B;; new) X=$F;; esac
  $X accuracy --backend metal --model $M8 --max-seq 4096 --golden $G/$g.safetensors --mode both --decode-positions 64 --baseline golden/baselines/$g.metal.baseline.json --allow-unstamped > $O/accuracy/$g-$t.out 2> $O/accuracy/$g-$t.err; echo "accuracy $g $t exit $?"
done
grep -E "verdict|golden gate" $O/accuracy/$g-new.out | head -4
cmp -s $O/accuracy/$g-base.out $O/accuracy/$g-new.out && echo "accuracy $g: base and new byte-identical" || echo "accuracy $g: DIFFERS"

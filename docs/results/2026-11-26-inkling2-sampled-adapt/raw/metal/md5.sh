#!/bin/bash
# endor, plain fleetq slot (no timing): Metal greedy md5, main6 (main aee06201c) vs ad4 (this branch), --spec off and the recipe default (DSpark k3),
# md5.json (6 short + 4k), 64 tokens, no INTARSIA_* variable set. Prints the md5 of the `ids[...]` lines per run.
set -u
W=/opt/code/intarsiaml/.worktrees/ink-sampled-adapt; S=$W/.scratch; O=$S/metal/md5; mkdir -p $O
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset $v; done
M=$HOME/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/b85c3cfa615fc413cc9af3379f491b963f3b586a
for b in main6 ad4; do for sp in off def; do
  x=""; [ $sp = off ] && x="--spec off"
  $S/bin/intarsia-$b generate -m $M --backend metal --max-seq 8192 --profile-every 0 --show-ids --max-tokens 64 --prompt-ids $S/prompts/md5.json $x > $O/$b-$sp.out 2> $O/$b-$sp.err
  echo "$b $sp rc $? ids-md5 $(grep -a '^ids\[' $O/$b-$sp.err | md5 -q) lines $(grep -ac '^ids\[' $O/$b-$sp.err) | $(grep -aE '^spec: verify' $O/$b-$sp.err | cut -c1-110)"
done; done

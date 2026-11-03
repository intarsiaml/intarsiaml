#!/bin/zsh
BIN=/opt/code/intarsiaml/.worktrees/m27/.scratch/intarsia-nax; O=/opt/code/intarsiaml/.worktrees/m27/.scratch/mtp
M=/Users/josef/models/Qwen3.8-27B-Quark-AWQ-MXFP4; mkdir -p $O
P1="Write a short story about a lighthouse keeper who discovers a message in a bottle. Include dialogue."
P2="Write a Rust function that parses a CSV line into fields, handling quoted commas, with unit tests."
for pi in 1 2; do
  eval P=\$P$pi
  for mode in off k2 k3; do
    case $mode in off) A=(--spec off);; k2) A=(--spec mtp --spec-k 2 --mtp-draft-vocab 65536);; k3) A=(--spec mtp --spec-k 3 --mtp-draft-vocab 65536);; esac
    $BIN generate --model $M --backend metal --max-seq 4096 --temperature 0 --max-tokens 160 --ignore-eos --profile-every 0 $A --chat --no-think -p "$P" > $O/p$pi-$mode.out 2> $O/p$pi-$mode.err
    echo "p$pi $mode: $(grep -h 'finish=' $O/p$pi-$mode.err | grep -o 'decode [0-9]* ms ([0-9.]* tok/s)') $(grep -io 'accept[^|]*' $O/p$pi-$mode.err | head -2 | tr '\n' ' ')"
  done
done

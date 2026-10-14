#!/bin/zsh
# usage: reps.sh <ctx short|4k> <reps> <variant spec>...  where spec = tag:binary:commit_every(or -)
# Interleaved cb-trace runs (129 forced tokens); prints one line per run: tag period busy idle tok/s. Run inside a fleetq slot.
CTX=$1; REPS=$2; shift 2
W=/opt/code/intarsiaml/.claude/worktrees/agent-a40aebdca23def5e0
R=$W/docs/results/raw/2026-10-14-metal-phase-b
O=$R/b2b/reps
M=/Users/josef/models/Qwen3-8B
for rep in $(seq 1 $REPS); do
  for spec in "$@"; do
    tag=${spec%%:*}; rest=${spec#*:}; bin=${rest%%:*}; ce=${rest#*:}
    if [ "$ce" != "-" ]; then export INTARSIA_METAL_COMMIT_EVERY=$ce; else unset INTARSIA_METAL_COMMIT_EVERY; fi
    f=$O/$tag-$CTX-$rep
    INTARSIA_METAL_PROFILE=cb:$f.jsonl $bin generate --model $M --backend metal --max-seq 8192 --profile-every 0 --temperature 0 --max-tokens 129 --ignore-eos --prompt "$(cat $R/prompt-$CTX.txt)" > $f.out 2> $f.err
    unset INTARSIA_METAL_PROFILE
    D=544; [ "$tag" = base ] && D=616
    echo "$tag $CTX rep$rep $(python3 $R/b2b/period.py $f.jsonl $D 128) $(grep -o '([0-9.]* tok/s)' $f.err | tail -1)"
  done
done

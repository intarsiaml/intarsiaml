#!/bin/zsh
# prefill time of the 4k and 1k prompts (1 generated token), interleaved variants. Run inside a fleetq slot.
S=/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/b2fuse
W=/opt/code/intarsiaml/.claude/worktrees/agent-a40aebdca23def5e0
R=$W/docs/results/raw/2026-10-14-metal-phase-b
M=/Users/josef/models/Qwen3-8B
run() { # label bin ce
  if [ "$3" != "-" ]; then export INTARSIA_METAL_COMMIT_EVERY=$3; else unset INTARSIA_METAL_COMMIT_EVERY; fi
  $2 generate --model $M --backend metal --max-seq 8192 --profile-every 0 --temperature 0 --max-tokens 2 --ignore-eos --prompt "$(cat $R/prompt-4k.txt)" 2>&1 >/dev/null | grep -o "prefill [0-9]* ms ([0-9.]* tok/s)" | sed "s/^/$1 /"
}
for rep in 1 2 3 4; do
  run base $S/intarsia-base0 -
  run new-c128 $S/intarsia-final -
  run new-c4096 $S/intarsia-final 4096
  run new-c1024 $S/intarsia-final 1024
done

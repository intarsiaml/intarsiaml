#!/bin/zsh
# commit-cadence sweep (run inside a fleetq slot): decode tok/s of 257 forced tokens, short prompt
S=/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/b2fuse
W=/opt/code/intarsiaml/.claude/worktrees/agent-a40aebdca23def5e0
R=$W/docs/results/raw/2026-10-14-metal-phase-b
NEW=$W/target/release/intarsia
BASE=$S/intarsia-base0
M=/Users/josef/models/Qwen3-8B
run() { # label bin envval
  if [ -n "$3" ]; then export INTARSIA_METAL_COMMIT_EVERY=$3; else unset INTARSIA_METAL_COMMIT_EVERY; fi
  $2 generate --model $M --backend metal --max-seq 8192 --profile-every 0 --temperature 0 --max-tokens 257 --ignore-eos --prompt "$(cat $R/prompt-short.txt)" 2>&1 >/dev/null | grep -o "decode [0-9]* ms ([0-9.]* tok/s)" | sed "s/^/$1 /"
}
for rep in 1 2 3; do
  run base $BASE ""
  run new-4096 $NEW ""
  run new-512 $NEW 512
  run new-128 $NEW 128
  run new-64 $NEW 64
  run new-32 $NEW 32
done

#!/bin/zsh
# Lane lane-ds: the 4k greedy-ids check of endor-gates.sh redone: the first run's 4096-token prompt did not fit the default 4096 context (both binaries
# failed alike, empty ids). --max-seq 8192 here.
[ -n "${FLEETQ_JOB:-}" ] || exit 2
cd /opt/code/intarsiaml/.worktrees/device-sampling || exit 2
S=$PWD/.scratch; O=$PWD/docs/results/2026-11-05-device-sampling-default/raw/endor; W=/Users/josef/models
P4=docs/results/raw/2026-10-15-metal-longctx-profile/prompts/prompt-4096.txt
for t in main ds; do
  $S/intarsia.$t generate --model $W/Qwen3-8B --backend metal --max-seq 8192 -p "$(cat $P4)" --max-tokens 256 --ignore-eos --temperature 0 --show-ids > $O/ids4k-Qwen3-8B.$t.out 2> $O/ids4k-Qwen3-8B.$t.err
  echo "$t exit $? ids lines: $(grep -a -c '^ids' $O/ids4k-Qwen3-8B.$t.err) md5 $(grep -a '^ids' $O/ids4k-Qwen3-8B.$t.err | md5 | cut -c1-12)"
done

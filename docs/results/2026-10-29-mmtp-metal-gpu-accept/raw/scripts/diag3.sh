#!/bin/bash
# MMTP gate: main 0d02d3f36 binary's MTP ids on the 4 gate prompts (vs this branch's window-1 ids), and the reuse tests.
set -u
[ -n "${FLEETQ_JOB:-}" ] || exit 2
W=/opt/code/intarsiaml/.worktrees/mmtp; O=$W/.lane/diag3; mkdir -p $O
MAIN=$W/.lane/main-target/release/intarsia; M=$HOME/models/Qwen3.6-35B-A3B-NVFP4
for p in short 4ka 4kb 16k; do
  $MAIN generate --model $M --backend metal --max-seq 20000 --profile-every 0 --temperature 0 --max-tokens 256 --ignore-eos --show-ids --spec mtp --prompt "$(cat $W/.lane/prompts/$p.txt)" > $O/$p-mainmtp.out 2> $O/$p-mainmtp.err
  a=$(grep -E "^ids" $O/$p-mainmtp.err | md5); b=$(grep -E "^ids" $W/.lane/w1/$p-mtp1.err | md5)
  [ "$a" = "$b" ] && echo "$p: main MTP == branch MTP ($a)" || echo "$p: main MTP != branch MTP DIFFERS"
  grep "verify passes" $O/$p-mainmtp.err
done
cd $W
cargo test --release -p intarsia-models --test exact_reuse --test prefix_reuse_real > $O/reuse-tests.log 2>&1; echo "reuse tests exit $?"; grep -E "^test |test result" $O/reuse-tests.log

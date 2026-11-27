#!/usr/bin/env bash
# endor EXCLUSIVE fleetq slot (lane ink-hip-overlap2, the Metal no-op gate), <= 15 min:
# (1) greedy md5 on Inkling-2 on Metal: main6 (origin/main) vs o6 (the branch), `--spec off` and the default (DSpark), the 7 prompts of
#     md5.json (6 short + 4k; lane ink-sampled-adapt's file), 256 tokens, --ignore-eos: md5 of the ids lines, and of each prompt's first 64 ids;
# (2) the Metal tiny suites on the branch: inkling2_batch, inkling2 (INTARSIA_INKLING2_BACKEND=metal), from the test executables
#     endor_build.sh built (no rebuild inside the slot).
set -u
: "${FLEETQ_JOB:?run under fleetq}"
trap 'for c in $(pgrep -P $$ 2>/dev/null); do pkill -P $c 2>/dev/null; kill $c 2>/dev/null; done' EXIT
W=/opt/code/intarsiaml/.worktrees/ink-hip-overlap; S=$W/.scratch; O=$S/metal
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset $v; done
M=$HOME/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/b85c3cfa615fc413cc9af3379f491b963f3b586a
echo "[metal] start $(date +%T); main6 $(cat $S/intarsia-metal-main6.commit | cut -c1-9), o6 $(cat $S/intarsia-metal-o6.commit | cut -c1-9)"
for b in main6 o6; do for sp in off def; do
  x=""; [ $sp = off ] && x="--spec off"
  f=$S/metal-md5-$b-$sp
  timeout 300 $S/intarsia-metal-$b generate -m $M --backend metal --max-seq 8192 --profile-every 0 --show-ids --ignore-eos --max-tokens 256 --prompt-ids $S/metal-md5-prompts.json $x > $f.out 2> $f.err
  rc=$?
  m256=$(grep -a '^ids\[' $f.err | md5 -q)
  m64=$(grep -a '^ids\[' $f.err | python3 -c 'import sys,re
for l in sys.stdin:
    ids=re.findall(r"-?\d+", l.split("]",1)[1] if "]" in l else l)
    print(" ".join(ids[:64]))' | md5 -q)
  echo "[metal] md5 $b $sp rc $rc ids256 $m256 first64 $m64 lines $(grep -ac '^ids\[' $f.err) $(date +%T) | $(grep -aE '^spec: verify' $f.err | cut -c1-100)"
done; done
for t in inkling2_batch inkling2; do
  x=$(ls -t $W/target/release/deps/$t-* 2>/dev/null | grep -v '\.d$' | head -1)
  [ -x "$x" ] || { echo "[metal] tiny $t: no executable"; continue; }
  ( cd $W/crates/intarsia-models && env INTARSIA_INKLING2_BACKEND=metal timeout 330 "$x" --test-threads=1 ) > $S/metal-tiny-$t.txt 2>&1
  echo "[metal] tiny $t rc $? $(date +%T) | $(grep -aE '^test result' $S/metal-tiny-$t.txt | tr '\n' ' ')"
  grep -aE "FAILED|panicked" $S/metal-tiny-$t.txt | head -5
done
echo "[metal] end $(date +%T)"

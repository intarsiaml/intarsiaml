#!/usr/bin/env bash
# Window C (lane ink-hip-spec), card 0: the HIP DSpark default (binary spec3: [tuning.hip] mtp_default_k = 3, cut rule) and the spec-off
# paths the branch changed. Nothing set in the environment. Full logs in logs/wC/.
#   (1) default on: spec3 with no --spec over greedy.json (ids md5 must be 24b7ad602ac6, the --spec off value), stderr names the default
#   (2) a sampled request (--temperature 0.7) under the default: plain steps (spec passes 0)
#   (3) no --max-seq: the context the load reports, default (draft resident) vs --spec off
#   (4) spec off on prompts whose prefill has a 2..=16-row pass (10, 16, 2048+8, 2048+2 ids; chunk 2048): base vs spec3, 64 tokens,
#       ids per prompt and prefill ms (the branch runs those passes under exact_rows on hip, base does not)
#   (5) base --spec off over speed.json, one process (main's column of the report; ids against the branch's off arm)
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-spec
S=$REPO/.scratch
O=$REPO/logs/wC
rm -rf "$O"; mkdir -p "$O"
M=/models/Inkling-2-Mini
BB=/work/.scratch/bin/intarsia-base
B=/work/.scratch/bin/intarsia-${BR:-spec3}
now() { date '+%T'; }
r() { bash "$S/run.sh" "$@"; }
md5ids() { grep -a '^ids\[' "$@" 2>/dev/null | sed 's/^[^:]*://' | md5sum | cut -c1-12; }
echo "[C] (1) default on, greedy.json $(now)"
r $B generate -m $M --backend hip --max-seq 8192 --profile-every 0 --show-ids --max-tokens 128 --prompt-ids /work/.scratch/prompts/greedy.json > $O/dflt.out 2> $O/dflt.err
echo "rc $? md5 $(md5ids $O/dflt.out $O/dflt.err) (spec off: 24b7ad602ac6)"
grep -aE '^--spec dflash|^speculative decoding|^decode overall|^spec: verify|DSpark default|dspark:' $O/dflt.err | cut -c1-300 | head -8
echo "[C] (2) sampled request under the default $(now)"
r $B generate -m $M --backend hip --max-seq 8192 --profile-every 0 --max-tokens 48 --temperature 0.7 --seed 7 -p "$(cat $REPO/scripts/cmx/ids-prompts/short-0.txt)" > $O/temp.out 2> $O/temp.err
echo "rc $?"; grep -aE '^--spec dflash|sampled|plain|^spec: |decode [0-9]+ ms|finish=' $O/temp.out $O/temp.err | sed 's/^[^:]*://' | cut -c1-300 | head -6
echo "[C] (3) context without --max-seq $(now)"
for a in dflt off; do
  [ $a = off ] && X="--spec off" || X=""
  r $B generate -m $M --backend hip --profile-every 0 --max-tokens 8 $X -p "Hello" > $O/ctx-$a.out 2> $O/ctx-$a.err
  echo "ctx $a rc $? | $(grep -aoE 'context [0-9]+ tokens[^)]*' $O/ctx-$a.err | head -1) | $(grep -aE '^--spec dflash' $O/ctx-$a.err | cut -c1-80)"
done
echo "[C] (4) spec off, prompts with a 2..16-row pass $(now)"
for a in base spec3; do
  [ $a = base ] && BIN=$BB || BIN=$B
  r $BIN generate -m $M --backend hip --max-seq 8192 --profile-every 0 --show-ids --ignore-eos --max-tokens 64 --spec off --prompt-ids /work/.scratch/prompts/rows16.json > $O/rows-$a.out 2> $O/rows-$a.err
  echo "rows $a rc $? md5 $(md5ids $O/rows-$a.out $O/rows-$a.err)"
  for i in 0 1 2 3; do
    echo "  [$i] ids $(grep -a "^ids\[$i\]" $O/rows-$a.out $O/rows-$a.err | sed 's/^[^:]*://' | md5sum | cut -c1-12) | $(grep -aE "^prefill\[$i\]" $O/rows-$a.err) | $(grep -aE "^decode\[$i\]" $O/rows-$a.err)"
  done
done
echo "[C] (5) base --spec off over speed.json $(now)"
r $BB generate -m $M --backend hip --max-seq 20480 --profile-every 0 --show-ids --max-tokens 128 --prompt-ids /work/.scratch/prompts/speed.json --spec off > $O/base-off.out 2> $O/base-off.err
echo "rc $? md5 $(md5ids $O/base-off.out $O/base-off.err) | $(grep -aE '^decode overall' $O/base-off.err)"
echo "[C] done $(now)"

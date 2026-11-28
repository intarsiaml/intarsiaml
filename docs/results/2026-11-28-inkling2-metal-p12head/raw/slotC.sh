#!/bin/bash
# endor EXCLUSIVE fleetq slot C, lane ink-metal-p12head: the two bitwise Metal opt-ins together on main 2de468108 + #560
# (both = INTARSIA_DFLASH2_HEAD=p12 + INTARSIA_DFLASH_DEVICE_TOPK=1) against the default, for the morning flip question. Same checks as H2 / H4.
set -u
: "${FLEETQ_JOB:?run under fleetq}"
. $(dirname $0)/lib.sh
B=$S/bin/intarsia-pc
O=$R/C; mkdir -p $O/d; T0=$(date +%s)
echo "slot C start $(date +%T) binary md5 $(md5 -q $B) commit $(cat $B.commit)"
macmon pipe -i 250 > $S/macmon-C.jsonl 2>/dev/null &
G() { env $(envs $1) /usr/bin/time -l $B generate -m $M --backend metal --profile-every 0 --show-ids "${@:2}" $(flags $1); }
echo "== C1: device top-k check mode with the P12 head (every pass: device candidates vs host), speed.json, 256 tokens"
G chk --max-seq 20480 --max-tokens 256 --prompt-ids $R/speed.json > $O/c-chk.out 2> $O/c-chk.err
echo "c-chk rc $? t+$(( $(date +%s) - T0 ))s | $(grep -aE '^spec: verify' $O/c-chk.err | head -1 | cut -c1-140)"
grep -aE '^dflash: DFlash2 draft head' $O/c-chk.err | head -1
echo "topk check lines $(grep -ac 'dflash2 topk check: rank 0 rows' $O/c-chk.err), differing rows total $(grep -a 'dflash2 topk check: rank 0 rows' $O/c-chk.err | awk '{s+=$NF} END {print s+0}')"
echo "== C2: greedy ids, g3.json, 256 tokens, --ignore-eos: off / dflt / both"
for arm in off dflt both; do
  G $arm --max-seq 20480 --max-tokens 256 --ignore-eos --prompt-ids $R/g3.json > $O/g-$arm.out 2> $O/g-$arm.err
  echo "g-$arm rc $? t+$(( $(date +%s) - T0 ))s | $(grep -aE '^spec: verify' $O/g-$arm.err | head -1 | cut -c1-140)"
done
python3 $R/md5.py $O/g-off.err $O/g-dflt.err $O/g-both.err
echo "== C3: decode ABBA x 4, speed.json (19 prompts, 256 tokens, EOS honoured), 45 s idle before each"
for tok in dflt@r1 both@r1 both@r2 dflt@r2 dflt@r3 both@r3 both@r4 dflt@r4; do
  arm=${tok%@*}; f=$O/d/$tok
  sleep 45
  G $arm --max-seq 20480 --max-tokens 256 --prompt-ids $R/speed.json > $f.out 2> $f.err
  echo "$tok rc 0 $(date +%T) t+$(( $(date +%s) - T0 ))s | $(grep -aE '^decode overall' $f.err) | $(grep -aE '^spec: verify' $f.err | head -1 | cut -c1-120)"
done
RULE_BAR=1.01 python3 $R/rule.py $O/d both dflt
echo "slot C end $(date +%T) t+$(( $(date +%s) - T0 ))s"

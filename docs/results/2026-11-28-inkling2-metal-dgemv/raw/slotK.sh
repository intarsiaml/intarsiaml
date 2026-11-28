#!/bin/bash
# endor EXCLUSIVE fleetq slot K, lane ink-metal-dgemv: gemv_g0 on the Inkling draft shapes at 8 rows (every candidate kernel interleaved
# per round, multi-row forms checked bitwise against 8 serving 1-row launches first), macmon logged.
set -u
: "${FLEETQ_JOB:?run under fleetq}"
. $(dirname $0)/lib.sh
O=$R/K; mkdir -p $O; T0=$(date +%s)
echo "slot K start $(date +%T) gemv_g0 md5 $(md5 -q $S/bin/gemv_g0)"
macmon pipe -i 250 > $S/macmon-K.jsonl 2>/dev/null &
env G0_ROWS=7,8 G0_ROUNDS=20 G0_OUT=$O/g0.csv $S/bin/gemv_g0 ink_gu ink_gup ink_down ink_q ink_kv > $O/g0.out 2> $O/g0.err
echo "gemv_g0 rc $? t+$(( $(date +%s) - T0 ))s"
cat $O/g0.err | grep -vE '^\s*$' | tail -80
echo "slot K end $(date +%T) t+$(( $(date +%s) - T0 ))s"

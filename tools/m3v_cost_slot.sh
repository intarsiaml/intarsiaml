#!/bin/zsh
# M3 stage 0 cost slot: m3_verify_cost (plain batched steps, solo verify passes, today's iterations) and g0b_bench attn at
# the same contexts, with macmon at 100 ms. Run inside an exclusive fleetq slot on endor:
#   fleetq run --host endor --exclusive --mem 24 --label "..." -- tools/m3v_cost_slot.sh OUTDIR
# env: EX (m3_verify_cost binary), G0B (g0b_bench binary), CTX (512,1024,4096), ROUNDS (10), REPS (12)
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a slot" >&2; exit 2; }
OUT=${1:?outdir}; mkdir -p $OUT; OUT=${OUT:A}
EX=${EX:?}; G0B=${G0B:?}; CTX=${CTX:-512,1024,4096}; ROUNDS=${ROUNDS:-10}; REPS=${REPS:-12}
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
MPID=""
trap '[ -n "$MPID" ] && kill $MPID 2>/dev/null' EXIT
trap "exit 143" INT TERM
echo "slot $FLEETQ_JOB $(date); ex $(shasum -a 256 $EX | cut -c1-16) g0b $(shasum -a 256 $G0B | cut -c1-16); ctx $CTX rounds $ROUNDS reps $REPS"
ps -axo pid,rss,%cpu,etime,command | grep -iE "kev|tensorfold|omlx|splash|llama|vllm|mlx|ollama|intarsia|python" | grep -v "grep\|ps -axo" | cut -c1-170 > $OUT/ps-start.txt
macmon pipe -i 100 > $OUT/macmon.jsonl 2>/dev/null &
MPID=$!
sleep 1
echo "== m3_verify_cost $(date +%T)"
$EX --ctx $CTX --rounds $ROUNDS --reps $REPS > $OUT/cost.csv 2> $OUT/cost.err; echo "rc=$?"
echo "== g0b_bench attn $(date +%T)"
G0B_OUT=$OUT/g0b-attn.csv G0B_ROUNDS=20 $G0B attn ${=CTX//,/ } > $OUT/g0b-attn.out 2>&1; echo "rc=$?"
kill $MPID 2>/dev/null; MPID=""
gzip -f $OUT/macmon.jsonl
pgrep -fl "m3_verify_cost|g0b_bench" || echo "no runner left"
echo "done $(date)"

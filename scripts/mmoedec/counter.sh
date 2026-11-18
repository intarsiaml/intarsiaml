#!/bin/zsh
# mmoedec: dispatch proof. The temporary counter build (build.sh COUNTER=1, raw/count.patch) prints one `MOECNT rows= t= gu= dn=` line per MoE experts launch;
# a short greedy `generate --spec mtp --spec-k 2` with the knob on and off, GDN verify fuse on, counts the launches by (rows, T, kernel).
#   fleetq run --host endor --quiet gpu,cpu --mem 40 --expect 10 --label "mmoedec: counter proof" -- scripts/mmoedec/counter.sh
source ${0:A:h}/lib.sh
CNT=$W/target-mmoedec-cnt/intarsia-cnt; O=$RAW/counter; mkdir -p $O
for arm in off on; do
  E=(MOECNT=1 INTARSIA_METAL_GDN_VERIFY_FUSE=1); [ $arm = on ] && E+=(INTARSIA_METAL_MOE_BITDEC=1)
  echo "-- $arm $(cool)"
  env $E $CNT generate -m $M --backend metal --max-seq 4096 --spec mtp --spec-k 2 --chat --no-think --temperature 0 --profile-every 0 --show-ids \
    -p "$(cat $CELLS_DIR/${CELL:-chat-short}.txt)" --max-tokens ${NTOK:-300} < /dev/null > $O/gen-$arm.out 2> $O/gen-$arm.err
  echo "exit $?"; grep -a '^ids:' $O/gen-$arm.err > $O/gen-$arm.ids
  grep -a '^MOECNT' $O/gen-$arm.err | sort | uniq -c | sort -k3,3 -k4,4 > $O/count-$arm.txt; cat $O/count-$arm.txt
done
cmp $O/gen-off.ids $O/gen-on.ids && echo "ids identical off/on ($(tr ',' '\n' < $O/gen-on.ids | wc -l | tr -d ' ') ids)" || echo "IDS DIFFER"

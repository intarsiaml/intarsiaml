#!/bin/zsh
# mcopylen: the result tables from raw/ab (scripts/mcopy/tables.py), against base, against n20L7 and against n20L15, plus the thermal levels per arm.
W=${0:A:h:h:h}; R=$W/docs/results/2026-11-18-mcopylen/raw
for b in base n20L7 n20L15; do python3 $W/scripts/mcopy/tables.py $R/ab $b > $R/table-vs-$b.md; done
for a in base n20L7 n20L15 n20L15a; do echo "$a thermal (before->after, per cell):" $(cat $R/ab-run-r*.log | grep -E "^$a r[0-9] .* thermal" | sed 's/.*thermal \([0-9]*\)->\([0-9]*\):.*/\1->\2/' | sort | uniq -c | tr '\n' ';'); done > $R/thermal.txt
cat $R/thermal.txt

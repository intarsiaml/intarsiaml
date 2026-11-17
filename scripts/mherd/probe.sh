#!/bin/zsh
# probe: one server (store on, max-batch 8), time a solo 8K prompt and watch RSS. fleetq run --host endor ... -- scripts/mherd/probe.sh
source ${0:A:h}/lib.sh; D=$R/probe; mkdir -p $D
start $D - --max-seq 9728 --max-batch 8 --prefix-store on --spec off || exit 1
echo "up; rss MB $(ps -o rss= -p $TP | awk '{print int($1/1024)}')"
python3 $S/../herd/herd_bench.py --url $URL --model intarsia --arm probe --mode solo --seed 901 --n 1 --shared 4096 --own 100 --max-tokens 16 --out $D/p.jsonl
python3 $S/../herd/herd_bench.py --url $URL --model intarsia --arm probe --mode solo --seed 902 --n 1 --shared 8192 --own 100 --max-tokens 16 --out $D/p.jsonl
echo "rss MB $(ps -o rss= -p $TP | awk '{print int($1/1024)}')"; vm_stat | head -4
stop $D

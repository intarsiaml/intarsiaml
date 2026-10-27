#!/usr/bin/env bash
# Lane spf: SM clock / power while (a) the engine prefills 16k tokens of the 8B and (b) lt_sweep runs, same slot.
#   fleetq run --host spark1 --quiet gpu,cpu --mem 45 --expect 6 --label "spf: clocks" -- ssh spark1 'bash -s' < scripts/spf/clk.sh
export PATH=/usr/local/cuda/bin:$PATH
P=$HOME/intarsiaml-spf/base; B=$P/target/release/intarsia; O=$P/out/clk; mkdir -p $O
M=$(ls -d /home/nvidia/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/* | head -1)
head -c 55000 $P/tools/prompts/corpus_docs.txt > $O/p.txt
Q="timestamp,clocks.sm,clocks.max.sm,power.draw,clocks_throttle_reasons.active,temperature.gpu"
nvidia-smi --query-gpu=$Q --format=csv -lms 250 > $O/engine.csv &
S=$!
$B generate -m $M --backend cuda --mem-budget 40 --max-seq 20480 --prefill-chunk 4096 --profile-every 0 --max-tokens 1 -p "$(cat $O/p.txt)" </dev/null 2>&1 | grep -aE "finish="
kill $S
nvidia-smi --query-gpu=$Q --format=csv -lms 250 > $O/micro.csv &
S=$!
~/intarsiaml-spf/lt/lt_sweep 4096 | grep -E "cublasGemmEx"
kill $S
for f in engine micro; do echo "== $f"; awk -F, 'NR>1 && $2+0>300 {n++; c+=$2; p+=$4; if (min==""||$2<min) min=$2; if ($2>max) max=$2} END {print n" samples (sm>300MHz) mean sm "c/n" MHz, min "min", max "max", mean power "p/n" W"}' $O/$f.csv; done
tail -3 $O/engine.csv

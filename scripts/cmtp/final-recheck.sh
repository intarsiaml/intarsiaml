#!/usr/bin/env bash
# Lane cmtp: re-check on the merged head (after origin/main's lone --spec-k change): ids off vs default on six prompts, then a lone `--spec-k 3`.
#   fleetq run --host spark1 --mem 50 --expect 25 --label "cmtp: recheck merged head" -- ssh spark1 'bash -s' < scripts/cmtp/final-recheck.sh
R=$HOME/intarsiaml-cmtp; H=/home/nvidia/.cache/huggingface/hub; M35=$(ls -d $H/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/* | head -1)
O=$R/out-recheck; mkdir -p $O
O=$O ARMS='off auto' bash $R/src/scripts/cmtp/ids.sh short-0 short-3 4k-0 4k-2 16k-0 16k-3
P=$R/src/scripts/cmtp/ids-prompts
$R/target-dev/release/intarsia generate -m $M35 --backend cuda --mem-budget 40 --max-seq 4096 --spec-k 3 --chat --no-think --temperature 0 --profile-every 0 --ignore-eos --show-ids \
  -p "$(cat $P/short-0.txt)" --max-tokens 256 < /dev/null > /dev/null 2> $O/gen-short-0-lonek3.err
grep -a '^ids:' $O/gen-short-0-lonek3.err > $O/gen-short-0-lonek3.ids
echo "lone --spec-k 3: $(grep -a 'default of recipe' $O/gen-short-0-lonek3.err | cut -c1-120) | $(grep -a '^spec:' $O/gen-short-0-lonek3.err | tail -1 | cut -c1-100)"
cmp -s $O/gen-short-0-off.ids $O/gen-short-0-lonek3.ids && echo "PASS lone --spec-k 3 ids == off" || echo "FAIL lone --spec-k 3 ids differ"

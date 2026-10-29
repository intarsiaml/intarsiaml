#!/usr/bin/env bash
# lane arx: HIP no-op check of main bfbee3cb6 (dev) vs 200be3f05 (base): plan show 8B / 35B tp2, 35B tp2 greedy ids short-0 / 4k-0 spec off
source ~/intarsiaml-arx2/scripts/arx/lib.sh
O=$R/arx-out/noop
trap restore EXIT; stop_radiance
M35=/models/intarsia/Qwen3.6-35B-A3B-NVFP4
for b in base dev; do
  G=0 MODELS=/data/models/intarsia gpu 90 /work/arx-$b-cli plan show --backend hip --model /models/Qwen3-8B --mem-budget 29 --max-seq 4096 --json > $O/plan8-$b.json 2> $O/plan8-$b.err
  MODELS=/data/models gpu 120 /work/arx-$b-cli plan show --backend hip --model $M35 --tp 2 --devices 0,1 --mem-budget 29 --max-seq 4096 --json > $O/plan35-$b.json 2> $O/plan35-$b.err
  for p in short-0 4k-0; do
    MODELS=/data/models gpu 120 /work/arx-$b-cli generate -m $M35 --backend hip --tp 2 --devices 0,1 --mem-budget 29 --max-seq 8192 --chat --no-think \
      --profile-every 0 --ignore-eos --show-ids --spec off --max-tokens 256 -p "$(cat $R/scripts/dfh/ids-prompts/$p.txt)" > /dev/null 2> $O/ids-$p-$b.err
    grep -a '^ids' $O/ids-$p-$b.err > $O/ids-$p-$b.ids
  done
done
for f in plan8 plan35; do echo "== $f: $(wc -c < $O/$f-base.json) / $(wc -c < $O/$f-dev.json) bytes, $(cmp -s $O/$f-base.json $O/$f-dev.json && echo IDENTICAL || echo DIFFER)"; done
for p in short-0 4k-0; do echo "== ids $p: $(wc -w < $O/ids-$p-base.ids) words, $([[ -s $O/ids-$p-base.ids ]] && cmp -s $O/ids-$p-base.ids $O/ids-$p-dev.ids && echo IDENTICAL || echo DIFFER)"; done

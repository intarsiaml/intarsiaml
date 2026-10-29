#!/bin/bash
# greedy ids gate: main vs dev, 4 prompts, 256 tokens (+ MTP spec on short & 4k). Run inside a fleetq slot.
[ -n "${FLEETQ_JOB:-}" ] || { echo "not in slot"; exit 2; }
W=/opt/code/intarsiaml/.worktrees/mnq; O=$W/.lane/g1; M=$HOME/models/Qwen3.6-35B-A3B-NVFP4
run() { # tag bin prompt extra...
  local tag=$1 bin=$2 pf=$3; shift 3
  $bin generate --model $M --backend metal --max-seq 20000 --profile-every 0 --temperature 0 --max-tokens 256 --ignore-eos "$@" --prompt "$(cat $pf)" > $O/$tag.out 2> $O/$tag.err
  echo "$tag $(md5 -q $O/$tag.out) $(grep -iE 'decode' $O/$tag.err | tail -1 | cut -c1-120)"
}
for p in short p4k-a p4k-b p16k; do
  run $p-main $W/.lane/bin/intarsia-main $W/.lane/prompts/$p.txt
  INTARSIA_METAL_NORM_FP8=1 run $p-dev1 $W/.lane/bin/intarsia-dev $W/.lane/prompts/$p.txt
  run $p-dev $W/.lane/bin/intarsia-dev $W/.lane/prompts/$p.txt
done
for p in short p4k-a; do
  INTARSIA_METAL_NORM_FP8=1 run $p-dev1-mtp $W/.lane/bin/intarsia-dev $W/.lane/prompts/$p.txt --spec mtp
  run $p-main-mtp $W/.lane/bin/intarsia-main $W/.lane/prompts/$p.txt --spec mtp
done
# EOS stop (no --ignore-eos): token feedback discards its held steps when the sequence ends
for b in main dev; do
  INTARSIA_METAL_NORM_FP8=1 $W/.lane/bin/intarsia-$b generate --model $M --backend metal --max-seq 4096 --profile-every 0 --temperature 0 --max-tokens 200 --prompt "Reply with exactly one word: yes" > $O/eos-$b.out 2> $O/eos-$b.err
  echo "eos-$b $(md5 -q $O/eos-$b.out) $(wc -c < $O/eos-$b.out) bytes"
done
echo done

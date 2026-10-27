#!/usr/bin/env bash
# G4X G0 on CUDA (run on spark2 INSIDE a fleetq slot, --mem 24): `plan show` (text and --json) of the 8B and the 35B for each binary; every
# output must be byte-identical to the base build's.   g0.sh <binary-name> ...   (first = base)
R=$HOME/intarsiaml-g4x; O=$R/replay-out/g0; mkdir -p $O
M8=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
M35=$HOME/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
for b in "$@"; do
  for m in 8b:$M8 35b:$M35; do
    n=${m%%:*}; p=${m#*:}
    $R/replay-out/bins/$b plan show -m $p --backend cuda --max-seq 4096 --mem-budget 24 > $O/$b-$n.txt 2>$O/$b-$n.err
    $R/replay-out/bins/$b plan show -m $p --backend cuda --max-seq 4096 --mem-budget 24 --json > $O/$b-$n.json 2>>$O/$b-$n.err
  done
done
base=$1; shift
for b in "$@"; do for n in 8b 35b; do for e in txt json; do
  cmp -s $O/$base-$n.$e $O/$b-$n.$e && echo "$b $n $e SAME $(wc -c < $O/$b-$n.$e) B" || echo "$b $n $e DIFFERENT"
done; done; done

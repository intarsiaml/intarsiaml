#!/usr/bin/env bash
# ON tatooine, inside burst.sh (radiance down). M3: HIP main build, 35b-long dumps, reuse diag at chunk 1024 / 4096,
# KL21 against two fresh vLLM samples of the long golden.
set -u
cd ~/intarsiaml-m3
export INTARSIA_REPO=$HOME/intarsiaml-m3 INTARSIA_TARGET_VOL=intarsia-cargo-target-m3 HIP_VISIBLE_DEVICES=0
D=deploy/tatooine/dev.sh; B=/target/release/intarsia; M=/models/Qwen3.6-35B-A3B-NVFP4
O=/work/m3out; mkdir -p ~/intarsiaml-m3/m3out
A="accuracy --backend hip --model $M --mem-budget 29 --max-seq 16384 --allow-unstamped --allow-dirty"
t() { echo "[m3] $1 $(date +%T)"; }
t "long c4096 prefill dump"
INTARSIA_ACC_POS_DUMP=$O/long-c4096.pos.tsv $D $B $A --golden /models/golden/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors --mode prefill --prefill-chunk 4096 --write-self-golden $O/long-c4096.dump.safetensors > ~/intarsiaml-m3/m3out/long-c4096.log 2>&1
grep -E "^OVERALL|^prefill floor|^chat_reuse" ~/intarsiaml-m3/m3out/long-c4096.log
for c in 1024 4096; do
  t "reuse diag c$c"
  $D $B $A --golden /work/m3data/docs4k.safetensors --mode reuse --reuse-diag --prefill-chunk $c > ~/intarsiaml-m3/m3out/reuse-docs4k-c$c.log 2>&1
  grep -E "^long_docs_4k|prefix-reuse gate" ~/intarsiaml-m3/m3out/reuse-docs4k-c$c.log
done
for g in long-A long-B; do
  t "vs $g"
  $D $B $A --golden /work/m3data/$g.safetensors --mode prefill --prefill-chunk 4096 > ~/intarsiaml-m3/m3out/acc-$g.log 2>&1
  grep -E "^OVERALL|^prefill floor|^chat_reuse" ~/intarsiaml-m3/m3out/acc-$g.log
done
t "done"

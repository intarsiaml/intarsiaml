#!/usr/bin/env bash
# ON tatooine, inside hip_burst_m3.sh (radiance down). M3 window 2: reuse diag at c1024 with --no-graphs (does the engine
# then match its scripted replay?), and the tool-exact per-row KL21 dump of the 35b-long prefill on main.
set -u
cd ~/intarsiaml-m3
export INTARSIA_REPO=$HOME/intarsiaml-m3 INTARSIA_TARGET_VOL=intarsia-cargo-target-m3 HIP_VISIBLE_DEVICES=0
D=deploy/tatooine/dev.sh; B=/target/release/intarsia; M=/models/Qwen3.6-35B-A3B-NVFP4
A="accuracy --backend hip --model $M --mem-budget 29 --max-seq 16384 --allow-unstamped --allow-dirty"
t() { echo "[m3] $1 $(date +%T)"; }
t "reuse diag c1024 --no-graphs"
$D $B $A --golden /work/m3data/docs4k.safetensors --mode reuse --reuse-diag --no-graphs --prefill-chunk 1024 > ~/intarsiaml-m3/m3out/reuse-docs4k-c1024-nographs.log 2>&1
grep -E "^long_docs_4k|engine|prefix-reuse gate" ~/intarsiaml-m3/m3out/reuse-docs4k-c1024-nographs.log
t "long c4096 prefill pos dump"
$D env INTARSIA_ACC_POS_DUMP=/work/m3out/long-c4096.pos.tsv $B $A --golden /models/golden/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors --mode prefill --prefill-chunk 4096 > ~/intarsiaml-m3/m3out/long-c4096-pos.log 2>&1
grep -E "^OVERALL" ~/intarsiaml-m3/m3out/long-c4096-pos.log; wc -l ~/intarsiaml-m3/m3out/long-c4096.pos.tsv
t "done"

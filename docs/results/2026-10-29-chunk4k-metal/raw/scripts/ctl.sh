#!/bin/bash
W=/opt/code/intarsiaml/.worktrees/chunk4k; cd $W
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer CARGO_TARGET_DIR=$W/.lane/target INTARSIA_REPO=$W
cargo build --release --locked -p intarsia-cli --features metal 2>&1 | tail -1
cp .lane/target/release/intarsia .lane/intarsia-main
cp .lane/recipe-new.toml recipes/qwen3_5_moe.toml
for b in intarsia-main intarsia-metal; do
 echo "== $b"
 .lane/$b accuracy --model $HOME/Models/Qwen3.6-35B-A3B-NVFP4 --backend metal --device 0 --mem-budget 40 --golden golden/qwen36-35b-a3b-nvfp4-vllm-4k-top20.safetensors --mode reuse --decode-positions 64 --max-seq 8192 --prefill-chunk 1024 --reuse-require-all --allow-unstamped --allow-same-code </dev/null > .lane/ctl-$b.out 2>&1; echo "exit $?"
 grep -a -E "FAIL|prefix reuse:|^long_code_4k|^long_docs_4k" .lane/ctl-$b.out | cut -c1-220 | head -12
done

#!/bin/bash
W=/opt/code/intarsiaml/.worktrees/chunk4k; cd $W; export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer CARGO_TARGET_DIR=$W/.lane/target INTARSIA_REPO=$W
cargo build --release --locked -p intarsia-cli --features metal 2>&1 | tail -1
cp .lane/target/release/intarsia .lane/intarsia-final
G=$W/golden/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors; OR=$W/golden/oracle/qwen36-35b-a3b-nvfp4-cpu-oracle.safetensors
.lane/intarsia-final qualify --backend metal --golden $G --oracle-golden $OR --model-name Qwen3.6-35B-A3B-NVFP4 --mem-budget 40 --stage-timeout-min 25 --model $HOME/Models/Qwen3.6-35B-A3B-NVFP4 --out $W/.lane/final-q \
  --skip kernels --skip gate --skip bench --skip bench-long --skip ab --skip tp --skip acc-oracle --skip acc-long --skip acc-4k --skip envelope </dev/null 2>&1 | grep -E "\] [a-z0-9-]+: (PASS|FAIL|SKIP|ERROR|WARN|MISSING)" | cut -c1-300

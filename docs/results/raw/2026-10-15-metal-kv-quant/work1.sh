#!/bin/zsh
# slot 1: fp8 kernel tests + attn_bw bandwidth check
source /opt/code/intarsiaml-kvq/docs/results/raw/2026-10-15-metal-kv-quant/common.sh
cd $W
mkdir -p $R/slot1
cargo test -p intarsia-backend-metal --release --test kv_fp8 -- --nocapture --test-threads=1 > $R/slot1/kv_fp8.txt 2>&1; echo "kv_fp8 exit $?"
grep -E "^test |test result|panicked|decode f8|rows f8|fp8 encode" $R/slot1/kv_fp8.txt | cut -c1-260
macmon_start $R/slot1/macmon.jsonl
for ctx in 4096 16384 32768; do
  $W/target/release/examples/attn_bw splitkv f8 $ctx 2>&1 | tee -a $R/slot1/attn_bw.txt
  $W/target/release/examples/attn_bw f8 splitkv $ctx 2>&1 | tail -2 | tee -a $R/slot1/attn_bw.txt
done
macmon_stop

#!/bin/zsh
# slot 1c: fp8 split-KV decode kernel, units-in-flight variants (INTARSIA_METAL_F8_U), vs bf16, microbench
source /opt/code/intarsiaml-kvq/docs/results/raw/2026-10-15-metal-kv-quant/common.sh
cd $W; mkdir -p $R/slot1c
macmon_start $R/slot1c/macmon.jsonl
for ctx in 4096 16384 32768; do
  for U in 4 2 8 4; do
    echo "U=$U" | tee -a $R/slot1c/attn_bw.txt
    INTARSIA_METAL_F8_U=$U $W/target/release/examples/attn_bw f8 $ctx 2>&1 | tail -1 | tee -a $R/slot1c/attn_bw.txt
  done
  $W/target/release/examples/attn_bw splitkv $ctx 2>&1 | tail -1 | tee -a $R/slot1c/attn_bw.txt
done
macmon_stop
python3 - <<'PY'
import json,statistics as st
f=[json.loads(l)["gpu_usage"][0] for l in open("/opt/code/intarsiaml-kvq/docs/results/raw/2026-10-15-metal-kv-quant/slot1c/macmon.jsonl") if l.strip().startswith("{")]
b=[x for x in f if x>400]
print("macmon busy-sample median MHz", st.median(b) if b else None, "n", len(b))
PY

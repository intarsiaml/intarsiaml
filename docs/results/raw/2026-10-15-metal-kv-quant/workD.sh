#!/bin/zsh
# slot D: int8 (K+V, block 32) kernel tests, more fake-quant probe configs, real int8 + fp8 accuracy, int8 ids / spec, attn_bw
source /opt/code/intarsiaml-kvq/docs/results/raw/2026-10-15-metal-kv-quant/common.sh
NEW=/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/intarsia-i8
mkdir -p $R/slotD $R/probe $R/acc $R/ids
cd $W
macmon_start $R/slotD/macmon.jsonl
trap macmon_stop EXIT
echo "== 1. kv_i8 + kv_fp8 tests"
cargo test -p intarsia-backend-metal --release --test kv_i8 --test kv_fp8 -- --nocapture --test-threads=1 > $R/slotD/kv_i8.txt 2>&1; echo "exit $?"
grep -E "^test |test result|panicked|decode i8|rows i8|differ|bytes" $R/slotD/kv_i8.txt | cut -c1-260
echo "== 2. more fake-quant probe configs (block,mask): K-only 8/16, K+V 8, V-only 8/128"
for cfg in "8,1" "8,3" "16,1" "4,1" "128,2"; do
  tag=${cfg//,/_}
  env INTARSIA_METAL_KV_FAKEQ=$cfg $NEW accuracy --model $M --backend metal --kv-dtype bf16 --golden $G/qwen3-8b-long-fp32.safetensors --gate-class bf16 --mode all --max-seq 8192 > $R/probe/long-$tag.out 2> $R/probe/long-$tag.err; echo "probe $cfg long exit $?"
  env INTARSIA_METAL_KV_FAKEQ=$cfg $NEW accuracy --model $M --backend metal --kv-dtype bf16 --golden $G/qwen3-8b-fp32.safetensors --gate-class bf16 --max-seq 4096 > $R/probe/short-$tag.out 2> $R/probe/short-$tag.err; echo "probe $cfg short exit $?"
done
echo "== 3. attn_bw int8 vs bf16 vs fp8"
for ctx in 4096 16384 32768; do
  /private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/attn_bw-i8 splitkv i8 f8 $ctx 2>&1 | tail -3 | tee -a $R/slotD/attn_bw.txt
done
echo "== 4. real int8 accuracy vs fp32 goldens (bf16 gate class); 4k for bf16, fp8, int8 without baseline files (unstamped build)"
for KV in int8; do
  $NEW accuracy --model $M --backend metal --kv-dtype $KV --golden $G/qwen3-8b-fp32.safetensors --gate-class bf16 --max-seq 4096 > $R/acc/$KV-short.out 2> $R/acc/$KV-short.err; echo "$KV short exit $?"
  $NEW accuracy --model $M --backend metal --kv-dtype $KV --golden $G/qwen3-8b-long-fp32.safetensors --gate-class bf16 --mode all --max-seq 8192 > $R/acc/$KV-long.out 2> $R/acc/$KV-long.err; echo "$KV long exit $?"
done
for KV in bf16 fp8 int8; do
  $NEW accuracy --model $M --backend metal --kv-dtype $KV --golden $G/qwen3-8b-4k-fp32-top20.safetensors --gate-class bf16 --mode all --decode-positions 64 --max-seq 8192 > $R/acc/$KV-4k.out 2> $R/acc/$KV-4k.err; echo "$KV 4k exit $?"
done
python3 $R/acc_table.py
echo "== 5. int8 greedy ids vs bf16, spec ngram vs off"
typeset -A FL PF
FL=(open-256 "--chat --no-think" grounded-256 "--chat --no-think" golden-docs-256 "--ignore-eos" golden-code-256 "--ignore-eos" 4k-256 "--ignore-eos" short-256 "--ignore-eos")
PF=(open-256 open grounded-256 grounded golden-docs-256 golden-docs golden-code-256 golden-code 4k-256 4k short-256 short)
for c in short-256 4k-256 open-256 grounded-256 golden-docs-256 golden-code-256; do
  $NEW generate --model $M --backend metal --max-seq 8192 --profile-every 0 --temperature 0 --show-ids --max-tokens 256 ${=FL[$c]} --kv-dtype int8 --prompt "$(cat $PG/${PF[$c]}.txt)" < /dev/null > $R/ids/i8-$c.out 2> $R/ids/i8-$c.err
  python3 -I $W/docs/results/raw/2026-10-14-metal-prefill-p1/full/first_divergence.py $R/ids/new-$c.err $R/ids/i8-$c.err "int8 KV vs bf16 KV, $c" | tee -a $R/ids/summary.txt
  $NEW generate --model $M --backend metal --max-seq 8192 --profile-every 0 --temperature 0 --show-ids --max-tokens 256 ${=FL[$c]} --kv-dtype int8 --spec ngram --prompt "$(cat $PG/${PF[$c]}.txt)" < /dev/null > $R/ids/i8spec-$c.out 2> $R/ids/i8spec-$c.err
  python3 -I $W/docs/results/raw/2026-10-14-metal-prefill-p1/full/first_divergence.py $R/ids/i8-$c.err $R/ids/i8spec-$c.err "int8 KV: spec ngram vs off, $c" | tee -a $R/ids/summary.txt
done
echo "done $(date)"

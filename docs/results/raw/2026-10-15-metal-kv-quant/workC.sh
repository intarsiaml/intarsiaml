#!/bin/zsh
# slot C: 16k+ proxy and localeval, bf16 vs fp8 vs int8
source /opt/code/intarsiaml-kvq/docs/results/raw/2026-10-15-metal-kv-quant/common.sh
NEW=/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/intarsia-i8
O=$R/speed2; mkdir -p $O
export INTARSIA_METAL_F8_U=4
macmon_start $O/macmon-C.jsonl
trap macmon_stop EXIT
echo "== 16k+ proxy (no fp32 golden exists): last-position top-50 logits and 128 greedy ids, bf16 vs fp8 KV"
mkdir -p $R/long16k
for ctx in 16384 32764; do
  for kv in bf16 fp8 int8; do
    $NEW generate --model $M --backend metal --max-seq 40960 --temperature 0 --max-tokens 129 --ignore-eos --profile-every 0 --prefill-chunk 2048 --show-logits 50 --show-ids --kv-dtype $kv --prompt "$(cat $PL/prompt-$ctx.txt)" > $R/long16k/$kv-$ctx.out 2> $R/long16k/$kv-$ctx.err
  done
  for kv in fp8 int8; do python3 -I $R/long_cmp.py $R/long16k/bf16-$ctx.err $R/long16k/$kv-$ctx.err "$kv $ctx" | tee -a $R/long16k/summary.txt; done
done

echo "== localeval speed through intarsia serve, bf16 then fp8 then bf16... (one server per dtype), sizes 4096 16384"
for kv in bf16 fp8 int8 int8 fp8 bf16; do
  LD=$R/localeval/$kv-$(date +%H%M%S); mkdir -p $LD
  cooldown $O/macmon-C.jsonl "localeval-$kv"
  $NEW serve --model $M --backend metal --device 0 --host 127.0.0.1 --port 8791 --served-model-name intarsia-metal --max-seq 20480 --profile-every 0 --kv-dtype $kv --no-supervisor --no-ui > $LD/server.out 2> $LD/server.err &
  SPID=$!
  for i in $(seq 1 120); do curl -sf http://127.0.0.1:8791/health >/dev/null 2>&1 && break; kill -0 $SPID 2>/dev/null || { echo server died; break; }; sleep 2; done
  grep -E "kv|KV" $LD/server.err | head -5 > $LD/kv-lines.txt
  (cd /opt/code/deepeval && uv run localeval speed --url http://127.0.0.1:8791/v1 --model intarsia-metal --sizes 4096,16384 --gen 256 --reps 3 --thinking off --label "metal-kvq-$kv") > $LD/localeval.out 2> $LD/localeval.err
  echo "localeval $kv exit $?"; tail -12 $LD/localeval.out
  kill $SPID; wait $SPID 2>/dev/null
done

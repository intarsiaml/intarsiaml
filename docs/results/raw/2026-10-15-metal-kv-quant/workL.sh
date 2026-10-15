#!/bin/zsh
# localeval speed (sizes 4096 and 16384) through intarsia serve, one server per KV dtype, order bf16 fp8 int8 int8 fp8 bf16
source /opt/code/intarsiaml-kvq/docs/results/raw/2026-10-15-metal-kv-quant/common.sh
echo "######## merged-tree gates (workM) $(date)"; zsh /opt/code/intarsiaml-kvq/docs/results/raw/2026-10-15-metal-kv-quant/workM.sh
echo "######## localeval on merged binary $(date)"
NEW=/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/intarsia-merged
O=$R/speed4; mkdir -p $O $R/localeval2
export INTARSIA_METAL_F8_U=4
macmon_start $O/macmon-L.jsonl
trap macmon_stop EXIT
for kv in bf16 fp8 int8 int8 fp8 bf16; do
  LD=$R/localeval2/$kv-$(date +%H%M%S); mkdir -p $LD
  COOL=62 COOLN=12 cooldown $O/macmon-L.jsonl "localeval-$kv"
  $NEW serve --model $M --backend metal --device 0 --host 127.0.0.1 --port 8791 --served-model-name intarsia-metal --max-seq 20480 --profile-every 0 --kv-dtype $kv --no-supervisor --no-ui > $LD/server.out 2> $LD/server.err &
  SPID=$!
  for i in $(seq 1 120); do curl -sf http://127.0.0.1:8791/health >/dev/null 2>&1 && break; kill -0 $SPID 2>/dev/null || { echo server died; break; }; sleep 2; done
  grep -E "context|kv|KV" $LD/server.err | head -5 > $LD/kv-lines.txt
  S0=$(date +%s)
  (cd /opt/code/deepeval && uv run localeval speed --url http://127.0.0.1:8791/v1 --model intarsia-metal --sizes 4096,16384 --gen 256 --reps 3 --thinking off --label "metal-kvq-$kv") > $LD/localeval.out 2> $LD/localeval.err
  echo "localeval $kv exit $? ($(( $(date +%s) - S0 )) s) $LD"; cat $LD/kv-lines.txt | head -2 | cut -c1-160
  kill $SPID; wait $SPID 2>/dev/null
  sleep 3
done

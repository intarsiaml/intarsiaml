#!/bin/zsh
# slot 3: engine-side decode / prefill, bf16 vs fp8 KV, ABBA per context, GPU clock logged
source /opt/code/intarsiaml-kvq/docs/results/raw/2026-10-15-metal-kv-quant/common.sh
export INTARSIA_METAL_F8_U=$(pickU); echo "INTARSIA_METAL_F8_U=$INTARSIA_METAL_F8_U"
zmodload zsh/datetime
O=$R/speed; mkdir -p $O
cd $W
macmon_start $O/macmon.jsonl
trap macmon_stop EXIT
{ echo "date: $(date -Iseconds)"; echo "fleetq job: $FLEETQ_JOB"; echo "head: $(git rev-parse HEAD)"; ls -l $NEW; shasum -a 256 $NEW; $MACMON --version; pmset -g therm; } > $O/versions.txt 2>&1
: > $O/runs.tsv
run() { # <label> <ctx> <kv> [env...]
  local lab=$1 ctx=$2 kv=$3; shift 3
  local s=$EPOCHREALTIME
  env "$@" $NEW generate --model $M --backend metal --max-seq 40960 --temperature 0 --max-tokens 129 --ignore-eos --profile-every 0 \
    --prefill-chunk 2048 --kv-dtype $kv --prompt "$(cat $PL/prompt-$ctx.txt)" > $O/$lab.out 2> $O/$lab.err
  printf "%.3f\t%.3f\t%s\n" $s $EPOCHREALTIME "$lab" >> $O/runs.tsv
  echo "$lab [$(date +%H:%M:%S)]: $(grep -o 'prefill [0-9]* ms ([0-9.]* tok/s) | decode [0-9]* ms ([0-9.]* tok/s)' $O/$lab.err | tail -1) T=$(gpu_temp $O/macmon.jsonl)C"
}
echo "== warmup (page-in)"; run warm 4096 bf16 NOENV=1
: > $O/runs.tsv
for ctx in 4096 16384 32764 40000; do
  i=0
  for kv in bf16 fp8 fp8 bf16; do
    i=$((i+1)); COOL=72 cooldown $O/macmon.jsonl "$ctx-$kv-$i" | tee -a $O/temps.log
    run ctx$ctx-$i-$kv $ctx $kv NOENV=1
  done
done
echo "== kernel-mode trace: which kernels run under fp8 (4k, 33 decode steps)"
run trace-kernel-fp8-4k 4096 fp8 INTARSIA_METAL_PROFILE=kernel:$O/trace-kernel-fp8-4k.jsonl
python3 $R/summ.py $O $O/macmon.jsonl | tee $O/summary.txt

echo "== 16k+ proxy (no fp32 golden exists): last-position top-50 logits and 128 greedy ids, bf16 vs fp8 KV"
mkdir -p $R/long16k
for ctx in 16384 32764; do
  for kv in bf16 fp8; do
    $NEW generate --model $M --backend metal --max-seq 40960 --temperature 0 --max-tokens 129 --ignore-eos --profile-every 0 --prefill-chunk 2048 --show-logits 50 --show-ids --kv-dtype $kv --prompt "$(cat $PL/prompt-$ctx.txt)" > $R/long16k/$kv-$ctx.out 2> $R/long16k/$kv-$ctx.err
  done
  python3 -I $R/long_cmp.py $R/long16k/bf16-$ctx.err $R/long16k/fp8-$ctx.err $ctx | tee -a $R/long16k/summary.txt
done

echo "== localeval speed through intarsia serve, bf16 then fp8 then bf16... (one server per dtype), sizes 4096 16384"
for kv in bf16 fp8 bf16 fp8; do
  LD=$R/localeval/$kv-$(date +%H%M%S); mkdir -p $LD
  cooldown $O/macmon.jsonl "localeval-$kv"
  $NEW serve --model $M --backend metal --device 0 --host 127.0.0.1 --port 8791 --served-model-name intarsia-metal --max-seq 20480 --profile-every 0 --kv-dtype $kv --no-supervisor --no-ui > $LD/server.out 2> $LD/server.err &
  SPID=$!
  for i in $(seq 1 120); do curl -sf http://127.0.0.1:8791/health >/dev/null 2>&1 && break; kill -0 $SPID 2>/dev/null || { echo server died; break; }; sleep 2; done
  grep -E "kv|KV" $LD/server.err | head -5 > $LD/kv-lines.txt
  (cd /opt/code/deepeval && uv run localeval speed --url http://127.0.0.1:8791/v1 --model intarsia-metal --sizes 4096,16384 --gen 256 --reps 3 --thinking off --label "metal-kvq-$kv") > $LD/localeval.out 2> $LD/localeval.err
  echo "localeval $kv exit $?"; tail -12 $LD/localeval.out
  kill $SPID; wait $SPID 2>/dev/null
done

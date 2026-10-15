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
echo "== PROBE: fake-quant int8 (bf16 cache holding dequant(quant(x)) per block) accuracy vs fp32 goldens"
PROBE=/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/intarsia-probe
mkdir -p $R/probe
for cfg in "32,3" "16,3" "32,2" "32,1" "0,0"; do
  tag=${cfg//,/_}
  if [ $cfg = "0,0" ]; then E=NOENV=1; else E=INTARSIA_METAL_KV_FAKEQ=$cfg; fi
  env $E $PROBE accuracy --model $M --backend metal --kv-dtype bf16 --golden $G/qwen3-8b-long-fp32.safetensors --gate-class bf16 --mode all --max-seq 8192 > $R/probe/long-$tag.out 2> $R/probe/long-$tag.err; echo "probe $cfg long exit $?"
  grep -E "floor \[bf16\]" $R/probe/long-$tag.out | cut -c1-200
  env $E $PROBE accuracy --model $M --backend metal --kv-dtype bf16 --golden $G/qwen3-8b-fp32.safetensors --gate-class bf16 --max-seq 4096 > $R/probe/short-$tag.out 2> $R/probe/short-$tag.err; echo "probe $cfg short exit $?"
  grep -E "floor \[bf16\]" $R/probe/short-$tag.out | cut -c1-200
done

echo "== KV memory sizing: context that fits --mem-budget 40 (GiB) with --max-seq 4000000"
mkdir -p $R/sizing
for kv in bf16 fp8; do
  $NEW generate --model $M --backend metal --max-seq 4000000 --mem-budget 40 --kv-dtype $kv --profile-every 0 --temperature 0 --max-tokens 1 --prompt "hi" < /dev/null > $R/sizing/budget40-$kv.out 2> $R/sizing/budget40-$kv.err
  echo "$kv: $(grep -o 'context [0-9]* tokens' $R/sizing/budget40-$kv.err | head -1)"
done
echo "== warmup (page-in)"; run warm 4096 bf16 NOENV=1
: > $O/runs.tsv
for ctx in 4096 16384 32764; do
  i=0
  for kv in bf16 fp8 fp8 bf16; do
    i=$((i+1)); COOL=72 cooldown $O/macmon.jsonl "$ctx-$kv-$i" | tee -a $O/temps.log
    run ctx$ctx-$i-$kv $ctx $kv NOENV=1
  done
done
echo "== kernel-mode trace: which kernels run under fp8 (4k, 33 decode steps)"
run trace-kernel-fp8-4k 4096 fp8 INTARSIA_METAL_PROFILE=kernel:$O/trace-kernel-fp8-4k.jsonl
python3 $R/summ.py $O $O/macmon.jsonl | tee $O/summary.txt


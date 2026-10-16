#!/bin/zsh
# slot 3: engine-side decode / prefill, bf16 vs fp8 KV, ABBA per context, GPU clock logged
source /opt/code/intarsiaml-kvq/docs/results/raw/2026-10-15-metal-kv-quant/common.sh
NEW=/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/intarsia-i8
export INTARSIA_METAL_F8_U=$(pickU); echo "INTARSIA_METAL_F8_U=$INTARSIA_METAL_F8_U"
zmodload zsh/datetime
O=$R/speed2; mkdir -p $O
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
echo "== split policy sweep for the quantised kernels (INTARSIA_METAL_SPLITKV=min_keys,target_tgs), attn_bw per-call us"
for pol in 128,120 128,240 128,360 64,480 256,120; do
  for ctx in 4096 16384 32768; do
    echo "policy $pol ctx $ctx: $(INTARSIA_METAL_SPLITKV=$pol /private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/attn_bw-i8 splitkv i8 f8 $ctx 2>&1 | tail -3 | awk '{printf "%s %s us | ", $2, $3}')" | tee -a $R/speed2/split_sweep.txt
  done
done
echo "== KV memory sizing: context that fits --mem-budget 20 and 18 (GiB), --max-seq 4000000 (recipe cap 40960)"
mkdir -p $R/sizing
for mb in 20 18; do
  for kv in bf16 fp8 int8; do
    $NEW generate --model $M --backend metal --max-seq 4000000 --mem-budget $mb --kv-dtype $kv --profile-every 0 --temperature 0 --max-tokens 1 --prompt "hi" < /dev/null > $R/sizing/b$mb-$kv.out 2> $R/sizing/b$mb-$kv.err
    echo "budget $mb GiB $kv: $(grep -o 'context [0-9]* tokens' $R/sizing/b$mb-$kv.err | head -1) $(grep -ioE 'kv[^|,]*(GB|GiB|MB)[^|,]*' $R/sizing/b$mb-$kv.err | head -1)"
  done
done
echo "== warmup (page-in)"; run warm 4096 bf16 NOENV=1
: > $O/runs.tsv
for ctx in 4096 16384 32764 40000; do
  i=0
  for kv in bf16 fp8 int8 int8 fp8 bf16; do
    i=$((i+1)); COOL=${COOL:-62} COOLN=${COOLN:-18} cooldown $O/macmon.jsonl "$ctx-$kv-$i" | tee -a $O/temps.log
    run ctx$ctx-$i-$kv $ctx $kv NOENV=1
  done
done
echo "== kernel-mode trace: which kernels run under int8 (4k, 33 decode steps)"
run trace-kernel-i8-4k 4096 int8 INTARSIA_METAL_PROFILE=kernel:$O/trace-kernel-i8-4k.jsonl
python3 $R/summ.py $O $O/macmon.jsonl | tee $O/summary.txt


#!/usr/bin/env bash
# M3 CUDA window 1 (spark1, non-exclusive, vLLM up): main-code engine dumps on 35b-long + reuse diag for long_docs_4k.
set -u
: "${FLEETQ_JOB:?run under fleetq}"
D=/private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/scratchpad/m3
ssh -o BatchMode=yes spark1 "mkdir -p ~/m3/out"
scp -q $D/docs4k.safetensors $D/chat4k.safetensors spark1:m3/
ssh -n -o BatchMode=yes spark1 "FLEETQ_JOB=$FLEETQ_JOB bash -s" <<'EOS'
set -u
B=~/intarsia-decfuse/m3-main/target/acc/release/intarsia
M=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
L=~/intarsiaml-golden/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors
O=~/m3/out
avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo); echo "MemAvailable ${avail} GB"
[ "$avail" -ge 44 ] || { echo "ABORT: less than 44 GB available"; exit 3; }
$B --version
t0=$(date +%s)
INTARSIA_ACC_POS_DUMP=$O/long-c2048.pos.tsv timeout 900 $B accuracy --model $M --backend cuda --golden $L --mode prefill --max-seq 16384 --mem-budget 40 --prefill-chunk 2048 --allow-dirty --write-self-golden $O/long-c2048.dump.safetensors > $O/long-c2048.log 2>&1
echo "long c2048 exit $? $(( $(date +%s) - t0 )) s"; grep -E "^OVERALL|^prefill floor|^chat_reuse" $O/long-c2048.log
for c in 1024 2048; do
  t0=$(date +%s)
  timeout 900 $B accuracy --model $M --backend cuda --golden ~/m3/docs4k.safetensors --mode reuse --reuse-diag --max-seq 16384 --mem-budget 40 --prefill-chunk $c --allow-dirty > $O/reuse-docs4k-c$c.log 2>&1
  echo "reuse c$c exit $? $(( $(date +%s) - t0 )) s"; grep -E "^long_docs_4k|prefix-reuse gate" $O/reuse-docs4k-c$c.log
done
t0=$(date +%s)
timeout 900 $B accuracy --model $M --backend cuda --golden ~/m3/chat4k.safetensors --mode prefill --max-seq 16384 --mem-budget 40 --prefill-chunk 2048 --kv-dtype fp8_e4m3 --allow-dirty --write-self-golden $O/chat4k-fp8kv.dump.safetensors > $O/chat4k-fp8kv.log 2>&1
echo "chat fp8kv exit $? $(( $(date +%s) - t0 )) s"; grep -E "^OVERALL" $O/chat4k-fp8kv.log
EOS
mkdir -p $D/spark1
scp -q 'spark1:m3/out/*' $D/spark1/
echo done

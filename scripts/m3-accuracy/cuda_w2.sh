#!/usr/bin/env bash
# M3 CUDA window 2 (spark1, non-exclusive): tool-exact KL21 of the main build against two fresh vLLM samples of the
# 35b-long golden (same ids, salted, same server/version), to show how much the floor verdict depends on the golden sample.
set -u
: "${FLEETQ_JOB:?run under fleetq}"
D=/private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/scratchpad/m3
ssh -o BatchMode=yes spark1 "mkdir -p ~/m3/out2"
scp -q $D/rq/long-A.safetensors $D/rq/long-B.safetensors spark1:m3/
ssh -n -o BatchMode=yes spark1 "FLEETQ_JOB=$FLEETQ_JOB bash -s" <<'EOS'
set -u
B=~/intarsia-decfuse/m3-main/target/acc/release/intarsia
M=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
O=~/m3/out2
avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo); echo "MemAvailable ${avail} GB"
[ "$avail" -ge 44 ] || { echo "ABORT: less than 44 GB available"; exit 3; }
for g in long-A long-B; do
  t0=$(date +%s)
  timeout 900 $B accuracy --model $M --backend cuda --golden ~/m3/$g.safetensors --mode prefill --max-seq 16384 --mem-budget 40 --prefill-chunk 2048 --allow-dirty > $O/acc-$g.log 2>&1
  echo "$g exit $? $(( $(date +%s) - t0 )) s"; grep -E "^OVERALL|^prefill floor|^chat_reuse|^long_" $O/acc-$g.log
done
EOS
mkdir -p $D/spark1
scp -q 'spark1:m3/out2/*' $D/spark1/
echo done

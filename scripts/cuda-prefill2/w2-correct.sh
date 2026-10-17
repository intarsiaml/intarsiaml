#!/usr/bin/env bash
# Window 2 (spark1, vLLM down, INSIDE scripts/cuda-decode/window.sh): the dev tree (chunked GatedDeltaNet).
#  phase 1: tests/gdn_chunked (accuracy vs f64 reference and the rows kernel, correlated keys, carry) and the gdn_timing
#           microbench. The window ends here only if the test binary crashes (exit other than 0 / 101) or a printed chunked
#           rel_l2 is above 1e-3 (a wrong kernel); a narrow miss of the test bound still runs phase 2.
#  phase 2 (parallel): lane A = the 9 mutant test binaries (mutcheck.sh), then the admission test; lane B = dev prefill KL
#           against the vLLM golden with INTARSIA_CUDA_GDN_CHUNKED=1 (35b-short, 35b-long at chunk 2048, then the chunk-256
#           pair if time remains).
# Env: W_BUDGET_MIN (script budget, default 7). Output ~/intarsia-decfuse/out/w2
set -u
P=$HOME/intarsia-decfuse
DEV=$P/dev
BIN=$DEV/target/acc/release/intarsia
DEPS=$DEV/target/acc/release/deps
O=$P/out/w2; rm -rf $O; mkdir -p $O
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
GD=$HOME/intarsiaml-golden
DEADLINE=$(( $(date +%s) + ${W_BUDGET_MIN:-7} * 60 ))
left() { echo $(( DEADLINE - $(date +%s) )); }   # seconds
tb() { ls -t $DEPS/$1-* | grep -v '\.d$' | head -1; }
gb=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1024 / 1024 )); echo "MemAvailable ${gb} GB"; (( gb >= 90 )) || { echo "ABORT: memory"; exit 1; }
echo "dev: $($BIN --version 2>&1 | head -1)"
nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv | tee $O/clocks.txt

echo "== phase 1 $(date +%T)"
GC=$(tb gdn_chunked)
timeout 170 $GC --test-threads=1 --nocapture > $O/t-gdn_chunked.log 2>&1; rc1=$?
echo "gdn_chunked rc=$rc1: $(grep -E '^test result' $O/t-gdn_chunked.log | head -1)"; grep -E '^test ' $O/t-gdn_chunked.log | cut -c1-160
maxrel=$(grep -oE 'chunked kernel [0-9.e+-]+' $O/t-gdn_chunked.log | awk '{if ($3+0 > m) m = $3+0} END {print m+0}')
echo "largest chunked rel_l2 printed: $maxrel"
GT=$(tb gdn_timing)
timeout 80 $GT --nocapture > $O/t-gdn_timing.log 2>&1; echo "gdn_timing rc=$?"; grep -E 'us/launch' $O/t-gdn_timing.log | grep gdn_delta | cut -c1-100
if { [ $rc1 -ne 0 ] && [ $rc1 -ne 101 ]; } || awk -v m="$maxrel" 'BEGIN {exit !(m > 1e-3)}'; then
  grep -E "panicked|assert|rel_l2" $O/t-gdn_chunked.log | head -20 | cut -c1-260
  echo "== gdn_chunked crashed or is wrong: ending the window ($(left) s left) $(date +%T)"; exit 3
fi
echo "== phase 1 done, $(left) s left $(date +%T)"

echo "== phase 2 $(date +%T)"
acc() { # acc <tag> <golden> <chunk>
  local tag=$1 g=$2 chunk=$3; local t0=$(date +%s)
  INTARSIA_CUDA_GDN_CHUNKED=1 timeout 330 $BIN accuracy --model $M35 --backend cuda --golden $g --mode prefill --max-seq 16384 --mem-budget 40 --prefill-chunk $chunk --allow-dirty > $O/acc-$tag.log 2>&1
  echo "acc $tag exit $? $(( $(date +%s) - t0 )) s: $(grep -E '^prefill floor' $O/acc-$tag.log | head -1 | cut -c1-150) | selected: $(grep -c gdn_delta_rule_bf16_chunked $O/acc-$tag.log) lines name the chunked kernel" >> $O/lanes.log
}
S=$GD/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors; L=$GD/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors
(
  bash $DEV/scripts/cuda-prefill2/mutcheck.sh $O > $O/mutcheck.out 2>&1
  AD=$(tb admission)
  timeout 200 $AD every_cuda_kernel_matches_the_reference --nocapture --test-threads=1 > $O/t-admission.log 2>&1; echo "admission rc=$?: $(grep -E '^test result' $O/t-admission.log | head -1)" >> $O/mutcheck.out
  grep -E "gdn_delta_rule_bf16_chunked" $O/t-admission.log | head -8 | cut -c1-200 >> $O/mutcheck.out
) &
(
  acc short-c2048 $S 2048
  acc long-c2048 $L 2048
  (( $(left) > 120 )) && acc short-c256 $S 256
  (( $(left) > 200 )) && acc long-c256 $L 256
) &
wait
echo "== phase 2 done, $(left) s left $(date +%T)"
cat $O/mutcheck.out; cat $O/lanes.log

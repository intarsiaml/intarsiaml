#!/usr/bin/env bash
# The full gb10-d2h matrix on one GB10 box (run it under a fleetq slot). Usage: run_all.sh <repo-root> <out-dir>
# PHASES="p6 p7 p8" runs only those (default: all; p6 to p8 were added after a first pass of p1-p5).
# Phases write <out-dir>/p*.log; each phase's exit status is recorded in <out-dir>/summary.txt (a "bad" run makes it 1).
set -u
repo=$1; out=$2
PHASES=${PHASES:-"p1 p2 p3 p4 p5 p6 p7 p8"}
want() { case " $PHASES " in *" $1 "*) return 0;; esac; return 1; }
R="$repo/scripts/gb10-d2h/run.sh $repo $out"
mkdir -p "$out"; : >> "$out/summary.txt"
phase() { name=$1; shift; echo "== $name: $*" | tee -a "$out/summary.txt"; "$@" > "$out/$name.log" 2>&1; echo "   exit $? : $(grep ^TOTAL "$out/$name.log")" | tee -a "$out/summary.txt"; }
free -g | head -2 >> "$out/summary.txt"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader >> "$out/summary.txt"
uname -r >> "$out/summary.txt"; cat /sys/module/nvidia_uvm/parameters/uvm_ats_mode >> "$out/summary.txt" 2>&1
# p1: every method x destination kind, KB to 64 MiB, one thread, quiet box
want p1 && phase p1_baseline $R --iters 200
# p2: large copies (256 and 600 MiB), ours and the driver call, fresh and cold destinations
want p2 && phase p2_large $R --iters 200 --sizes 268435456,629145600 --methods ours,sync_drv --kinds fresh,cold
# p3: serving-like pressure: 4 threads, 12 GB held host memory, 256 MiB fault/drop churn, a GPU kernel stream
want p3 && phase p3_pressure $R --iters 200 --threads 4 --pressure 12 --gpuload --sizes 4096,303104,1048576,16777216 --methods ours,ours_hp,sync_rt,sync_drv --kinds fresh,cold,rand_off
# p4: page-cache churn (reading a 27B checkpoint in a loop) with p3's pressure
if want p4; then
( while :; do cat ~/models/Qwen3.8-27B-NVFP4-52d1adc5f38a/model-0000*.safetensors > /dev/null; done ) & dd_pid=$!
phase p4_pagecache $R --iters 200 --threads 4 --pressure 12 --sizes 303104,4194304,67108864 --methods ours,sync_rt,sync_drv --kinds fresh,cold
kill $dd_pid 2>/dev/null; wait $dd_pid 2>/dev/null
fi
# p5: many 296 KiB logits-row-sized copies (TensorFold's size), 8 threads, churn pool
want p5 && phase p5_many $R --iters 3000 --threads 8 --pressure 8 --sizes 303104 --methods ours,sync_drv --kinds fresh,cold
# p6: destinations backed by huge pages (MADV_HUGEPAGE) and destinations whose pages are in swap when the copy starts (MADV_PAGEOUT)
want p6 && phase p6_thp_pageout $R --iters 200 --threads 4 --pressure 12 --sizes 303104,4194304,16777216,67108864 --methods ours,sync_rt,sync_drv --kinds thp,pageout
# p7: soak, 8 threads, small and mid sizes, all destination kinds, 2000 copies per cell and thread
want p7 && phase p7_soak $R --iters 2000 --threads 8 --pressure 12 --sizes 4096,303104,1048576 --methods ours,ours_hp,sync_drv --kinds fresh,cold,rand_off,thp,pageout
# p8: harness self-test: two mutants that lose bytes (nothing copied; one page skipped) must be flagged on every run; pageout rows show how
# many destination pages were really resident when the copy began
want p8 && phase p8_selftest $R --iters 20 --sizes 303104,4194304 --methods ours,mut_noop,mut_skippage --kinds fresh,cold,pageout --no-pool
echo DONE >> "$out/summary.txt"

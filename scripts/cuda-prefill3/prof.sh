#!/usr/bin/env bash
# In-situ GPU kernel profile of a 35B prefill (~4160-token prompt, --prefill-chunk 2048, 1 generated token) on spark1,
# one nsys run per arm, kernel summary CSV per arm. vLLM stays UP (its GPU is idle; the job holds the memory it needs):
#   fleetq run --host spark1 --exclusive --mem 34 --label "prefill-3 prof" --wait-timeout 60 -- scripts/cuda-prefill3/prof.sh <name:tree[:ENV=V,..]> ...
# Output ~/intarsia-decfuse/out/p3prof-<name>/{kern_cuda_gpu_kern_sum.csv,nsys.err}
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
HOST=${HOST:-spark1}
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
L=$S/.prof_remote_$FLEETQ_JOB.sh
cat > "$L" <<'EOS'
set -u
P=$HOME/intarsia-decfuse
M=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
NSYS=$(ls /opt/nvidia/nsight-systems*/*/bin/nsys /usr/local/cuda/bin/nsys 2>/dev/null | head -1)
gb=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1024 / 1024 )); echo "MemAvailable ${gb} GB"; (( gb >= 34 )) || { echo "ABORT: memory"; exit 1; }
nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu --format=csv
for spec in "$@"; do
  name=${spec%%:*}; rest=${spec#*:}; T=${rest%%:*}; envs=""; [[ "$rest" == *:* ]] && envs=${rest#*:}; envs=${envs//,/ }
  BIN=$P/$T/target/acc/release/intarsia
  O=$P/out/p3prof-$name; rm -rf $O; mkdir -p $O
  head -c 14000 $P/$T/docs/accuracy.md > $O/prompt.txt
  echo "== $name (tree $T, env '$envs') $(date +%T)"
  env $envs $NSYS profile -t cuda --force-overwrite true -o $O/nsys $BIN generate -m $M --backend cuda --mem-budget 29 --max-seq 8192 --prefill-chunk 2048 --profile-every 0 --max-tokens 1 -p "$(cat $O/prompt.txt)" > $O/nsys.out 2> $O/nsys.err
  grep -aE "^finish=|prompt=" $O/nsys.err | head -3 | cut -c1-250
  $NSYS stats -r cuda_gpu_kern_sum -f csv --force-export=true -o $O/kern $O/nsys.nsys-rep > $O/stats.log 2>&1 || tail -3 $O/stats.log
  rm -f $O/nsys.sqlite $O/nsys.nsys-rep
  python3 - "$O/kern_cuda_gpu_kern_sum.csv" <<'PY'
import csv, re, sys
rows = list(csv.DictReader(open(sys.argv[1])))
tot = sum(float(r["Total Time (ns)"]) for r in rows)
groups = {}
for r in rows:
    n = r["Name"]; t = float(r["Total Time (ns)"]); c = int(r["Instances"])
    m = re.search(r"wq_gemm_kernel<\(int\)(\d)", n)
    if m: key = "wq_gemm FMT=" + ("FP8" if m.group(1) == "1" else "NVFP4")
    elif "fp8_f8_gemm_kernel" in n: key = "fp8_f8_gemm (e4m3 mma)"
    elif "fp8_actq" in n: key = "fp8_actq"
    elif "gdn_delta_lanes" in n: key = "gdn lanes"
    elif "gdn_delta_rows" in n: key = "gdn rows"
    elif "moe" in n: key = "moe (all)"
    else: key = re.sub(r"\(.*", "", n.replace("void ", ""))[:40]
    g = groups.setdefault(key, [0.0, 0]); g[0] += t; g[1] += c
print(f"kernel total {tot/1e6:.1f} ms")
for k, (t, c) in sorted(groups.items(), key=lambda x: -x[1][0])[:16]:
    print(f"{k:<44} {t/1e6:8.1f} ms {100*t/tot:5.1f} % {c:6d} calls")
PY
done
EOS
scp -q "$L" "$HOST:/tmp/p3prof_$FLEETQ_JOB.sh"
rm -f "$L"
ssh -n -o BatchMode=yes "$HOST" "export PATH=\$HOME/.cargo/bin:/usr/local/cuda/bin:\$PATH; export FLEETQ_JOB=$FLEETQ_JOB; bash /tmp/p3prof_$FLEETQ_JOB.sh $*; rc=\$?; rm -f /tmp/p3prof_$FLEETQ_JOB.sh; exit \$rc"

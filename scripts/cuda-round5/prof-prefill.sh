#!/usr/bin/env bash
# Prefill kernel profile of the 35B per arm and prompt length (spark1 side; ONE EXCLUSIVE fleetq slot with vLLM up and its
# GPU idle: shares, not absolute speed). Pattern of scripts/cuda-prefill3/prof.sh: nsys over `generate --max-tokens 1`,
# --prefill-chunk 2048, kernel summary grouped by kernel name, ranked by share.
#   via scripts/cuda-round5/dev.sh STEPS=pprof (fleetq run --host spark1 --exclusive --mem 34 ...)
# ARMS: ";"-separated "name:ENV=V ENV=V" (ARGS=--a=b+--c=d: extra generate arguments), default
# "bf16:INTARSIA_X=0;f8:ARGS=--kv-dtype=fp8_e4m3"; CHARS: prompt characters of tools/prompts/corpus_docs.txt (default
# "14000 54000": ~4k and ~16k tokens). Output ~/intarsia-decfuse/out/r5pprof-<arm>-<chars>/
set -u
P=$HOME/intarsia-decfuse
T=${TREE:-r5-f8}
BIN=$P/$T/target/acc/release/intarsia
M=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
NSYS=$(ls /opt/nvidia/nsight-systems*/*/bin/nsys /usr/local/cuda/bin/nsys 2>/dev/null | head -1)
gb=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1024 / 1024 )); echo "MemAvailable ${gb} GB"; (( gb >= 34 )) || { echo "ABORT: memory"; exit 1; }
nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu --format=csv
IFS=';' read -r -a arms <<< "${ARMS:-bf16:INTARSIA_X=0;f8:ARGS=--kv-dtype=fp8_e4m3}"
for chars in ${CHARS:-14000 54000}; do
  for a in "${arms[@]}"; do
    name=${a%%:*}; envs=${a#*:}
    gargs=""; for e in $envs; do [[ $e == ARGS=* ]] && gargs=${e#ARGS=}; done; gargs=${gargs//+/ }
    envs=$(for e in $envs; do [[ $e == ARGS=* ]] || printf '%s ' "$e"; done)
    O=$P/out/r5pprof-$name-$chars; rm -rf $O; mkdir -p $O
    head -c $chars $P/$T/tools/prompts/corpus_docs.txt > $O/prompt.txt
    echo "== $name, $chars chars, env '$envs' args '$gargs' $(date +%T)"
    # plain run first (timing without the profiler), then the nsys run
    env $envs $BIN generate -m $M --backend cuda --mem-budget 29 --max-seq 20480 --prefill-chunk 2048 --profile-every 0 --max-tokens 1 $gargs -p "$(cat $O/prompt.txt)" > $O/plain.out 2> $O/plain.err
    grep -aE "finish=" $O/plain.err | head -1 | cut -c1-200
    env $envs $NSYS profile -t cuda --force-overwrite true -o $O/nsys $BIN generate -m $M --backend cuda --mem-budget 29 --max-seq 20480 --prefill-chunk 2048 --profile-every 0 --max-tokens 1 $gargs -p "$(cat $O/prompt.txt)" > $O/nsys.out 2> $O/nsys.err
    $NSYS stats -r cuda_gpu_kern_sum -f csv --force-export=true -o $O/kern $O/nsys.nsys-rep > $O/stats.log 2>&1 || tail -3 $O/stats.log
    rm -f $O/nsys.sqlite $O/nsys.nsys-rep
    python3 - "$O/kern_cuda_gpu_kern_sum.csv" <<'PY' | tee $O/rank.txt
import csv, re, sys
rows = list(csv.DictReader(open(sys.argv[1])))
tot = sum(float(r["Total Time (ns)"]) for r in rows)
groups = {}
for r in rows:
    n = r["Name"].replace("void ", "")
    key = re.sub(r"^(\(anonymous namespace\)|<unnamed>)::", "", n)
    key = re.sub(r"\(.*", "", key)
    key = re.sub(r"<\(int\)", "<", key).replace("(bool)", "")[:70]
    g = groups.setdefault(key, [0.0, 0]); g[0] += float(r["Total Time (ns)"]); g[1] += int(r["Instances"])
print(f"kernel total {tot/1e6:.1f} ms")
for k, (t, c) in sorted(groups.items(), key=lambda x: -x[1][0])[:18]:
    print(f"{k:<72} {t/1e6:8.1f} ms {100*t/tot:5.1f} % {c:6d} calls")
PY
  done
done

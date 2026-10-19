#!/usr/bin/env bash
# Round-6 prefill kernel profile of the 35B per arm and prompt length (spark1 side; ONE EXCLUSIVE fleetq slot with vLLM up
# and its GPU idle: shares and gaps, not absolute speed). scripts/cuda-round5/prof-prefill.sh plus a launch-gap analysis:
# nsys over `generate --max-tokens 1`, kernel summary grouped by kernel name and ranked by share, then from the GPU trace
# the prefill span (first to last kernel of the prompt pass), the busy time (union of kernel intervals) and the idle gaps.
#   fleetq run --host spark1 --exclusive --mem 34 --label "..." -- scripts/cuda-prefill4/remote.sh r6-base scripts/cuda-round6/prof-prefill.sh
# ARMS: ";"-separated "name:ENV=V ENV=V" (ARGS=--a=b+--c=d: extra generate arguments), default "base:INTARSIA_X=0";
# CHARS: prompt characters of tools/prompts/corpus_docs.txt (default "3400 14000 54000": ~1k, ~4k and ~16k tokens);
# CHUNK (default 2048). Output ~/intarsia-decfuse/out/${OUTP:-r6pprof}-<arm>-<chars>/
set -u
P=$HOME/intarsia-decfuse
T=${TREE:-r6-base}
BIN=$P/$T/target/acc/release/intarsia
M=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
NSYS=$(ls /opt/nvidia/nsight-systems*/*/bin/nsys /usr/local/cuda/bin/nsys 2>/dev/null | head -1)
gb=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1024 / 1024 )); echo "MemAvailable ${gb} GB"; (( gb >= 34 )) || { echo "ABORT: memory"; exit 1; }
nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu --format=csv
CH=${CHUNK:-2048}
IFS=';' read -r -a arms <<< "${ARMS:-base:INTARSIA_X=0}"
for chars in ${CHARS:-3400 14000 54000}; do
  for a in "${arms[@]}"; do
    name=${a%%:*}; envs=${a#*:}
    gargs=""; for e in $envs; do [[ $e == ARGS=* ]] && gargs=${e#ARGS=}; done; gargs=${gargs//+/ }
    envs=$(for e in $envs; do [[ $e == ARGS=* ]] || printf '%s ' "$e"; done)
    O=$P/out/${OUTP:-r6pprof}-$name-$chars; rm -rf $O; mkdir -p $O
    head -c $chars $P/$T/tools/prompts/corpus_docs.txt > $O/prompt.txt
    echo "== $name, $chars chars, chunk $CH, env '$envs' args '$gargs' $(date +%T)"
    env $envs $BIN generate -m $M --backend cuda --mem-budget 29 --max-seq 20480 --prefill-chunk $CH --profile-every 0 --max-tokens 1 $gargs -p "$(cat $O/prompt.txt)" > $O/plain.out 2> $O/plain.err
    grep -aE "finish=" $O/plain.err | head -1 | cut -c1-200
    env $envs $NSYS profile -t cuda --force-overwrite true -o $O/nsys $BIN generate -m $M --backend cuda --mem-budget 29 --max-seq 20480 --prefill-chunk $CH --profile-every 0 --max-tokens 1 $gargs -p "$(cat $O/prompt.txt)" > $O/nsys.out 2> $O/nsys.err
    grep -aE "finish=" $O/nsys.err | head -1 | cut -c1-200
    $NSYS stats -r cuda_gpu_kern_sum,cuda_gpu_trace -f csv --force-export=true -o $O/kern $O/nsys.nsys-rep > $O/stats.log 2>&1 || tail -3 $O/stats.log
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
for k, (t, c) in sorted(groups.items(), key=lambda x: -x[1][0])[:20]:
    print(f"{k:<72} {t/1e6:8.1f} ms {100*t/tot:5.1f} % {c:6d} calls")
PY
    # launch gaps: the prompt pass is everything up to the last moe_w4a16 / big-chunk kernel; we take the window from the
    # first kernel to the start of the last decode-sized burst by cutting at the largest host gap after the last flash/moe
    python3 - "$O/kern_cuda_gpu_trace.csv" <<'PY' | tee $O/gaps.txt
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r.get("Name") and not r["Name"].startswith("[CUDA")]
ev = sorted((int(r["Start (ns)"]), int(r["Start (ns)"]) + int(r["Duration (ns)"]), r["Name"]) for r in rows)
if not ev:
    print("no kernels"); sys.exit()
# prompt pass: from the first kernel to the end of the last kernel whose name marks a prefill-only path
pre = [i for i, e in enumerate(ev) if any(s in e[2] for s in ("moe_w4a16", "flash_kernel", "gemm_reg", "gdn_delta_split", "moe_experts_mma", "wq_gemm"))]
last = pre[-1] if pre else len(ev) - 1
win = ev[: last + 1]
t0, t1 = win[0][0], max(e[1] for e in win)
busy, cur_s, cur_e = 0, win[0][0], win[0][1]
gaps = []
for s, e, n in win[1:]:
    if s > cur_e:
        busy += cur_e - cur_s; gaps.append((s - cur_e, n)); cur_s, cur_e = s, e
    else:
        cur_e = max(cur_e, e)
busy += cur_e - cur_s
ksum = sum(e - s for s, e, _ in win)
print(f"prompt window: {len(win)} kernels, span {(t1-t0)/1e6:.2f} ms, busy {busy/1e6:.2f} ms, idle {(t1-t0-busy)/1e6:.2f} ms ({100*(t1-t0-busy)/(t1-t0):.1f} %), kernel sum {ksum/1e6:.2f} ms")
small = sum(g for g, _ in gaps if g < 20000); big = [(g, n) for g, n in gaps if g >= 20000]
print(f"gaps: {len(gaps)}; < 20 us total {small/1e6:.2f} ms (mean {small/max(1,len([g for g,_ in gaps if g<20000]))/1e3:.1f} us); >= 20 us: {len(big)} totalling {sum(g for g,_ in big)/1e6:.2f} ms")
for g, n in sorted(big, reverse=True)[:8]:
    print(f"  gap {g/1e3:9.1f} us before {n[:90]}")
PY
  done
done

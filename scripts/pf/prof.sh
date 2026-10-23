#!/usr/bin/env bash
# Lane pf, spark side: nsys profile of a 35B prefill (generate --max-tokens 1) per arm and prompt size, ranked by scripts/s1/rank.py.
# Run inside a fleetq slot (FLEETQ_JOB set). Usage: prof.sh <outname> <chars,...> <arm> ...   arm = name:tree:chunk[:ENV=V,ENV=V]
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
P=$HOME/intarsia-decfuse; OUT=$P/out/$1; CH=$2; shift 2
export PATH=/usr/local/cuda/bin:$PATH
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
CORPUS=$P/pf-base/tools/prompts/corpus_docs.txt
mkdir -p $OUT
for spec in "$@"; do
  IFS=: read -r name tree chunk envs <<< "$spec"; envs=${envs//,/ }
  for chars in ${CH//,/ }; do
    d=$OUT/$name-$chars; rm -rf $d; mkdir -p $d; head -c $chars $CORPUS > $d/prompt.txt
    echo "== $name chars=$chars chunk=$chunk env='$envs' $(date +%T)"
    env $envs nsys profile -t cuda --force-overwrite true -o $d/nsys $P/$tree/target/acc/release/intarsia generate -m $M35 --backend cuda \
      --mem-budget 29 --max-seq 20480 --prefill-chunk $chunk --profile-every 0 --max-tokens 1 -p "$(cat $d/prompt.txt)" < /dev/null > $d/nsys.out 2> $d/nsys.err
    grep -aE 'finish=' $d/nsys.err | head -1 | cut -c1-200
    nsys stats -r cuda_gpu_kern_sum,cuda_gpu_trace -f csv --force-export=true -o $d/kern $d/nsys.nsys-rep < /dev/null > $d/stats.log 2>&1 || tail -3 $d/stats.log
    rm -f $d/nsys.sqlite
    python3 $P/pf-scripts/s1/rank.py $d/kern_cuda_gpu_kern_sum.csv $d/kern_cuda_gpu_trace.csv > $d/rank.txt 2>&1
    head -30 $d/rank.txt
  done
done
echo "== done $(date +%T)"

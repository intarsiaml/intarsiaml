#!/usr/bin/env bash
# Lane sp2: in-engine schedule sweep of the routed-expert kernels. One nsys prefill (4k prompt, chunk 2048) per setting; prints
# the per-mode (gate/up = MODE 0, down = MODE 1) average microseconds of moe_w4a16_kernel.
#   fleetq run --host spark1 --exclusive --mem 40 --label "..." -- scripts/sp2/sweep-job.sh "0,0,0 1,1,0 ..."
# Setting = cfg0,cfg1,abl  (INTARSIA_W4_CFG0, INTARSIA_W4_CFG1, INTARSIA_W4_ABL).  TREE (default sp2-dev), CHARS (default 14000).
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
HOST=${HOST:-spark1}
ssh -n -o BatchMode=yes "$HOST" "export FLEETQ_JOB=$FLEETQ_JOB PATH=/usr/local/cuda/bin:\$PATH; set -u
P=\$HOME/intarsia-decfuse; O=\$P/out/sp2S; mkdir -p \$O
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
NSYS=\$(ls /opt/nvidia/nsight-systems*/*/bin/nsys /usr/local/cuda/bin/nsys 2>/dev/null | head -1)
head -c ${CHARS:-14000} \$P/sp2-dev/tools/prompts/corpus_docs.txt > \$O/prompt.txt
for st in $1; do
  IFS=, read c0 c1 ab <<< \"\$st\"
  d=\$O/\${st//,/_}-${CHARS:-14000}; rm -rf \$d; mkdir -p \$d
  env INTARSIA_W4_CFG0=\$c0 INTARSIA_W4_CFG1=\$c1 INTARSIA_W4_ABL=\$ab \$NSYS profile -t cuda --force-overwrite true -o \$d/nsys \$P/${TREE:-sp2-dev}/target/acc/release/intarsia generate -m \$M35 --backend cuda --mem-budget 29 --max-seq 20480 --prefill-chunk 2048 --profile-every 0 --max-tokens 1 -p \"\$(cat \$O/prompt.txt)\" < /dev/null > \$d/run.out 2> \$d/run.err
  \$NSYS stats -r cuda_gpu_kern_sum -f csv --force-export=true -o \$d/kern \$d/nsys.nsys-rep < /dev/null > \$d/stats.log 2>&1; rm -f \$d/nsys.sqlite
  echo \"== cfg0=\$c0 cfg1=\$c1 abl=\$ab: \$(grep -aE 'finish=' \$d/run.err | head -1 | cut -c1-120)\"
  python3 - \$d/kern_cuda_gpu_kern_sum.csv <<'PY'
import csv, re, sys
tot = 0
for r in csv.DictReader(open(sys.argv[1])):
    tot += float(r['Total Time (ns)'])
    m = re.search(r'moe_w4a16_kernel<\(int\)(\d+), \(int\)(\d+), \(int\)(\d+), \(int\)(\d+), \(int\)(\d+), \(int\)(\d+), \(int\)(\d+)(, \(int\)(\d+))?', r['Name'])
    if m and m.group(2) == '4':
        print('  mode %s R=%s JW=%s MINB=%s LA=%s LR=%s abl=%s: %s calls, avg %.1f us, total %.1f ms' % (m.group(4), m.group(1), m.group(3), m.group(5), m.group(6), m.group(7), m.group(9) or 0, r['Instances'], float(r['Avg (ns)'])/1e3, float(r['Total Time (ns)'])/1e6))
print('  kernel total %.1f ms' % (tot/1e6))
PY
done"

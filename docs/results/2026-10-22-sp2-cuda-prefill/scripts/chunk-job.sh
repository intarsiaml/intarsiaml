#!/usr/bin/env bash
# Lane sp2: is --prefill-chunk bitwise-neutral on the 35B (greedy ids), and what does it do to prefill speed? Base tree (origin/main),
# exclusive slot:  fleetq run --host spark1 --exclusive --mem 40 --label "..." -- docs/results/2026-10-22-sp2-cuda-prefill/scripts/chunk-job.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
ssh -n -o BatchMode=yes "${HOST:-spark1}" "export FLEETQ_JOB=$FLEETQ_JOB; set -u
P=\$HOME/intarsia-decfuse; O=\$P/out/sp2C; rm -rf \$O; mkdir -p \$O; B=\$P/sp2-base/target/acc/release/intarsia
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
C=\$P/sp2-dev/tools/prompts/corpus_docs.txt
for ch in 2048 4096 8192 2048 4096 8192; do
  for n in 5000:256 14000:1 55000:1; do chars=\${n%%:*}; tok=\${n##*:}
    f=\$O/c\$ch-\$chars-\$RANDOM
    \$B generate -m \$M35 --backend cuda --mem-budget 40 --max-seq 20480 --prefill-chunk \$ch --profile-every 0 --ignore-eos --show-ids --max-tokens \$tok -p \"\$(head -c \$chars \$C)\" < /dev/null > \$f.out 2> \$f.err
    echo \"chunk \$ch chars \$chars: \$(grep -aE 'finish=' \$f.err | head -1 | cut -c1-110) ids_md5=\$(grep -a '^ids:' \$f.err | md5sum | cut -c1-8)\"
  done
done"

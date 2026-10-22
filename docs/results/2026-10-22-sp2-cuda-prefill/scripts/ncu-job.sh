#!/usr/bin/env bash
# Lane sp2: ncu of the routed-expert w4a16 kernels (MODE 0 gate/up, MODE 1 down, first layer, first 2048 chunk) in an exclusive slot.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
HOST=${HOST:-spark1}; TREE=${TREE:-sp2-base}; TAG=${TAG:-base}
ssh -n -o BatchMode=yes "$HOST" "export FLEETQ_JOB=$FLEETQ_JOB PATH=/usr/local/cuda/bin:\$PATH; set -u
P=\$HOME/intarsia-decfuse; O=\$P/out/sp2N; mkdir -p \$O
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
head -c ${CHARS:-14000} \$P/sp2-dev/tools/prompts/corpus_docs.txt > \$O/prompt.txt
ncu --target-processes all --kernel-name regex:moe_w4a16_kernel --launch-skip ${SKIP:-0} --launch-count ${COUNT:-2} --set full -f -o \$O/ncu-$TAG \
  \$P/$TREE/target/acc/release/intarsia generate -m \$M35 --backend cuda --mem-budget 29 --max-seq 20480 --prefill-chunk 2048 --profile-every 0 --max-tokens 1 -p \"\$(cat \$O/prompt.txt)\" < /dev/null > \$O/ncu-$TAG.out 2> \$O/ncu-$TAG.err
echo ncu exit \$?
ncu -i \$O/ncu-$TAG.ncu-rep --page details > \$O/ncu-$TAG.txt 2>&1; wc -l \$O/ncu-$TAG.txt"

#!/usr/bin/env bash
# Lane FX5c, spark1: the 35B reuse gate (what qualify's acc-long runs) at chunks 1024 and 2048 with the
# oracle-relative continue rule, then mutation and fallback runs on long_docs_4k. The tree must already be built there
# (`fleetq run --host spark1 --mem 4 -- scripts/cuda-decode/build.sh fx5c`).
#   fleetq run --host spark1 --mem 45 --expect 60 --label "fx5c: qualify acc-long 35B CUDA, oracle-relative continue (non-exclusive, numerics only)" -- scripts/fx5c/validate-cuda.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$REPO"
HOST=${HOST:-spark1}
RAW=${RAW:-$REPO/docs/results/2026-10-22-fx5c-oracle-gate/raw/cuda}
mkdir -p "$RAW"
R='~/intarsia-decfuse'
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
G='$HOME/intarsiaml-golden/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors'
BIN="$R/fx5c/target/acc/release/intarsia"
OUT="$R/out/fx5c/val"
if [[ "${BUILD:-0}" == 1 ]]; then   # a box without the tree: seed the dependency cache from another tree there, build in this slot (cpu)
  rsh "mkdir -p $R/fx5c; test -d $R/fx5c/target || cp -a ${SEED_TARGET:-$R/fx3-main/target} $R/fx5c/target"
  HOST=$HOST "$REPO/scripts/cuda-decode/build.sh" fx5c 2>&1 | tail -4
fi
rsh "mkdir -p ~/intarsiaml-golden"
rsync -a /opt/code/intarsiaml/golden/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors "$HOST:intarsiaml-golden/"
rsh "rm -rf $OUT; mkdir -p $OUT ~/intarsiaml-golden/oracle"
rsync -a "$REPO"/golden/oracle/*.continue-oracle.safetensors "$HOST:intarsiaml-golden/oracle/"
rsh "ls -la ~/intarsiaml-golden/oracle/ | grep continue"
# the gate qualify's acc-long runs (reuse at chunk 1024 via acc-long-reuse, and inside `--mode all` at the CUDA plan chunk 2048):
# `accuracy --mode reuse --reuse-require-all` on the whole long golden at both chunks (qualify itself costs ~45 min of prefill/decode KL gates this change does not touch)
B="$BIN accuracy --model $M35 --backend cuda --golden $G --mode reuse --max-seq 16384 --mem-budget 40 --allow-dirty"
for c in 1024 2048; do
  rsh "cd $R/fx5c && $B --prefill-chunk $c --reuse-require-all > $OUT/reuse-c$c.log 2>&1; echo exit \$? >> $OUT/reuse-c$c.log"
  rsync -a "$HOST:intarsia-decfuse/out/fx5c/val/" "$RAW/"
  grep -a -e 'continue judged' -e '^exit' -e 'prefix-reuse' -e '^long_\|^chat_' "$RAW/reuse-c$c.log" | cut -c1-300 | tail -24
done
# mutation (a deliberate error on the warm rows only) and the absolute fallback, long_docs_4k, chunk 1024
A="$B --prefill-chunk 1024 --reuse-prompt long_docs_4k"
for m in "x1.2 scale:1.2" "x1.1 scale:1.1" "x1.01 scale:1.01"; do set -- $m
  rsh "cd $R/fx5c && INTARSIA_REUSE_TEST_BUG=$2 $A > $OUT/mut-$1.log 2>&1; echo exit \$? >> $OUT/mut-$1.log"
done
rsh "cd $R/fx5c && $A --no-reuse-oracle > $OUT/noorc.log 2>&1; echo exit \$? >> $OUT/noorc.log"
rsync -a "$HOST:intarsia-decfuse/out/fx5c/val/" "$RAW/"
grep -a -e 'continue judged' -e '^exit' -e 'prefix-reuse' "$RAW"/mut-*.log "$RAW"/noorc.log | cut -c1-300

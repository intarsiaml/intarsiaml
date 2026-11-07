#!/usr/bin/env bash
# Lane mt2 CUDA no-op gates ON spark1 (vLLM up; no model is loaded), base = origin/main binary, dev = the lane binary (scripts/mt2/build.sh):
# `plan show --json` 35B + 8B at tp=1 and 35B at tp=2 `--spec off` (both ranks on device 0, host comm) must be byte-identical base vs dev;
# then `kernels test --quick` on dev. The 35B tp=2 `--spec mtp` plan is recorded (it differs by design: the whole head on rank 0).
#   fleetq run --host spark1 --mem 20 --load cpu --expect 20 --label "mt2 cuda no-op" -- ssh spark1 'bash ~/intarsia-mt2/cuda-gates.sh'
set -uo pipefail
P=$HOME/intarsia-mt2; O=$P/out/gates; mkdir -p $O
H=/home/nvidia/.cache/huggingface/hub
M35=$(ls -d $H/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/* | head -1)
M8=$(ls -d $H/models--Qwen--Qwen3-8B/snapshots/* | head -1)
echo "free GB: $(free -g | awk '/Mem:/ {print $7}')"
for t in base dev; do echo "$t: $(strings $P/bin/$t-intarsia | grep -m1 -o 'head=[0-9a-f]* code-tree=[0-9a-f]* dirty=[a-z]*')"; done
TP2="--tp 2 --comm host --devices 0,0"
plan() { # <tag> <model> <budget> <flags...>
  local tag=$1 m=$2 mb=$3; shift 3
  for t in base dev; do
    env INTARSIA_TP_SAME_DEVICE=1 $P/bin/$t-intarsia plan show --backend cuda --model $m --max-seq 8192 --mem-budget $mb --json "$@" < /dev/null > $O/plan-$tag-$t.json 2> $O/plan-$tag-$t.err
  done
  cmp -s $O/plan-$tag-base.json $O/plan-$tag-dev.json && [ -s $O/plan-$tag-dev.json ] && echo "plan $tag: byte-identical ($(wc -c < $O/plan-$tag-dev.json) B)" || echo "plan $tag: DIFFER or empty (base $(wc -c < $O/plan-$tag-base.json) B, dev $(wc -c < $O/plan-$tag-dev.json) B)"
}
plan 35b $M35 30
plan 8b $M8 25
plan 35b-mtp $M35 30 --spec mtp --spec-k 2
plan 35b-tp2-off $M35 18 $TP2 --spec off
plan 35b-tp2-default $M35 18 $TP2
INTARSIA_TP_SAME_DEVICE=1 $P/bin/dev-intarsia plan show --backend cuda --model $M35 --max-seq 8192 --mem-budget 18 --json $TP2 --spec mtp --spec-k 2 < /dev/null > $O/plan-35b-tp2-mtp-dev.json 2> $O/plan-35b-tp2-mtp-dev.err
echo "plan 35b-tp2-mtp dev (new, differs by design): exit $? $(wc -c < $O/plan-35b-tp2-mtp-dev.json) B; $(grep -aiE 'mtp|error' $O/plan-35b-tp2-mtp-dev.err | head -3 | tr '\n' '|')"
echo "== kernels test --quick (dev) $(date +%T)"
$P/bin/dev-intarsia kernels test --quick --backend cuda < /dev/null > $O/kernels-quick.log 2>&1
echo "exit $? FAIL lines: $(grep -c FAIL $O/kernels-quick.log)"; tail -3 $O/kernels-quick.log
echo "== done $(date +%T)"

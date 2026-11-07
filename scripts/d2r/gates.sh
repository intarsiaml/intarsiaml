#!/usr/bin/env bash
# Lane d2r CUDA no-op gates ON spark1 (vLLM up), base = origin/main binary, dev = the lane binary (scripts/d2r/build.sh): `plan show --json` of the
# 35B, 8B and 27B at tp=1 and of the 27B and 35B at tp=2 `--spec off` (both ranks on device 0, host comm) must be byte-identical base vs dev;
# then `kernels test --quick` on dev (0 FAIL lines).
#   fleetq run --host spark1 --mem 60 --load cpu --expect 25 --label "d2r cuda no-op" -- ssh spark1 'bash ~/intarsia-d2r/dev/scripts/d2r/gates.sh'
set -uo pipefail
P=$HOME/intarsia-d2r; O=$P/out/gates; mkdir -p $O
H=/home/nvidia/.cache/huggingface/hub
M35=$(ls -d $H/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/* | head -1)
M8=$(ls -d $H/models--Qwen--Qwen3-8B/snapshots/* | head -1)
M27=$P/models/Qwen3.8-27B-Quark-AWQ-MXFP4
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
plan 27b $M27 30
plan 27b-tp2-off $M27 14 $TP2 --spec off
plan 35b-tp2-off $M35 14 $TP2 --spec off
echo "== kernels test --quick (dev) $(date +%T)"
$P/bin/dev-intarsia kernels test --quick --backend cuda < /dev/null > $O/kernels-quick.log 2>&1
echo "exit $? FAIL lines: $(grep -c FAIL $O/kernels-quick.log)"; tail -3 $O/kernels-quick.log
echo "== done $(date +%T)"

#!/usr/bin/env bash
# cvatflip gate: `plan show --json` of main's binary and the lane's, INTARSIA_CUDA_VERIFY_ATTN_ROWS unset and =0 (and with INTARSIA_SPEC_COPY=20), must be byte-identical.
#   fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 15 --label "cvatflip plan show" -- ssh spark1 'bash -s' < scripts/cvatflip/plan.sh
set -uo pipefail
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
R=$HOME/intarsiaml-cvatflip; O=${O:-$R/out/plan}; mkdir -p $O
H=/home/nvidia/.cache/huggingface/hub
M35=$(ls -d $H/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/* | head -1)
show() { # <name> <binary> [ENV=val ...]
  n=$1; b=$2; shift 2
  env ${@:-X=1} $b plan show --model $M35 --backend cuda --spec mtp --spec-k 2 --mem-budget 40 --max-seq 8192 --json < /dev/null > $O/$n.json 2> $O/$n.err
  echo "$n: exit $? $(wc -c < $O/$n.json) bytes sha $(sha256sum < $O/$n.json | cut -c1-16)"
}
show main $R/bin/intarsia-main
show main-copy $R/bin/intarsia-main INTARSIA_SPEC_COPY=20
show new $R/bin/intarsia
show new-zero $R/bin/intarsia INTARSIA_CUDA_VERIFY_ATTN_ROWS=0
show new-copy $R/bin/intarsia INTARSIA_SPEC_COPY=20
show new-zero-copy $R/bin/intarsia INTARSIA_CUDA_VERIFY_ATTN_ROWS=0 INTARSIA_SPEC_COPY=20
for x in new new-zero; do cmp $O/main.json $O/$x.json && echo "PASS plan show $x == main (byte-identical)" || echo "FAIL plan show $x differs from main"; done
for x in new-copy new-zero-copy; do cmp $O/main-copy.json $O/$x.json && echo "PASS plan show $x == main with INTARSIA_SPEC_COPY=20 (byte-identical)" || echo "FAIL plan show $x differs from main-copy"; done

#!/usr/bin/env bash
# Lane spark35 early hard gate: the real bf16 state layout must give the SAME greedy ids as its emulation (same values, half the bytes), with the
# default MTP speculation and with --spec off, and the env-unset build must give the same ids as origin/main. 6 prompts (1k/4k/16k x 2), 256 tokens.
# Arms: main (origin/main binary), base (dev, no env), cemu, creal, cemu-nospec, creal-nospec. Exit 1 on any mismatch.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
trap 'stop_server' EXIT INT TERM
G="$H/../fp8kv/greedy.py"; D=$RES/idgate; mkdir -p "$D"
arm() { # arm <name> <bin> <env> <extra serve args>
  start_server "id-$1" "$2" "$3" ${4:-} || return 1
  python3 "$G" run --url "http://$HN:$PORT/v1" --gen 256 --out "$D/$1.json" > "$D/$1.log" 2>&1; tail -1 "$D/$1.log"
  stop_server
}
MAXSEQ=20480 MEMB=${MEMB:-45}
export MAXSEQ MEMB
arm main main "" ""; arm base dev "" ""
arm cemu dev "INTARSIA_CUDA_GDN_STATE_BF16_EMU=1" ""; arm creal dev "INTARSIA_GDN_STATE=bf16" ""
arm cemu-nospec dev "INTARSIA_CUDA_GDN_STATE_BF16_EMU=1" "--spec off"; arm creal-nospec dev "INTARSIA_GDN_STATE=bf16" "--spec off"
rc=0
cmp2() { if python3 "$G" compare "$D/$1.json" "$D/$2.json" > "$D/cmp-$1-vs-$2.txt" 2>&1; then echo "PASS ids $1 == $2"; else echo "FAIL ids $1 != $2 ($(grep -c differ "$D/cmp-$1-vs-$2.txt") prompts differ)"; rc=1; fi; }
cmp2 main base; cmp2 cemu creal; cmp2 cemu-nospec creal-nospec
exit $rc

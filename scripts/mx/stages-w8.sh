#!/usr/bin/env bash
# Lane mx window 8 (card 0): the 16-row peel. w7 showed peel 8 bitwise for 1..8-row tails but not for 9..16-row tails
# (split-KV attention at <= 16 rows on the 256 schedule, flash above); this window repeats w7's checks with
# INTARSIA_PREFILL_PEEL=16 on the PR-head build (e5f849b; its qwen3_5_vl.toml already resolves to 4096 + 16): tails dumps (vs w7's main chunk-256 dump), the 27B golden gate,
# 256 greedy ids through the engine at lengths whose last multi-row pass is 10..16 rows inside a 4096 pass (525, 529, 8459)
# or after it (4109), prefill speed.
set -uo pipefail
O=$1; S="$(dirname "${BASH_SOURCE[0]}")"; source "$S/lib.sh"
for t in base dev; do [[ $(T $t)/mx-intarsia -nt $(T $t)/Cargo.toml ]] || { log "NO FRESH BINARY in $(T $t); abort"; exit 3; }; done
log "w8 start; budget $(left) s; base $(cat $(T base)/.stamp) dev $(cat $(T dev)/.stamp)"
need 60 "plan" && E="INTARSIA_PREFILL_PEEL=16" plan dev dev-27 "$M27" --prefill-chunk 4096
need 200 "acc" && GT=g27-tails2 E="INTARSIA_PREFILL_PEEL=16" acc dev dev-peel16 4096
need 100 "accg" && E="INTARSIA_PREFILL_PEEL=16" accg dev dev16 --prefill-chunk 4096
for k in 525 529 8459 4109; do
  need 80 "ids" && E="INTARSIA_PREFILL_PEEL=16" ids dev dev16-$k "$M27" $k --prefill-chunk 4096
  need 80 "ids" && ids base base-$k "$M27" $k
  same ids-dev16-$k.ids ids-base-$k.ids "27B $k ids dev (4096 + peel 16) vs main (256)"
done
for n in 1024 4096 16384; do
  need 60 "pf" && E="INTARSIA_PREFILL_PEEL=16" pf dev peel16-$n "$M27" $n --prefill-chunk 4096
done
# LDS / VALU counters on the v2 GEMM at M = 4096 (w7 passed the counters without the M argument)
need 120 "pmc" && pmcbench dev lds 4096 SQ_LDS_BANK_CONFLICT SQ_LDS_IDX_ACTIVE
need 120 "pmc" && pmcbench dev valu 4096 SQ_INSTS_VALU SQ_INSTS_LDS SQ_WAIT_INST_LDS
log "w8 done; $(left) s left"

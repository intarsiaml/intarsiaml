#!/usr/bin/env bash
# Lane mx window 7 (card 0): the 27B HIP recipe default chunk 4096 + 8-row tail peel (qwen3_5.toml) vs main's chunk 256.
# Tails self-golden dumps (34 prompts: P1's golden + synthetic tails of 0..255 rows mod 256 and lengths above 4096) main
# chunk 256 vs dev default vs dev with the peel off (INTARSIA_PREFILL_PEEL=0, negative control), compared offline with
# scripts/ck/chunkcmp.py; the 27B golden gate (prefill + teacher-forced decode) dev default; 256 greedy ids through the
# engine at prompt lengths whose last multi-row pass is 3..8 rows (1029, 4100, 4360, 8201) and 16k; plan show; prefill
# speed main vs dev default at 1k / 4k / 16k (+ fold opt-in); mxbench with the nodq arm and LDS / VALU counters.
set -uo pipefail
O=$1; S="$(dirname "${BASH_SOURCE[0]}")"; source "$S/lib.sh"
for t in base dev; do [[ $(T $t)/mx-intarsia -nt $(T $t)/Cargo.toml ]] || { log "NO FRESH BINARY in $(T $t); abort"; exit 3; }; done
log "w7 start; budget $(left) s; base $(cat $(T base)/.stamp) dev $(cat $(T dev)/.stamp)"
need 60 "plan" && plan dev dev-27 "$M27"
log "plan dev 27B: $(grep -aoE '"prefill_(chunk|chunk_source|peel)": *[^,]*' "$O/plan-dev-27.out" "$O/plan-dev-27.err" 2>/dev/null | head -3 | tr '\n' '|')"
need 200 "acc" && GT=g27-tails2 acc base base-c256 256
need 200 "acc" && GT=g27-tails2 acc dev dev-default 4096
need 200 "acc" && GT=g27-tails2 E="INTARSIA_PREFILL_PEEL=0" acc dev dev-nopeel 4096
need 100 "accg" && accg base base
need 100 "accg" && accg dev dev
for k in 1029 4100 4360 8201 16k; do
  need 80 "ids" && ids dev dev-$k "$M27" $k
  need 80 "ids" && ids base base-$k "$M27" $k
  same ids-dev-$k.ids ids-base-$k.ids "27B $k ids dev (4096 + peel) vs main (256)"
done
for n in 1024 4096 16384; do
  need 60 "pf" && pf base main-$n "$M27" $n
  need 60 "pf" && pf dev dev-$n "$M27" $n
  need 60 "pf" && E="INTARSIA_HIP_MXFP4_FOLD=1" pf dev fold-$n "$M27" $n
done
# GEMM diagnostics for the next bitwise lever (section 7): v2 with the dequant ALU removed (nodq arm), LDS bank conflicts
need 90 "mxbench" && mxbench dev m4096-nodq 4096 10
need 120 "pmc" && pmcbench dev lds SQ_LDS_BANK_CONFLICT SQ_LDS_IDX_ACTIVE
need 120 "pmc" && pmcbench dev valu SQ_INSTS_VALU SQ_INSTS_LDS SQ_WAIT_INST_LDS
log "w7 done; $(left) s left"

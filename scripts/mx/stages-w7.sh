#!/usr/bin/env bash
# Lane mx window 7 (card 0): the 27B at chunk 4096 + 8-row tail peel vs main's chunk 256. The dev build of this window
# (174eb83) carries the peel but put the recipe table in qwen3_5.toml, which the 27B does not use, so the setting is given
# as `--prefill-chunk 4096` + INTARSIA_PREFILL_PEEL=8 (exactly what the table in qwen3_5_vl.toml resolves to).
# Tails self-golden dumps (34 prompts: P1's golden + synthetic tails of 0..255 rows mod 256 and lengths above 4096) main
# chunk 256 vs dev default vs dev with the peel off (INTARSIA_PREFILL_PEEL=0, negative control), compared offline with
# scripts/ck/chunkcmp.py; the 27B golden gate (prefill + teacher-forced decode) dev default; 256 greedy ids through the
# engine at prompt lengths whose last multi-row pass is 3..8 rows (1029, 4100, 4360, 8201) and 16k; plan show; prefill
# speed main vs dev default at 1k / 4k / 16k (+ fold opt-in); mxbench with the nodq arm and LDS / VALU counters.
set -uo pipefail
O=$1; S="$(dirname "${BASH_SOURCE[0]}")"; source "$S/lib.sh"
for t in base dev; do [[ $(T $t)/mx-intarsia -nt $(T $t)/Cargo.toml ]] || { log "NO FRESH BINARY in $(T $t); abort"; exit 3; }; done
log "w7 start; budget $(left) s; base $(cat $(T base)/.stamp) dev $(cat $(T dev)/.stamp)"
# batch tests: batch_recurrent on dev (w6's cargo run stopped at batch_cuda_real's failures), and main's run of the three
# batch_cuda_real tests that failed on dev under --include-ignored (CUDA-only expectations), from a base mtest build here
need 600 "mtest" && mtest dev recurrent --test batch_recurrent -- --test-threads=1 --include-ignored
need 300 "build" && { TREES="base=$(cat $(T base)/.stamp)" SEED_FROM=intarsia-cargo-target-mx-dev-mtest MTEST=1 bash "$S/build.sh" > "$O/build-base-mtest.txt" 2>&1; log "base mtest build: $(grep -E 'models tests build' "$O/build-base-mtest.txt")"; }
need 400 "mtest" && mtest base cudareal3 --test batch_cuda_real -- --test-threads=1 --include-ignored real_35b_batch_step_sampled_timing real_35b_busy_chunk_prefill_kl_gate real_qwen3_8b_batched_decode_is_bitwise_the_solo_step
need 60 "plan" && E="INTARSIA_PREFILL_PEEL=8" plan dev dev-27 "$M27" --prefill-chunk 4096
log "plan dev 27B: $(grep -aoE '"prefill_(chunk|chunk_source|peel)": *[^,]*' "$O/plan-dev-27.out" "$O/plan-dev-27.err" 2>/dev/null | head -3 | tr '\n' '|')"
need 200 "acc" && GT=g27-tails2 acc base base-c256 256
need 200 "acc" && GT=g27-tails2 E="INTARSIA_PREFILL_PEEL=8" acc dev dev-peel 4096
need 200 "acc" && GT=g27-tails2 E="INTARSIA_PREFILL_PEEL=0" acc dev dev-nopeel 4096
need 100 "accg" && accg base base
need 100 "accg" && E="INTARSIA_PREFILL_PEEL=8" accg dev dev --prefill-chunk 4096
for k in 1029 4100 4360 8201 16k; do
  need 80 "ids" && E="INTARSIA_PREFILL_PEEL=8" ids dev dev-$k "$M27" $k --prefill-chunk 4096
  need 80 "ids" && ids base base-$k "$M27" $k
  same ids-dev-$k.ids ids-base-$k.ids "27B $k ids dev (4096 + peel) vs main (256)"
done
for n in 1024 4096 16384; do
  need 60 "pf" && pf base main-$n "$M27" $n
  need 60 "pf" && pf dev v2-$n "$M27" $n
  need 60 "pf" && E="INTARSIA_PREFILL_PEEL=8" pf dev peel-$n "$M27" $n --prefill-chunk 4096
  need 60 "pf" && E="INTARSIA_PREFILL_PEEL=8 INTARSIA_HIP_MXFP4_FOLD=1" pf dev peelfold-$n "$M27" $n --prefill-chunk 4096
done
# 35B: KL golden (prefill + teacher-forced decode) main vs dev, acc-short (qualify, golden dir with the HIP baseline) on dev
need 120 "accg35" && accg35 base base
need 120 "accg35" && accg35 dev dev
need 300 "qual" && qual dev 35g "$M35" --golden-dir /work/golden
# GEMM diagnostics for the next bitwise lever (section 7): v2 with the dequant ALU removed (nodq arm), LDS bank conflicts
need 90 "mxbench" && mxbench dev m4096-nodq 4096 10
need 120 "pmc" && pmcbench dev lds SQ_LDS_BANK_CONFLICT SQ_LDS_IDX_ACTIVE
need 120 "pmc" && pmcbench dev valu SQ_INSTS_VALU SQ_INSTS_LDS SQ_WAIT_INST_LDS
log "w7 done; $(left) s left"

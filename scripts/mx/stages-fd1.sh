#!/usr/bin/env bash
# Lane mx fold-default, window G1 (card 0): gates of the fold as the default W4A4 MXFP4 prefill GEMM (user decision
# 2026-10-25 13:14; plan: docs/results/2026-10-25-mx-fold-bitwise-note.md section 6). Trees: dev = the branch, base = main.
# HARD STOPS (no re-record, fallback C): the widened fold_rows test, S4 batched == solo, the tails sweep (its dumps are
# compared on the Mac with scripts/ck/chunkcmp.py after this window, before G2 may run). Order: hard stops first.
set -uo pipefail
O=$1; S="$(dirname "${BASH_SOURCE[0]}")"; source "$S/lib.sh"
G16=/work/p1-golden/qwen3.8-27b-mxfp4-hfquark-w4a16-top20.safetensors
B16=/work/golden/baselines/qwen3.8-27b-mxfp4-hfquark-w4a16-top20.hip.baseline.json
E16=/work/golden/baselines/qwen3.8-27b-mxfp4-hfquark-w4a16-top20.hip.envelope.json
B4=/work/golden/baselines/qwen3.8-27b-mxfp4-hfquark-top20.hip.baseline.json
E4=/work/golden/baselines/qwen3.8-27b-mxfp4-hfquark-top20.hip.envelope.json
PROV="--pre-change bbc0cde --revlist /work/mx-out/revlist"   # #150's baselines were recorded from bbc0cde (an ancestor)
for t in base dev; do [[ $(T $t)/mx-intarsia -nt $(T $t)/Cargo.toml ]] || { log "NO FRESH BINARY in $(T $t); abort"; exit 3; }; done
log "fd1 start; budget $(left) s; base $(cat $(T base)/.stamp) dev $(cat $(T dev)/.stamp)"
# 1. HIP tests: the widened fold_rows (HARD STOP), the rest of mxfp4_fold, v2, rows_exact
need 400 "hip tests" && hiptest dev mx --test mxfp4_fold --test mxfp4_v2 --test rows_exact -- --test-threads=1 --nocapture
# 2. 27B S4 batched decode == solo step at max-batch 16 with the fold default (HARD STOP)
need 800 "S4" && mtest27 dev s4 --test batch_cuda_real real_35b_batched_decode_is_bitwise -- --test-threads=1 --nocapture
# 3. tails sweep dumps under the fold (HARD STOP, compared on the Mac): chunk 256 vs chunk 4096 + peel 16
need 420 "tails c256" && GT=g27-tails2 acc dev c256 256
need 380 "tails c4096" && GT=g27-tails2 E="INTARSIA_PREFILL_PEEL=16" acc dev c4096p16 4096
# 4. kernels test --quick
need 60 "kq" && kq dev 27 "$M27"
need 60 "kq" && kq dev 35 "$M35"
# 5. W4A16 golden (--mxfp4-act none): dumps main vs branch (must be bitwise) and the gate against #150's baseline + envelope
A16="--golden $G16 --mode both --decode-positions 16 --max-seq 4096 --mxfp4-act none"
need 90 "w4a16 base" && run base 300 a16-dump-base "$HUB" accuracy --backend hip --model $M27 --mem-budget 29 $A16 --write-self-golden /work/mx-out/a16-base.safetensors --allow-unstamped --allow-dirty
need 90 "w4a16 dev" && run dev 300 a16-gate "$HUB" accuracy --backend hip --model $M27 --mem-budget 29 $A16 --baseline $B16 --noise-envelope $E16 $PROV --write-self-golden /work/mx-out/a16-dev.safetensors
# 6. W4A4 golden: vs HF (prefill gate) and decode vs #150's baseline + envelope (expected to move: recorded, not a stop)
A4="--golden $G27 --mode both --decode-positions 16 --max-seq 4096 --ref-noise $N27"
need 90 "w4a4 dev" && run dev 300 a4-gate-old-baseline "$HUB" accuracy --backend hip --model $M27 --mem-budget 29 $A4 --baseline $B4 --noise-envelope $E4 $PROV --write-self-golden /work/mx-out/a4-dev.safetensors
need 60 "accg" && accg dev dev   # P1's settings (the 0.165 / 0.261 numbers of #144)
# 7. 35B no-op: plan show + 256 ids short / 4k, branch vs main
need 30 "plan" && plan base base-35 "$M35"
need 30 "plan" && plan dev dev-35 "$M35"
same plan-base-35.out plan-dev-35.out "plan show 35B main vs branch"
for k in short 4k; do
  need 40 "ids" && ids dev dev-35-$k "$M35" $k
  need 40 "ids" && ids base base-35-$k "$M35" $k
  same ids-dev-35-$k.ids ids-base-35-$k.ids "35B $k ids branch vs main"
done
# 8. 27B --spec mtp ids == spec off (branch), short / 4k
for k in short 4k; do
  need 40 "ids" && ids dev off-27-$k "$M27" $k --spec off
  need 40 "ids" && ids dev mtp-27-$k "$M27" $k --spec mtp --spec-k 2
  same ids-off-27-$k.ids ids-mtp-27-$k.ids "27B $k ids spec mtp vs off (branch)"
done
# 9. batch_recurrent
need 150 "mtest" && mtest dev recurrent --test batch_recurrent -- --test-threads=1 --include-ignored
for f in a16-base a16-dev a4-dev; do cp "$(T dev)/mx-out/$f.safetensors" "$O/" 2>/dev/null || cp "$(T base)/mx-out/$f.safetensors" "$O/" 2>/dev/null; done
log "fd1 done; $(left) s left"

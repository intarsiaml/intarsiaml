#!/usr/bin/env bash
# O5 (docs/decisions/LOG.md): is the backend's prefill invariant to a 1-row (or 2-row, ...) tail chunk? Runs ON the GPU
# box, inside a fleetq slot, for Qwen3-8B BF16 (attention-only, so Backend::avoid_single_row_prefill would apply; the hybrid
# 35B is excluded by !has_recurrent()). For each chunk size C, `intarsia accuracy` prefills the three prompts of the 8B 4k
# golden (4064 / 4064 / 4557 tokens) in chunks C, C, .., L mod C and dumps its own top-20 rows (--write-self-golden); the
# tail of a prompt is L mod C rows. tailcmp.py compares every dump bit by bit against the single-chunk run.
#   o5-tails.sh cuda|hip <outdir> [prefill|decode|all]
#   CUDA: tree ~/intarsia-decfuse/og-main (scripts/cuda-decode/build.sh og-main), golden ~/og-data/.
#   HIP : tree ~/og-hip in the dev image (deploy/tatooine/dev.sh), INTARSIA_TARGET_VOL volume, golden ~/og-hip/og-data/;
#         <outdir> must be under ~/og-hip (the container sees it as /work/...). Needs radiance down.
set -u
BE=${1:?cuda|hip}; OUT=${2:?outdir}; WHICH=${3:-all}
mkdir -p "$OUT"
case "$BE" in
  cuda)
    BIN=${BIN:-$HOME/intarsia-decfuse/og-main/target/acc/release/intarsia}
    M8=${M8:-/home/nvidia/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}
    G=${G:-$HOME/og-data/qwen3-8b-4k-fp32-top20.safetensors}
    BUDGET=${BUDGET:-26}
    OUTIN=$OUT
    run() { "$BIN" "$@"; }
    ;;
  hip)
    REPO=${INTARSIA_REPO:-$HOME/og-hip}
    export INTARSIA_REPO=$REPO INTARSIA_TARGET_VOL=${INTARSIA_TARGET_VOL:-intarsia-cargo-target-og} HIP_VISIBLE_DEVICES=${RS_GPU:-0}
    M8=/models/Qwen3-8B; G=/work/og-data/qwen3-8b-4k-fp32-top20.safetensors; BUDGET=${BUDGET:-29}
    OUTIN=/work/${OUT#$REPO/}
    run() { (cd "$REPO" && deploy/tatooine/dev.sh /target/release/intarsia "$@"); }
    ;;
  *) echo "backend?"; exit 2 ;;
esac
# prefill-mode chunks (prompt lengths 4064 / 4064 / 4557). Reference: 4557 = every prompt in ONE chunk (no tail).
#  4096          one chunk for the two 4064 prompts, (4096, 461) for the 4557 one
#  2032 1016     no tail, 2 / 4 chunks of the 4064 prompts (multi-chunk control)
#  239           4064 = 17 * 239 + 1: the realistic chunk*k+1 (tail 1)       4557 = 19 * 239 + 16
#  2031          4064 = 2 * 2031 + 2: chunk*k+2 (tail 2)
#  4063 4062 4061 4060 4059 4056 4048: tails 1 2 3 4 5 8 16 behind one big chunk
#  4556 4555 2278: the same for the 4557 prompt (tail 1, tail 2, 2 * 2278 + 1)
#  4032 4000 3936 3808: tails 32 64 128 256 (where does a small tail become invariant? added after the first HIP window
#  showed every tail of 1..16 rows differs from the single-chunk run)
PRE=${PRE:-4557 4096 2032 1016 239 2031 4063 4062 4061 4060 4059 4056 4048 4556 4555 2278 4032 4000 3936 3808}
# decode-mode chunks (prefill of 4000 / 4000 / 4493 tokens, then teacher-forced decode steps over the continuation):
#  4493 reference, 2000 no tail (2 chunks), 3999 / 4492 tail 1, 3998 / 1999 tail 2
DEC=${DEC:-4493 2000 3999 3998 1999 4492}
one() { # one <mode> <chunk>
  local mode=$1 c=$2 t0; t0=$(date +%s)
  run accuracy --backend "$BE" --model "$M8" --golden "$G" --mode "$mode" --prefill-chunk "$c" --max-seq 8192 --mem-budget "$BUDGET" \
    --decode-positions 16 --gate-class bf16 --allow-unstamped --allow-dirty --write-self-golden "$OUTIN/$mode-c$c.safetensors" --self-topk 20 \
    > "$OUT/$mode-c$c.log" 2>&1 < /dev/null
  echo "$mode chunk $c exit $? $(( $(date +%s) - t0 )) s  $(ls -la "$OUT/$mode-c$c.safetensors" 2>/dev/null | awk '{print $5}') bytes" | tee -a "$OUT/steps.log"
}
echo "== o5-tails $BE start $(date +%T) host $(hostname)" | tee -a "$OUT/steps.log"
if [[ $WHICH == prefill || $WHICH == all ]]; then for c in $PRE; do one prefill "$c"; done; fi
if [[ $WHICH == decode || $WHICH == all ]]; then for c in $DEC; do one decode "$c"; done; fi
echo "== o5-tails $BE done $(date +%T)" | tee -a "$OUT/steps.log"

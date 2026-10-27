#!/usr/bin/env bash
# Lane H27P planted mutants of the chunk-2048 + peel-16 recipe change, ON tatooine INSIDE a GPU window (win.sh). Each MUST make the ids gate
# (ids.sh, base = origin/main chunk 256) FAIL on the named prompts; the clean arm of the same prompts is the 18-prompt run (all PASS).
#   m1 no peel:        INTARSIA_PREFILL_PEEL=0 (the 1..16-row tail of 16k-3 runs on the GEMM tiles, not the row kernels)
#   m4 all-reduce bug: INTARSIA_TP_TEST_BUG=scale:1.003@1 (rank 1's partial scaled by 0.3 %: the tp gate's sensitivity mutant)
#   m5 no collectives: INTARSIA_TP_NO_COMM=1 (garbage activations)
# m2 (recipe chunk 2048 -> 1024) and m3 (peel 16 -> 0) are CPU mutants of tests/recipe_27b_chunk.rs, run on the Mac.
# Then the kernel admission sweep of the dev binary (kernels are untouched by this change; it must stay clean).
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
S="$(dirname "${BASH_SOURCE[0]}")"
mut() { # mut <tag> <env assignment> <prompts...>
  local tag=$1 e=$2; shift 2
  local d=$HOME/intarsiaml-h27p-logs/mut-$tag; mkdir -p "$d"; cp $HOME/intarsiaml-h27p-logs/ids/gen-*-base.ids "$d/" 2>/dev/null
  echo "== mutant $tag ($e): MUST FAIL"
  DEVENV="env $e" REUSE_BASE=1 O="$d" bash "$S/ids.sh" "$@" 2>&1 | grep -aE "^(PASS|FAIL)|fails="
}
mut m1 INTARSIA_PREFILL_PEEL=0 16k-3
mut m4 INTARSIA_TP_TEST_BUG=scale:1.003@1 4k-0 4k-3 16k-0
mut m5 INTARSIA_TP_NO_COMM=1 4k-0
echo "== kernel admission (dev binary, kernels untouched)"
gpu 230 /work/h27p-intarsia kernels test --backend hip --quick 2>&1 | tail -6

#!/bin/bash
# Knockout timings of attn_sparse_mma (PF=false form) in the bench: KNOCK=1 no key gather (zero tile), 2 no QK mma, 3 no online softmax,
# 4 no PV mma, 5 none of 2-4 (gather + barriers only). Timing only (the outputs are garbage). Run on a Spark inside a fleetq slot.
set -eu
SRC=${1:?repo src}; B=$(cd "$(dirname "$0")" && pwd); OUT=${OUT:-$HOME/intarsia-dsv41-attnfuse/bench}
export PATH=/usr/local/cuda/bin:$PATH
python3 - "$SRC/kernels/cuda/dsv41_attn.cu" "$OUT/attn_knock.cu" <<'PY'
import sys
s = open(sys.argv[1]).read()
def rep(a, b):
    global s
    assert a in s, a
    s = s.replace(a, b, 1)
import re
def sub(pat, repl):
    global s
    n = len(re.findall(pat, s))
    assert n == 1, (pat, n)
    s = re.sub(pat, repl, s)
sub(r"if \(valid\[r\]\) \{\n(\s+)const int64_t sr = srcrow\[r\];", r"if (valid[r] && KNOCK != 1) {\n\1const int64_t sr = srcrow[r];")
sub(r"for \(int k0 = 0; k0 < D; k0 \+= 16\)", "for (int k0 = 0; k0 < ((KNOCK == 2 || KNOCK == 5) ? 0 : D); k0 += 16)")
sub(r"(// online softmax: warp w rows 4w\.\.4w\+3, lane = key\n#pragma unroll\n\s+for \(int i = 0; i < )4(; \+\+i\) \{)", r"\g<1>((KNOCK == 3 || KNOCK == 5) ? 0 : 4)\g<2>")
sub(r"(#pragma unroll\n\s+for \(int ks = 0; ks < )2(; \+\+ks\) \{)", r"\g<1>((KNOCK == 4 || KNOCK == 5) ? 0 : 2)\g<2>")
open(sys.argv[2], "w").write(s)
PY
FL="-c -O3 --std=c++17 -lineinfo --expt-relaxed-constexpr -arch=sm_121 -I $SRC/kernels/cuda -I $SRC/kernels"
for k in 0 1 2 3 4 5; do nvcc $FL -DKNOCK=$k $OUT/attn_knock.cu -o $OUT/attn_k$k.o & done
wait
for k in 0 1 2 3 4 5; do
  nvcc -O3 --std=c++17 -arch=sm_121 $B/hc_rope_bench.cu $OUT/dsv41_misc.o $OUT/dsv41_hc.o $OUT/dsv41_hc_sub.o $OUT/attn_k$k.o -o $OUT/bench_k$k
  echo "== KNOCK=$k"; INTARSIA_DSV41_ATTN_PF=0 $OUT/bench_k$k 20 | grep -E "^attn"
done

#!/usr/bin/env bash
# Round-7 accuracy extras, spark1 side (vLLM down, inside the speed window before the A/B): (1) greedy ids of the real
# base build (r7-base, rebuilt from origin/main) against window 1's dev default ids (out/r7A); (2) chat_reuse_4k reuse
# with --reuse-diag (logit-level warm vs cold vs cold-at-another-chunk) for dev default and lt. Output out/r7D.
set -u
P=$HOME/intarsia-decfuse
O=$P/out/r7D; mkdir -p $O
A=$P/out/r7A
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
L=$HOME/intarsiaml-golden/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors
CORPUS=$P/r7-dev/tools/prompts/corpus_docs.txt
for t in r7-base r7-dev; do echo "bin $t: ik_matmul_fp8_gemm_lt symbols: $(grep -c ik_matmul_fp8_gemm_lt $P/$t/target/acc/release/intarsia)"; done
gen() { # gen <tag> <tree> <prompt chars> <tokens>
  timeout 300 $P/$2/target/acc/release/intarsia generate -m $M35 --backend cuda --mem-budget 29 --max-seq 20480 --chat --no-think --profile-every 0 --ignore-eos --show-ids \
    -p "$(head -c $3 $CORPUS)" --max-tokens $4 > $O/gen-$1.out 2> $O/gen-$1.err
  grep -a '^ids:' $O/gen-$1.err > $O/gen-$1.ids
  echo "gen $1: $(grep -aE 'finish=' $O/gen-$1.err | head -1 | cut -c1-160)"
}
gen base-graph r7-base 5000 256
gen base-16k r7-base 54000 64
for a in "dev INTARSIA_X=0" "lt INTARSIA_CUDA_FP8_GEMM_LT=1"; do
  set -- $a
  env $2 timeout 400 $P/r7-dev/target/acc/release/intarsia accuracy --model $M35 --backend cuda --golden $L --mode reuse --reuse-diag \
    --reuse-prompt chat_reuse_4k --max-seq 16384 --mem-budget 40 --prefill-chunk 2048 --allow-dirty > $O/reusediag-$1.log 2>&1
  echo "reusediag $1 exit $?: $(grep -aE '^chat_reuse_4k' $O/reusediag-$1.log | cut -c1-90 | tr -s ' ' | tr '\n' '|')"
done
python3 - $O $A <<'PY'
import sys
o, a = sys.argv[1], sys.argv[2]
def ids(p):
    try:
        s = open(p).read().split(":", 1)[1]
        return [int(v) for v in s.replace("[", " ").replace("]", " ").replace(",", " ").split()]
    except Exception:
        return None
# window 1's "base-16k" ran the mis-built r7-base binary (dev's kernels and registry, LT off): a dev default run
for b, d in [("base-graph", "dev-graph"), ("base-16k", "base-16k"), ("base-graph", "lt-graph"), ("base-16k", "lt-16k")]:
    x, y = ids(f"{o}/gen-{b}.ids"), ids(f"{a}/gen-{d}.ids")
    if not x or y is None: print(f"{d} vs real {b}: missing"); continue
    k = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), None)
    print(f"{d} (window 1) vs real {b}: " + ("identical" if k is None and len(x) == len(y) else f"first difference at generated token {k} of {len(x)}"))
PY

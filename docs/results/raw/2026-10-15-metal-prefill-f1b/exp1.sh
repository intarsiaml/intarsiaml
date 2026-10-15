#!/bin/zsh
# Experiment 1: P-mode variants of the NAX attention kernel (p0 bf16 = F1, p1 half, p2 float x bf16, p3 float x float,
# p4 bf16 hi+lo, p5 / p6 = p2 / p3 with relaxed precision, p7 = p1 with P scaled by 2^12, p8 = p0 with l summed over the
# rounded P).
# f64 check, numerics, kernel microbench, then end-to-end accuracy (both goldens) for base, F1 and every sane variant.
# Run inside a fleetq slot.
W=/opt/code/intarsiaml/.claude/worktrees/agent-afe24a73270ed0c31
RAW=$W/docs/results/raw/2026-10-15-metal-prefill-f1b
R=$RAW/exp1
S=/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/f1b
mkdir -p $R
cd $W
echo "start $(date)"
cargo test --release -p intarsia-backend-metal --test flash_prefill_nax -- --test-threads=1 --nocapture > $R/test.txt 2>&1
echo "test exit $? $(date)"
grep -E "worst error|numerics|panicked|test result|FAILED|failing" $R/test.txt
cargo build --release -p intarsia-backend-metal --example flash_bench 2>&1 | tail -1
for v in "" _p1 _p2 _p3 _p4 _p5 _p6 _p7 _p8; do
  $W/target/release/examples/flash_bench --reps 6 attn_nax_d128_s4$v 3840 > $R/bench$v.txt 2>&1
  echo "bench s4$v: $(tail -3 $R/bench$v.txt | tr '\n' ' ')"
done
echo "microbench done $(date)"

# end-to-end accuracy
$RAW/acc.sh base $S/intarsia-base
$RAW/acc.sh f1 $S/intarsia-f1
for k in 0 8 1 7 4 2 5 3 6; do
  v=attn_nax_d128_s4; [ $k != 0 ] && v=attn_nax_d128_s4_p$k
  # skip variants whose numerics are broken (rel rms at scale 1.0 more than 3x p0's, or not finite)
  ok=$(python3 -I - $R/test.txt $v <<'EOF'
import re, sys
t = open(sys.argv[1]).read()
def rr(name):
    m = re.findall(r"numerics scale 1.0000 " + re.escape(name) + r": rel rms ([0-9.e+-]+)", t)
    return float(m[0]) if m else None
base = rr("attn_nax_d128_s4"); x = rr(sys.argv[2])
print("yes" if (x is not None and base is not None and x < 3 * base) else "no")
EOF
)
  if [ "$ok" = yes ]; then
    echo "== accuracy variant p$k $(date)"
    $RAW/acc.sh var-p$k $S/intarsia-var3 $v
  else
    echo "== SKIP accuracy variant p$k (numerics broken or missing)"
  fi
done
echo "all done $(date)"

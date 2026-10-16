#!/bin/zsh
# Trimmed-branch gates (follow-up): cargo test metal, kernels test, CLI generate greedy ids at L=2049 vs base 00abb36. Exclusive slot.
# gates2.sh <outdir>
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a slot" >&2; exit 2; }
O=$1; mkdir -p $O
W=/opt/code/intarsiaml-cbgap; R=$W/docs/results/raw/2026-10-16-metal-prefill-cbgap
SC=/private/tmp/claude-501/-opt-code-intarsiaml/0e8026c5-6dd0-4b83-b375-5ff0a66894fc/scratchpad/cbgap
M=/Users/josef/models/Qwen3-8B
export INTARSIA_CPU_FREE_MEM_GB=60
cd $W
for ARM in base2 t1; do for f in p2049s1 p2049s2; do for C in 256 2048; do
  $SC/intarsia-$ARM generate --model $M --backend metal --max-seq 8192 --profile-every 0 --temperature 0 --max-tokens 48 --ignore-eos --show-ids \
    --prefill-chunk $C --prompt "$(cat $R/prompts/$f.txt)" > $O/gen-$ARM-$f-c$C.out 2> $O/gen-$ARM-$f-c$C.err
  echo "[$ARM] generate $f chunk=$C exit $?"
done; done; done
for f in p2049s1 p2049s2; do for C in 256 2048; do
  python3 $R/first_divergence.py $O/gen-base2-$f-c$C.err $O/gen-t1-$f-c$C.err "$f chunk $C base 00abb36 vs new"
  cmp $O/gen-base2-$f-c$C.out $O/gen-t1-$f-c$C.out && echo "$f chunk $C stdout byte-identical"
done; done
echo "== intarsia kernels test"
$SC/intarsia-t1 kernels test --backend metal --device 0 --model $M -v > $O/kernels-test.txt 2>&1; echo "kernels test exit $?"; tail -1 $O/kernels-test.txt
echo "== cargo test -p intarsia-backend-metal --release"
cargo test -p intarsia-backend-metal --release -- --test-threads=1 > $O/cargo-test-metal.txt 2>&1; echo "cargo test metal exit $?"
grep -h "test result" $O/cargo-test-metal.txt | awk '{p+=$4; f+=$6} END {print "total passed", p, "failed", f}'
echo GATES2_DONE

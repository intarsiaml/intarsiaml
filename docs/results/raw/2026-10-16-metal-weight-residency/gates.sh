#!/bin/zsh
# Gates for the keep-warm branch. Binaries in the scratch dir: intarsia-base2 (00abb36), intarsia-base (6d99a6f), intarsia-new (this branch).
# gates.sh <outdir>   Exclusive slot.
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a slot" >&2; exit 2; }
O=$1; mkdir -p $O
W=/opt/code/intarsiaml-residency; R=$W/docs/results/raw/2026-10-16-metal-weight-residency; P=$W/docs/results/raw/2026-10-16-metal-prefill-cbgap/prompts
SC=/private/tmp/claude-501/-opt-code-intarsiaml/0e8026c5-6dd0-4b83-b375-5ff0a66894fc/scratchpad/resid
M=/Users/josef/models/Qwen3-8B; G=/opt/code/intarsiaml/golden
export INTARSIA_CPU_FREE_MEM_GB=60
cd $W
echo "== generate (CLI) greedy ids, 48 tokens, L=2049"
for ARM in base2 base new newkw; do for f in p2049s1 p2049s2; do for C in 256 2048; do
  BIN=$SC/intarsia-$ARM; E=(A=1)
  [ $ARM = newkw ] && { BIN=$SC/intarsia-new; E=(INTARSIA_METAL_KEEPWARM=1); }
  env $E $BIN generate --model $M --backend metal --max-seq 8192 --profile-every 0 --temperature 0 --max-tokens 48 --ignore-eos --show-ids \
    --prefill-chunk $C --prompt "$(cat $P/$f.txt)" > $O/gen-$ARM-$f-c$C.out 2> $O/gen-$ARM-$f-c$C.err
  echo "[$ARM] generate $f chunk=$C exit $?"
done; done; done
for ARM in base new newkw; do for f in p2049s1 p2049s2; do for C in 256 2048; do
  python3 $R/first_divergence.py $O/gen-base2-$f-c$C.err $O/gen-$ARM-$f-c$C.err "$f chunk $C base 00abb36 vs $ARM"
  cmp $O/gen-base2-$f-c$C.out $O/gen-$ARM-$f-c$C.out && echo "$f chunk $C $ARM stdout byte-identical to 00abb36"
done; done; done
echo "== accuracy short / 4k, base 6d99a6f vs new"
for ARM in base new; do for C in 256 2048; do
  $SC/intarsia-$ARM accuracy --model $M --backend metal --spec off --prefill-chunk $C --golden $G/qwen3-8b-fp32.safetensors --mode both --decode-positions 16 --max-seq 4096 > $O/acc-$ARM-c$C-short.log 2>&1; echo "[$ARM] acc c$C short exit $?"
  $SC/intarsia-$ARM accuracy --model $M --backend metal --spec off --prefill-chunk $C --golden $G/qwen3-8b-4k-fp32-top20.safetensors --mode all --decode-positions 64 --max-seq 8192 --min-top1 0 > $O/acc-$ARM-c$C-4k.log 2>&1; echo "[$ARM] acc c$C 4k exit $?"
done; done
for C in 256 2048; do for k in short 4k; do
  if cmp -s $O/acc-base-c$C-$k.log $O/acc-new-c$C-$k.log; then echo "acc c$C $k: logs byte-identical"; else echo "acc c$C $k: DIFFER"; diff $O/acc-base-c$C-$k.log $O/acc-new-c$C-$k.log | head; fi
done; done
echo "== intarsia kernels test"
$SC/intarsia-new kernels test --backend metal --device 0 --model $M -v > $O/kernels-test.txt 2>&1; echo "kernels test exit $?"; tail -1 $O/kernels-test.txt
echo "== cargo test -p intarsia-backend-metal --release"
cargo test -p intarsia-backend-metal --release -- --test-threads=1 > $O/cargo-test-metal.txt 2>&1; echo "cargo test metal exit $?"
grep -h "test result" $O/cargo-test-metal.txt | awk '{p+=$4; f+=$6} END {print "total passed", p, "failed", f}'
echo GATES_DONE

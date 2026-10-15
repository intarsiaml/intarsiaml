#!/bin/zsh
# Merge bar, GPU parts (inside one exclusive slot, FLEETQ_JOB set by slot.sh). Binary built from the tip (stamped).
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a slot" >&2; exit 2; }
[ -d "$DEVELOPER_DIR" ] || { echo "DEVELOPER_DIR missing" >&2; exit 3; }
W=/opt/code/intarsiaml-spec
R=$W/docs/results/raw/2026-10-15-metal-spec-g2g3/merge
SB=/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/bin
FIN=$SB/intarsia-gate-stamped; BASE=$SB/intarsia-base-c556008
M=/Users/josef/models/Qwen3-8B
G=/opt/code/intarsiaml/golden; G4=$G/qwen3-8b-4k-fp32-top20.safetensors
SBL=$W/golden/baselines; H0=9eeb96c
PB=/opt/code/intarsiaml/docs/results/raw/2026-10-14-metal-phase-b
export INTARSIA_CPU_FREE_MEM_GB=60
mkdir -p $R/{tests,kernels,accuracy,identity,qualify}
run() { local n=$1; shift; "$@" > $n.out 2> $n.err; echo "$n exit $? $(date +%T)"; }
cd $W
echo "stamp: $(scripts/tree-stamp.sh)"; git rev-parse HEAD
echo "== 2. cargo test -p intarsia-backend-metal --release"
cargo test -p intarsia-backend-metal --release -- --test-threads=1 > $R/tests/cargo-test-metal.txt 2>&1; echo "exit $?"
awk '/^test result/{p+=$4; f+=$6} END{print "metal tests passed",p,"failed",f}' $R/tests/cargo-test-metal.txt
echo "== 3. kernels test"
run $R/kernels/kernels-test $FIN kernels test --backend metal -m $M -v
tail -3 $R/kernels/kernels-test.out | cut -c1-200
echo "== 4. accuracy (spec off, default), baseline-gated"
run $R/accuracy/short $FIN accuracy --model $M --backend metal --golden $G/qwen3-8b-fp32.safetensors --gate-class bf16 --max-seq 4096 --mode both --baseline $SBL/qwen3-8b-fp32.metal.baseline.json --noise-envelope $SBL/qwen3-8b-fp32.metal.envelope.json --allow-same-code --pre-change $H0
run $R/accuracy/long $FIN accuracy --model $M --backend metal --golden $G/qwen3-8b-long-fp32.safetensors --gate-class bf16 --max-seq 8192 --mode both --decode-positions 64 --baseline $SBL/qwen3-8b-long-fp32.metal.baseline.json --noise-envelope $SBL/qwen3-8b-long-fp32.metal.envelope.json --allow-same-code --pre-change $H0
run $R/accuracy/4k $FIN accuracy --model $M --backend metal --golden $G4 --max-seq 8192 --mode all --decode-positions 64 --min-top1 0 --baseline $SBL/qwen3-8b-4k-fp32-top20.metal.baseline.json --noise-envelope $SBL/qwen3-8b-4k-fp32-top20.metal.envelope.json --allow-same-code --pre-change $H0
echo "== 5. greedy ids, spike/metal c556008 vs tip with --spec off (short, 4k, w4107 x 48, 256)"
for N in 48 256; do
  for P in short 4k w4107; do
    if [ $P = w4107 ]; then PF=$R/w4107.txt; else PF=$PB/prompt-$P.txt; fi
    for B in base off; do
      BIN=$FIN; SPEC=(--spec off); [ $B = base ] && { BIN=$BASE; SPEC=(); }
      $BIN generate --model $M --backend metal --max-seq 8192 --profile-every 0 --temperature 0 --max-tokens $N --ignore-eos --show-ids $SPEC --prompt "$(cat $PF)" > $R/identity/$B-$P-$N.out 2> $R/identity/$B-$P-$N.err
    done
    python3 -I $R/first_divergence.py $R/identity/base-$P-$N.err $R/identity/off-$P-$N.err "$P $N tokens spike/metal vs tip --spec off" | tee -a $R/identity/summary.txt
  done
done
echo "== 5b. cut-path coverage: the same stop / max / min_tokens / EOS cases with the gate off (INTARSIA_SPEC_GATE=0,4,64,1 = fails 0, min suffix 1: the G1 drafter), on4 + on6, vs base"
RR=$W/docs/results/raw/2026-10-15-metal-spec-g2g3
mkdir -p $RR/identity-nogate
for f in $RR/identity/base-*.err; do n=${f:t}; c=${n#base-}; c=${c%.err}; grep -q "^$c|" $RR/cases-cut.txt && cp ${f%.err}.* $RR/identity-nogate/; done
INTARSIA_SPEC_GATE=0,4,64,1 $RR/run_identity.sh $RR/identity-nogate $RR/cases-cut.txt on4 on6
python3 $W/docs/results/raw/2026-10-15-metal-spec-g1/compare.py $RR/identity-nogate base > $RR/identity-nogate-vs-base.md; tail -1 $RR/identity-nogate-vs-base.md
echo "== 6. qualify (full, fresh)"
$FIN qualify --model $M --backend metal --bandwidth-gbs 614 --golden-dir $G --out $R/qualify > $R/qualify-console.log 2>&1; echo "qualify exit $?"
grep -E "^\[qualify.*(PASS|FAIL|SKIP|ERROR)" $R/qualify-console.log | cut -c1-260
echo done

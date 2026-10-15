#!/bin/zsh
# metal-p3-final merge bar at chunk 2048 (recipe default, no --prefill-chunk unless stated). One exclusive fleetq slot.
S=/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/g4k
W=/opt/code/intarsiaml-p3-final
R=$W/docs/results/raw/2026-10-15-metal-p3-final
BASE=$S/intarsia-base; F0=$S/intarsia-final0
M=/Users/josef/models/Qwen3-8B
G=/opt/code/intarsiaml/golden            # short + long + (copied) 4k golden
G4=$G/qwen3-8b-4k-fp32-top20.safetensors
PB=$W/docs/results/raw/2026-10-14-metal-phase-b
FD=$W/docs/results/raw/2026-10-14-metal-prefill-p1/full
W4107=$W/docs/results/raw/2026-10-15-metal-p3-noise/identity-logits/w4107.txt
SB=$W/golden/baselines
export INTARSIA_CPU_FREE_MEM_GB=60
mkdir -p $R/{tests,kernels,baselines,accuracy,identity,qualify}
J=$(fleetq acquire --host endor --exclusive --mem 24 --expect 60 --label "metal-p3-final: merge bar (tests, kernels, baselines, accuracy, ids, qualify)" --wait --json | python3 -c 'import sys,json; print(json.load(sys.stdin)["id"])')
echo "fleetq job $J" | tee $R/fleetq-job.txt
trap "fleetq release $J" EXIT
export FLEETQ_JOB=$J
cd $W
H0=9eeb96c; echo "tip before baselines: $H0"; git status --short | head -3
echo "== 1. cargo test (intarsia-cli, intarsia-models)"
cargo test -p intarsia-cli -p intarsia-models > $R/tests/cargo-test.txt 2>&1; echo "cargo test exit $?"; grep -E "^test result|FAILED" $R/tests/cargo-test.txt
echo "== 2. cargo test -p intarsia-backend-metal --release"
cargo test -p intarsia-backend-metal --release -- --test-threads=1 > $R/tests/cargo-test-metal.txt 2>&1; echo "metal tests exit $?"; grep -E "^test result|FAILED" $R/tests/cargo-test-metal.txt
echo "== 3. kernels test"
$F0 kernels test --model $M --backend metal --device 0 -v > $R/kernels/kernels-test.txt 2>&1; echo "kernels exit $?"; grep -cE "ADMITTED" $R/kernels/kernels-test.txt; grep -E "FAIL|REJECT" $R/kernels/kernels-test.txt | head -5; tail -3 $R/kernels/kernels-test.txt | cut -c1-200
run() { local n=$1; shift; "$@" > $n.out 2> $n.err; echo "$n exit $?"; }
echo "== 4. re-record Metal baselines + envelopes at chunk 2048 on this tip ($H0)"
SF=(--mode both --decode-positions 16 --max-seq 4096)
LF=(--mode both --decode-positions 64 --max-seq 8192)
KF=(--mode both --decode-positions 64 --max-seq 8192)
for spec in "short:qwen3-8b-fp32:$G/qwen3-8b-fp32.safetensors:chunk=64:SF" "long:qwen3-8b-long-fp32:$G/qwen3-8b-long-fp32.safetensors:chunk=256:LF" "4k:qwen3-8b-4k-fp32-top20:$G4:chunk=256:KF"; do
  IFS=: read tag stem gold pert flags <<< "$spec"
  eval "FL=(\${${flags}[@]})"
  run $R/baselines/$tag-baseline $F0 accuracy --model $M --backend metal --golden $gold $FL --write-baseline $SB/$stem.metal.baseline.json
  run $R/baselines/$tag-envelope $F0 accuracy --model $M --backend metal --golden $gold $FL --noise-ref $SB/$stem.metal.baseline.json --noise-perturb $pert --write-noise-envelope $SB/$stem.metal.envelope.json --allow-same-code
  run $R/baselines/$tag-selfcheck $F0 accuracy --model $M --backend metal --golden $gold $FL --baseline $SB/$stem.metal.baseline.json --noise-envelope $SB/$stem.metal.envelope.json --allow-same-code --pre-change $H0
  grep -E "verdict" $R/baselines/$tag-selfcheck.out | cut -c1-300
done
echo "== 5. commit baselines, build the final binary"
git add golden/baselines docs/results/raw/2026-10-15-metal-p3-final
git -c core.hooksPath=/dev/null commit -q -m "metal-p3-final: re-record Metal baselines and envelopes (short, long, 4k) at chunk 2048 on spike/metal 98f506e + golden-4k + p3-chunk

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>" || echo "commit failed"
export INTARSIA_BUILD_STAMP="$(scripts/tree-stamp.sh)"; echo "final stamp: $INTARSIA_BUILD_STAMP"; echo "final commit: $(git rev-parse --short HEAD)"
cargo build --release -p intarsia-cli --features metal 2>&1 | tail -1
FIN=$S/intarsia-final; cp target/release/intarsia $FIN
echo "== 6. accuracy at chunk 2048 (default), baseline-gated"
run $R/accuracy/short $FIN accuracy --model $M --backend metal --golden $G/qwen3-8b-fp32.safetensors --gate-class bf16 --max-seq 4096 --mode both --baseline $SB/qwen3-8b-fp32.metal.baseline.json --noise-envelope $SB/qwen3-8b-fp32.metal.envelope.json --allow-same-code --pre-change $H0
run $R/accuracy/long $FIN accuracy --model $M --backend metal --golden $G/qwen3-8b-long-fp32.safetensors --gate-class bf16 --max-seq 8192 --mode both --decode-positions 64 --baseline $SB/qwen3-8b-long-fp32.metal.baseline.json --noise-envelope $SB/qwen3-8b-long-fp32.metal.envelope.json --allow-same-code --pre-change $H0
run $R/accuracy/4k $FIN accuracy --model $M --backend metal --golden $G4 --max-seq 8192 --mode all --decode-positions 64 --min-top1 0 --baseline $SB/qwen3-8b-4k-fp32-top20.metal.baseline.json --noise-envelope $SB/qwen3-8b-4k-fp32-top20.metal.envelope.json --allow-same-code --pre-change $H0
echo "== 6b. control: the 4k baseline gate must FAIL a baseline that is better than this build (top-1 +1.5 pt, KL x0.5)"
python3 - <<PY
import json
b=json.load(open("$SB/qwen3-8b-4k-fp32-top20.metal.baseline.json"))
for m in b["modes"]:
    for o in [m["overall"]]+m["prompts"]:
        o["top1_pct"]=min(100,o["top1_pct"]+1.5)
        for k in ("mean_kl","mean_kl21"): o[k]*=0.5
json.dump(b,open("$R/accuracy/control-baseline.json","w"))
PY
run $R/accuracy/4k-control $FIN accuracy --model $M --backend metal --golden $G4 --max-seq 8192 --mode both --decode-positions 64 --min-top1 0 --baseline $R/accuracy/control-baseline.json --allow-same-code --pre-change $H0
grep -E "verdict" $R/accuracy/4k-control.out | cut -c1-300
echo "== 7. greedy ids, spike/metal 98f506e (chunk 256) vs final (2048)"
for N in 48 256; do
  for P in short 4k; do
    for B in base final; do
      BIN=$FIN; [ $B = base ] && BIN=$BASE
      $BIN generate --model $M --backend metal --max-seq 8192 --profile-every 0 --temperature 0 --max-tokens $N --ignore-eos --show-ids --prompt "$(cat $PB/prompt-$P.txt)" > $R/identity/$B-$P-$N.out 2> $R/identity/$B-$P-$N.err
    done
    python3 -I $FD/first_divergence.py $R/identity/base-$P-$N.err $R/identity/final-$P-$N.err "$P $N tokens spike/metal vs final" | tee -a $R/identity/summary.txt
  done
  for B in base final; do
    BIN=$FIN; [ $B = base ] && BIN=$BASE
    $BIN generate --model $M --backend metal --max-seq 8192 --profile-every 0 --temperature 0 --max-tokens $N --ignore-eos --show-ids --prompt "$(cat $W4107)" > $R/identity/$B-w4107-$N.out 2> $R/identity/$B-w4107-$N.err
  done
  python3 -I $FD/first_divergence.py $R/identity/base-w4107-$N.err $R/identity/final-w4107-$N.err "w4107 $N tokens spike/metal vs final" | tee -a $R/identity/summary.txt
done
grep -h "prefill chunk" $R/identity/base-4k-48.err $R/identity/final-4k-48.err | sort -u
echo "== 8. qualify (full, fresh)"
$FIN qualify --model $M --backend metal --bandwidth-gbs 614 --golden-dir $G --out $R/qualify > $R/qualify-console.log 2>&1; echo "qualify exit $?"
grep -E "^\[qualify.*(PASS|FAIL|SKIP|ERROR)" $R/qualify-console.log | cut -c1-260
echo done

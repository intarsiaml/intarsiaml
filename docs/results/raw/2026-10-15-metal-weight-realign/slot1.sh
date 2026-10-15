#!/bin/zsh
# metal-weight-realign slot 1 (correctness + memory + load time), one exclusive fleetq slot on endor.
# BASE = c556008 (git archive build), OFF = this branch without INTARSIA_METAL_REALIGN, ON = same binary with it.
S=/private/tmp/claude-501/-opt-code-intarsiaml/0e8026c5-6dd0-4b83-b375-5ff0a66894fc/scratchpad
W=/opt/code/intarsiaml-weight-realign
R=$W/docs/results/raw/2026-10-15-metal-weight-realign/slot1
BASE=$S/intarsia-c556008; BIN=$S/intarsia-realign
M=/Users/josef/models/Qwen3-8B
G=/opt/code/intarsiaml/golden
G4=$G/qwen3-8b-4k-fp32-top20.safetensors
SB=$W/golden/baselines
PB=$W/docs/results/raw/2026-10-14-metal-phase-b
FD=$W/docs/results/raw/2026-10-14-metal-prefill-p1/full
W4107=$W/docs/results/raw/2026-10-15-metal-p3-noise/identity-logits/w4107.txt
export INTARSIA_CPU_FREE_MEM_GB=60
mkdir -p $R/{tests,identity,memory,accuracy}
J=$(fleetq acquire --host endor --exclusive --mem 24 --expect 40 --label "metal-weight-realign: slot1 tests, identity, memory, accuracy" --wait --json | python3 -c 'import sys,json; print(json.load(sys.stdin)["id"])')
echo "fleetq job $J" | tee $R/fleetq-job.txt
trap "fleetq release $J" EXIT
export FLEETQ_JOB=$J
cd $W
echo "start $(date)"; git rev-parse HEAD; md5 $BASE $BIN
echo "== 1. tests"
cargo test -p intarsia-backend-metal --release > $R/tests/cargo-test-metal.log 2>&1; echo "cargo test metal exit $?"
cargo test --workspace --release > $R/tests/cargo-test-workspace.log 2>&1; echo "cargo test workspace exit $?"
echo "== 2. greedy ids: BASE vs OFF vs ON"
gen() { # tag bin envflag promptfile N
  local tag=$1 bin=$2 flag=$3 pf=$4 N=$5
  env INTARSIA_METAL_REALIGN=$flag /usr/bin/time -l $bin generate --model $M --backend metal --max-seq 8192 --profile-every 0 --temperature 0 --max-tokens $N --ignore-eos --show-ids --prompt "$(cat $pf)" > $R/identity/$tag.out 2> $R/identity/$tag.err
}
for P in short 4k; do
  for N in 48 256; do
    gen base-$P-$N $BASE 0 $PB/prompt-$P.txt $N
    gen off-$P-$N $BIN 0 $PB/prompt-$P.txt $N
    gen on-$P-$N $BIN 1 $PB/prompt-$P.txt $N
    for X in off on; do python3 -I $FD/first_divergence.py $R/identity/base-$P-$N.err $R/identity/$X-$P-$N.err "$P $N tokens c556008 vs $X" | tee -a $R/identity/summary.txt; done
  done
done
for N in 48 256; do
  gen base-w4107-$N $BASE 0 $W4107 $N; gen off-w4107-$N $BIN 0 $W4107 $N; gen on-w4107-$N $BIN 1 $W4107 $N
  for X in off on; do python3 -I $FD/first_divergence.py $R/identity/base-w4107-$N.err $R/identity/$X-w4107-$N.err "w4107 $N tokens c556008 vs $X" | tee -a $R/identity/summary.txt; done
done
echo "== 3. load time + peak memory, 5 reps each of a 1-token generate (page cache warm)"
for rep in 1 2 3 4 5; do
  for cfg in base off on; do
    case $cfg in base) B=$BASE; F=0;; off) B=$BIN; F=0;; on) B=$BIN; F=1;; esac
    env INTARSIA_METAL_REALIGN=$F /usr/bin/time -l $B generate --model $M --backend metal --max-seq 4096 --profile-every 0 --temperature 0 --max-tokens 1 --prompt hello > $R/memory/load-$cfg-$rep.out 2> $R/memory/load-$cfg-$rep.err
    echo "$cfg $rep: $(grep -h 'loaded' $R/memory/load-$cfg-$rep.err | grep -o 'in [0-9.]*s') | $(grep 'maximum resident' $R/memory/load-$cfg-$rep.err | xargs) | $(grep 'peak memory footprint' $R/memory/load-$cfg-$rep.err | xargs)"
  done
done
echo "== 3b. which ranges are copied (INTARSIA_METAL_DEBUG)"
INTARSIA_METAL_DEBUG=1 INTARSIA_METAL_REALIGN=1 $BIN generate --model $M --backend metal --max-seq 4096 --profile-every 0 --temperature 0 --max-tokens 1 --prompt hello > $R/memory/debug-on.out 2> $R/memory/debug-on.err
grep -c "realign " $R/memory/debug-on.err; grep "realign " $R/memory/debug-on.err | awk '{n++; b+=$3} END {print n " ranges, " b " bytes"}'
echo "== 4. accuracy (golden gates, baseline-gated) with ON, and BASE without gate for the same numbers"
H0=c556008
run() { local n=$1; shift; "$@" > $n.out 2> $n.err; echo "$n exit $?"; }
export INTARSIA_METAL_REALIGN=1
run $R/accuracy/short-on $BIN accuracy --model $M --backend metal --golden $G/qwen3-8b-fp32.safetensors --gate-class bf16 --max-seq 4096 --mode both --baseline $SB/qwen3-8b-fp32.metal.baseline.json --noise-envelope $SB/qwen3-8b-fp32.metal.envelope.json --allow-same-code --pre-change $H0
run $R/accuracy/long-on $BIN accuracy --model $M --backend metal --golden $G/qwen3-8b-long-fp32.safetensors --gate-class bf16 --max-seq 8192 --mode both --decode-positions 64 --baseline $SB/qwen3-8b-long-fp32.metal.baseline.json --noise-envelope $SB/qwen3-8b-long-fp32.metal.envelope.json --allow-same-code --pre-change $H0
run $R/accuracy/4k-on $BIN accuracy --model $M --backend metal --golden $G4 --max-seq 8192 --mode all --decode-positions 64 --min-top1 0 --baseline $SB/qwen3-8b-4k-fp32-top20.metal.baseline.json --noise-envelope $SB/qwen3-8b-4k-fp32-top20.metal.envelope.json --allow-same-code --pre-change $H0
export INTARSIA_METAL_REALIGN=0
run $R/accuracy/short-base $BASE accuracy --model $M --backend metal --golden $G/qwen3-8b-fp32.safetensors --gate-class bf16 --max-seq 4096 --mode both --baseline $SB/qwen3-8b-fp32.metal.baseline.json --noise-envelope $SB/qwen3-8b-fp32.metal.envelope.json --allow-same-code --pre-change $H0
run $R/accuracy/long-base $BASE accuracy --model $M --backend metal --golden $G/qwen3-8b-long-fp32.safetensors --gate-class bf16 --max-seq 8192 --mode both --decode-positions 64 --baseline $SB/qwen3-8b-long-fp32.metal.baseline.json --noise-envelope $SB/qwen3-8b-long-fp32.metal.envelope.json --allow-same-code --pre-change $H0
run $R/accuracy/4k-base $BASE accuracy --model $M --backend metal --golden $G4 --max-seq 8192 --mode all --decode-positions 64 --min-top1 0 --baseline $SB/qwen3-8b-4k-fp32-top20.metal.baseline.json --noise-envelope $SB/qwen3-8b-4k-fp32-top20.metal.envelope.json --allow-same-code --pre-change $H0
grep -h "verdict" $R/accuracy/*.out | cut -c1-250
echo "end $(date)"

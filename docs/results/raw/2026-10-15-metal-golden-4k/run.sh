#!/bin/zsh
# Metal 4k golden: accuracy (mode all), baselines/envelopes, qualify, on SAFE (metal-tail-fix + qualify 4k wiring, chunk 256) and
# P3 (that + metal-p3-chunk, chunk 2048). One exclusive fleetq slot.
S=/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/g4k
W=/opt/code/intarsiaml-golden-4k
R=$W/docs/results/raw/2026-10-15-metal-golden-4k
A=$S/intarsia-a; P=$S/intarsia-p3
M=/Users/josef/models/Qwen3-8B
G4=$W/golden/qwen3-8b-4k-fp32-top20.safetensors
GD=$S/gold
export INTARSIA_CPU_FREE_MEM_GB=60
mkdir -p $R/safe $R/p3 $W/golden/baselines
J=$(fleetq acquire --host endor --exclusive --mem 24 --expect 60 --label "metal golden-4k: accuracy, baselines, qualify (SAFE+tailfix vs P3)" --wait --json | python3 -c 'import sys,json; print(json.load(sys.stdin)["id"])')
echo "fleetq job $J" | tee $R/fleetq-job.txt
trap "fleetq release $J" EXIT
export FLEETQ_JOB=$J
cd $W
HA=$(strings $A | grep -o 'head=[0-9a-f]* code-tree=[0-9a-f]* dirty=[a-z]*' | head -1); HP=$(strings $P | grep -o 'head=[0-9a-f]* code-tree=[0-9a-f]* dirty=[a-z]*' | head -1)
echo "SAFE stamp: $HA"; echo "P3 stamp: $HP"
AF=(--model $M --backend metal --golden $G4 --decode-positions 64 --max-seq 8192)
run() { local n=$1; shift; "$@" > $n.out 2> $n.err; echo "$n exit $?"; }
SB=$W/golden/baselines
echo "== 1. accuracy --mode all on the 4k golden"
run $R/safe/all-default $A accuracy $AF --mode all
run $R/safe/all-c2048 $A accuracy $AF --mode all --prefill-chunk 2048
run $R/p3/all-default $P accuracy $AF --mode all
run $R/p3/all-c256 $P accuracy $AF --mode all --prefill-chunk 256
for f in safe/all-default safe/all-c2048 p3/all-default p3/all-c256; do echo "-- $f"; grep -E "prefill chunk|floor verdict|prefix-reuse gate|reuse gate|golden gate" $R/$f.out | cut -c1-260; done
echo "== 2. baseline + envelope + selfcheck, SAFE at chunk 256 (perturb chunk=2048)"
BF=(--model $M --backend metal --golden $G4 --decode-positions 64 --max-seq 8192 --mode both)
run $R/safe/baseline $A accuracy $BF --write-baseline $SB/qwen3-8b-4k-fp32-top20.metal.baseline.json
run $R/safe/envelope $A accuracy $BF --noise-ref $SB/qwen3-8b-4k-fp32-top20.metal.baseline.json --noise-perturb chunk=2048 --write-noise-envelope $SB/qwen3-8b-4k-fp32-top20.metal.envelope.json --allow-same-code
run $R/safe/selfcheck $A accuracy $BF --baseline $SB/qwen3-8b-4k-fp32-top20.metal.baseline.json --noise-envelope $SB/qwen3-8b-4k-fp32-top20.metal.envelope.json --allow-same-code --pre-change $(echo $HA | sed 's/head=\([0-9a-f]*\).*/\1/')
grep -E "verdict" $R/safe/selfcheck.out | cut -c1-300
echo "== 3. same for P3 at chunk 2048 (perturb chunk=256), into raw dir (P3's golden/baselines is for its own merge)"
PB=$R/p3-baselines; mkdir -p $PB
run $R/p3/baseline $P accuracy $BF --write-baseline $PB/qwen3-8b-4k-fp32-top20.metal.baseline.json
run $R/p3/envelope $P accuracy $BF --noise-ref $PB/qwen3-8b-4k-fp32-top20.metal.baseline.json --noise-perturb chunk=256 --write-noise-envelope $PB/qwen3-8b-4k-fp32-top20.metal.envelope.json --allow-same-code
run $R/p3/selfcheck $P accuracy $BF --baseline $PB/qwen3-8b-4k-fp32-top20.metal.baseline.json --noise-envelope $PB/qwen3-8b-4k-fp32-top20.metal.envelope.json --allow-same-code --pre-change $(echo $HP | sed 's/head=\([0-9a-f]*\).*/\1/')
grep -E "verdict" $R/p3/selfcheck.out | cut -c1-300
echo "== 4. qualify (full, fresh), both builds"
$A qualify --model $M --backend metal --bandwidth-gbs 614 --golden-dir $GD --out $R/safe/qualify > $R/safe/qualify-console.log 2>&1; echo "safe qualify exit $?"
$P qualify --model $M --backend metal --bandwidth-gbs 614 --golden-dir $GD --out $R/p3/qualify > $R/p3/qualify-console.log 2>&1; echo "p3 qualify exit $?"
for b in safe p3; do echo "-- $b"; grep -iE "verdict|acc-|Accuracy" $R/$b/qualify/full-report.md | cut -c1-300; done
echo done

#!/bin/zsh
# slot G: final gates of the default path on the FINAL binary: cargo test (metal), kernels test, ids vs base, fresh qualify. One exclusive slot (slot.sh sets FLEETQ_JOB).
source /opt/code/intarsiaml-kvq/docs/results/raw/2026-10-15-metal-kv-quant/common.sh
NEW=/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/intarsia-final2
export INTARSIA_METAL_F8_U=$(pickU); echo "INTARSIA_METAL_F8_U=$INTARSIA_METAL_F8_U"
mkdir -p $R/gatesG $R/idsG
cd $W
{ echo "date: $(date -Iseconds)"; echo "fleetq job: $FLEETQ_JOB"; echo "head: $(git rev-parse HEAD)"; echo "F8_U=$INTARSIA_METAL_F8_U"; git status --short | head; ls -l $BASE $NEW; shasum -a 256 $BASE $NEW; } > $R/gatesG/versions.txt 2>&1

echo "== 0. KV memory sizing: tokens that fit a 40 GiB budget, bf16 vs fp8 (plan lines)"
mkdir -p $R/sizing
for kv in bf16 fp8; do
  $NEW generate --model $M --backend metal --mem-budget 40 --kv-dtype $kv --profile-every 0 --temperature 0 --max-tokens 1 --prompt "hi" < /dev/null > $R/sizing/$kv.out 2> $R/sizing/$kv.err
  echo "$kv: $(grep -iE 'kv' $R/sizing/$kv.err | head -4 | tr '\n' '|' | cut -c1-400)"
done
$NEW plan show --model $M --backend metal --mem-budget 40 --kv-dtype fp8 --json > $R/sizing/plan-fp8.json 2>&1; $NEW plan show --model $M --backend metal --mem-budget 40 --json > $R/sizing/plan-bf16.json 2>&1
grep -iE "kv" $R/sizing/plan-*.json | head -10 | cut -c1-220

echo "== 1. cargo test -p intarsia-backend-metal --release (all, incl. kv_fp8)"
cargo test -p intarsia-backend-metal --release -- --test-threads=1 --nocapture > $R/gatesG/cargo-test-metal.txt 2>&1; echo "exit $?"
grep -E "^test result|FAILED|panicked" $R/gatesG/cargo-test-metal.txt | sort | uniq -c | head -40

echo "== 2. kernels test --backend metal"
$NEW kernels test --backend metal --device 0 --model $M -v > $R/gatesG/kernels-test.txt 2>&1; echo "exit $?"
grep -cE "ADMITTED" $R/gatesG/kernels-test.txt; grep -E "FAIL|REJECT|NO CASES" $R/gatesG/kernels-test.txt | head -20
grep -E "f8" $R/gatesG/kernels-test.txt | head -30 | cut -c1-200

typeset -A FL PF
FL=(open-256 "--chat --no-think" grounded-256 "--chat --no-think" golden-docs-256 "--ignore-eos" golden-code-256 "--ignore-eos" 4k-256 "--ignore-eos" short-256 "--ignore-eos")
PF=(open-256 open grounded-256 grounded golden-docs-256 golden-docs golden-code-256 golden-code 4k-256 4k short-256 short)
echo "== 3. default-path greedy ids, base vs new (bf16 KV), 256 tokens"
for c in short-256 4k-256 open-256 grounded-256 golden-docs-256 golden-code-256; do
  for B in base new; do
    BIN=$NEW; [ $B = base ] && BIN=$BASE
    $BIN generate --model $M --backend metal --max-seq 8192 --profile-every 0 --temperature 0 --show-ids --max-tokens 256 ${=FL[$c]} --prompt "$(cat $PG/${PF[$c]}.txt)" < /dev/null > $R/idsG/$B-$c.out 2> $R/idsG/$B-$c.err
  done
  python3 -I $W/docs/results/raw/2026-10-14-metal-prefill-p1/full/first_divergence.py $R/idsG/base-$c.err $R/idsG/new-$c.err "bf16 KV, new vs base binary, $c" | tee -a $R/idsG/summary.txt
done
echo "== 6. qualify, fresh full run on the default (bf16 KV)"
mkdir -p $R/qualifyG-bf16
$NEW qualify --model $M --backend metal --bandwidth-gbs 614 --golden-dir $G --out $R/qualifyG-bf16 > $R/qualifyG-bf16-console.log 2>&1; echo "qualify bf16 exit $?"
grep -E "Verdict|verdict|PASS|FAIL" $R/qualifyG-bf16/full-report.md | head -20 | cut -c1-240
echo "done $(date)"

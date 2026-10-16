#!/bin/zsh
# slot M: gates on the MERGED tree (metal-kv-quant + spike/metal 912f3f8): cargo test (metal + models incl. batch identity with fp8 / int8 KV),
# kernels test, bf16 greedy ids vs the 912f3f8 binary, accuracy (default gate + fp8 / int8 sanity), qualify last.
source /opt/code/intarsiaml-kvq/docs/results/raw/2026-10-15-metal-kv-quant/common.sh
NEW=/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/intarsia-merged
BASE=/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/intarsia-base2
mkdir -p $R/gatesM $R/idsM $R/accM
cd $W
{ echo "date: $(date -Iseconds)"; echo "fleetq job: $FLEETQ_JOB"; echo "head: $(git rev-parse HEAD)"; git status --short | head -5; ls -l $BASE $NEW; shasum -a 256 $BASE $NEW; } > $R/gatesM/versions.txt 2>&1
echo "== 1. cargo test -p intarsia-backend-metal -p intarsia-models --release"
cargo test -p intarsia-backend-metal -p intarsia-models --release -- --test-threads=1 --nocapture > $R/gatesM/cargo-test.txt 2>&1; echo "exit $?"
grep -E "^test result|FAILED|panicked|bitwise|SKIP" $R/gatesM/cargo-test.txt | cut -c1-200 | sort | uniq -c | sort -rn | head -30
echo "== 2. kernels test --backend metal"
$NEW kernels test --backend metal --device 0 --model $M -v > $R/gatesM/kernels-test.txt 2>&1; echo "exit $?"
grep -cE "ADMITTED" $R/gatesM/kernels-test.txt; grep -E "FAIL|REJECT|NO CASES" $R/gatesM/kernels-test.txt | head -10
typeset -A FL PF
FL=(open-256 "--chat --no-think" grounded-256 "--chat --no-think" golden-docs-256 "--ignore-eos" golden-code-256 "--ignore-eos" 4k-256 "--ignore-eos" short-256 "--ignore-eos")
PF=(open-256 open grounded-256 grounded golden-docs-256 golden-docs golden-code-256 golden-code 4k-256 4k short-256 short)
echo "== 3. bf16 greedy ids, 912f3f8 binary vs merged"
for c in short-256 4k-256 open-256 grounded-256 golden-docs-256 golden-code-256; do
  for B in base new; do
    BIN=$NEW; [ $B = base ] && BIN=$BASE
    $BIN generate --model $M --backend metal --max-seq 8192 --profile-every 0 --temperature 0 --show-ids --max-tokens 256 ${=FL[$c]} --prompt "$(cat $PG/${PF[$c]}.txt)" < /dev/null > $R/idsM/$B-$c.out 2> $R/idsM/$B-$c.err
  done
  python3 -I $W/docs/results/raw/2026-10-14-metal-prefill-p1/full/first_divergence.py $R/idsM/base-$c.err $R/idsM/new-$c.err "bf16 KV, merged vs 912f3f8 binary, $c" | tee -a $R/idsM/summary.txt
done
echo "== 4. accuracy on the merged binary: default (bf16) gate with baselines, then fp8 / int8 short + int8 4k"
$NEW accuracy --model $M --backend metal --golden $G/qwen3-8b-fp32.safetensors --gate-class bf16 --max-seq 4096 > $R/accM/bf16-short.out 2> $R/accM/bf16-short.err; echo "bf16 short exit $?"
for KV in fp8 int8; do
  $NEW accuracy --model $M --backend metal --kv-dtype $KV --golden $G/qwen3-8b-fp32.safetensors --gate-class bf16 --max-seq 4096 > $R/accM/$KV-short.out 2> $R/accM/$KV-short.err; echo "$KV short exit $?"
done
$NEW accuracy --model $M --backend metal --kv-dtype int8 --golden $G/qwen3-8b-4k-fp32-top20.safetensors --gate-class bf16 --mode all --decode-positions 64 --max-seq 8192 > $R/accM/int8-4k.out 2> $R/accM/int8-4k.err; echo "int8 4k exit $?"
cp $R/acc_table.py $R/acc_table_M.py; sed -i '' 's#/acc"#/accM"#; s#/acc/#/accM/#' $R/acc_table_M.py; python3 $R/acc_table_M.py
echo "== 5. qualify, fresh full run on the default (bf16 KV), merged binary"
mkdir -p $R/qualifyM-bf16
$NEW qualify --model $M --backend metal --bandwidth-gbs 614 --golden-dir $G --out $R/qualifyM-bf16 > $R/qualifyM-bf16-console.log 2>&1; echo "qualify exit $?"
grep -E "Verdict" $R/qualifyM-bf16/full-report.md | head -3 | cut -c1-240
grep -E "^\| *[0-9]+ \|" $R/qualifyM-bf16/full-report.md | cut -c1-200
echo "done $(date)"

#!/bin/zsh
# slot A: gates of the default path + fp8 accuracy + greedy ids + qualify. One exclusive slot (slot.sh sets FLEETQ_JOB).
source /opt/code/intarsiaml-kvq/docs/results/raw/2026-10-15-metal-kv-quant/common.sh
export INTARSIA_METAL_F8_U=$(pickU); echo "INTARSIA_METAL_F8_U=$INTARSIA_METAL_F8_U"
mkdir -p $R/gates $R/acc $R/ids
cd $W
{ echo "date: $(date -Iseconds)"; echo "fleetq job: $FLEETQ_JOB"; echo "head: $(git rev-parse HEAD)"; echo "F8_U=$INTARSIA_METAL_F8_U"; git status --short | head; ls -l $BASE $NEW; shasum -a 256 $BASE $NEW; } > $R/gates/versions.txt 2>&1

echo "== 1. cargo test -p intarsia-backend-metal --release (all, incl. kv_fp8)"
cargo test -p intarsia-backend-metal --release -- --test-threads=1 --nocapture > $R/gates/cargo-test-metal.txt 2>&1; echo "exit $?"
grep -E "^test result|FAILED|panicked" $R/gates/cargo-test-metal.txt | sort | uniq -c | head -40

echo "== 2. kernels test --backend metal"
$NEW kernels test --backend metal --device 0 --model $M -v > $R/gates/kernels-test.txt 2>&1; echo "exit $?"
grep -cE "ADMITTED" $R/gates/kernels-test.txt; grep -E "FAIL|REJECT|NO CASES" $R/gates/kernels-test.txt | head -20
grep -E "f8" $R/gates/kernels-test.txt | head -30 | cut -c1-200

typeset -A FL PF
FL=(open-256 "--chat --no-think" grounded-256 "--chat --no-think" golden-docs-256 "--ignore-eos" golden-code-256 "--ignore-eos" 4k-256 "--ignore-eos" short-256 "--ignore-eos")
PF=(open-256 open grounded-256 grounded golden-docs-256 golden-docs golden-code-256 golden-code 4k-256 4k short-256 short)
echo "== 3. default-path greedy ids, base vs new (bf16 KV), 256 tokens"
for c in short-256 4k-256 open-256 grounded-256 golden-docs-256 golden-code-256; do
  for B in base new; do
    BIN=$NEW; [ $B = base ] && BIN=$BASE
    $BIN generate --model $M --backend metal --max-seq 8192 --profile-every 0 --temperature 0 --show-ids --max-tokens 256 ${=FL[$c]} --prompt "$(cat $PG/${PF[$c]}.txt)" < /dev/null > $R/ids/$B-$c.out 2> $R/ids/$B-$c.err
  done
  python3 -I $W/docs/results/raw/2026-10-14-metal-prefill-p1/full/first_divergence.py $R/ids/base-$c.err $R/ids/new-$c.err "bf16 KV, new vs base binary, $c" | tee -a $R/ids/summary.txt
done
echo "== 4. fp8 KV vs bf16 KV greedy ids (256 tokens), spec off and spec ngram vs off under fp8"
for c in short-256 4k-256 open-256 grounded-256 golden-docs-256 golden-code-256; do
  $NEW generate --model $M --backend metal --max-seq 8192 --profile-every 0 --temperature 0 --show-ids --max-tokens 256 ${=FL[$c]} --kv-dtype fp8 --prompt "$(cat $PG/${PF[$c]}.txt)" < /dev/null > $R/ids/fp8-$c.out 2> $R/ids/fp8-$c.err
  python3 -I $W/docs/results/raw/2026-10-14-metal-prefill-p1/full/first_divergence.py $R/ids/new-$c.err $R/ids/fp8-$c.err "fp8 KV vs bf16 KV, $c" | tee -a $R/ids/summary.txt
  $NEW generate --model $M --backend metal --max-seq 8192 --profile-every 0 --temperature 0 --show-ids --max-tokens 256 ${=FL[$c]} --kv-dtype fp8 --spec ngram --prompt "$(cat $PG/${PF[$c]}.txt)" < /dev/null > $R/ids/fp8spec-$c.out 2> $R/ids/fp8spec-$c.err
  python3 -I $W/docs/results/raw/2026-10-14-metal-prefill-p1/full/first_divergence.py $R/ids/fp8-$c.err $R/ids/fp8spec-$c.err "fp8 KV: spec ngram vs off, $c" | tee -a $R/ids/summary.txt
  grep -E "spec" $R/ids/fp8spec-$c.err | tail -2 | cut -c1-200
done

echo "== 5. fp8 accuracy vs fp32 goldens (bf16 gate class: KL 1e-3, top-1 99.5), and bf16 for reference"
for KV in bf16 fp8; do
  $NEW accuracy --model $M --backend metal --kv-dtype $KV --golden $G/qwen3-8b-fp32.safetensors --gate-class bf16 --max-seq 4096 > $R/acc/$KV-short.out 2> $R/acc/$KV-short.err; echo "$KV short exit $?"
  $NEW accuracy --model $M --backend metal --kv-dtype $KV --golden $G/qwen3-8b-long-fp32.safetensors --gate-class bf16 --mode all --max-seq 8192 > $R/acc/$KV-long.out 2> $R/acc/$KV-long.err; echo "$KV long exit $?"
  $NEW accuracy --model $M --backend metal --kv-dtype $KV --golden $G/qwen3-8b-4k-fp32-top20.safetensors --gate-class bf16 --mode all --decode-positions 64 --max-seq 8192 --baseline $W/golden/baselines/qwen3-8b-4k-fp32-top20.metal.baseline.json --noise-envelope $W/golden/baselines/qwen3-8b-4k-fp32-top20.metal.envelope.json --allow-config-mismatch > $R/acc/$KV-4k.out 2> $R/acc/$KV-4k.err; echo "$KV 4k exit $?"
done
for f in $R/acc/*.out; do echo "--- $f"; grep -E "floor|top-1|KL|PASS|FAIL|reuse" $f | head -12 | cut -c1-250; done
echo "== 6. qualify, fresh full run on the default (bf16 KV)"
mkdir -p $R/qualify-bf16 $R/qualify-fp8
$NEW qualify --model $M --backend metal --bandwidth-gbs 614 --golden-dir $G --out $R/qualify-bf16 > $R/qualify-bf16-console.log 2>&1; echo "qualify bf16 exit $?"
grep -E "Verdict|verdict|PASS|FAIL" $R/qualify-bf16/full-report.md | head -20 | cut -c1-240
echo "== 7. qualify acc stages on fp8 KV (skip bench stages)"
$NEW qualify --model $M --backend metal --kv-dtype fp8 --bandwidth-gbs 614 --golden-dir $G --out $R/qualify-fp8 --skip bench --skip bench-long --skip envelope --skip ab --skip tp > $R/qualify-fp8-console.log 2>&1; echo "qualify fp8 exit $?"
grep -E "Verdict|verdict|acc-|PASS|FAIL|SKIP" $R/qualify-fp8/full-report.md | head -30 | cut -c1-240
echo "done $(date)"

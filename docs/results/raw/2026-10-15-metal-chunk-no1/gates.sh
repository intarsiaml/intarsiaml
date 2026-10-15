#!/bin/zsh
# Gates for the "no 1-row prefill tail on Metal" change, run inside an already-acquired exclusive endor fleetq slot.
# base = spike/metal d4e6c32 (git archive, release, --features metal), after = metal-chunk-no1 932e3f4 (same build).
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a slot" >&2; exit 2; }
SC=/private/tmp/claude-501/-opt-code-intarsiaml/0e8026c5-6dd0-4b83-b375-5ff0a66894fc/scratchpad/cn1
W=/opt/code/intarsiaml-chunk-no1
R=$W/docs/results/raw/2026-10-15-metal-chunk-no1
M=/Users/josef/models/Qwen3-8B; G=/opt/code/intarsiaml/golden
BIN_base=$SC/intarsia-base; BIN_after=$SC/intarsia-after
export INTARSIA_CPU_FREE_MEM_GB=60
mkdir -p $R/identity $R/accuracy
cd $W
echo "== 0. prompt token counts"
for f in $R/prompts/*.txt; do
  $BIN_after generate --model $M --backend metal --max-seq 8192 --profile-every 0 --max-tokens 1 --prompt "$(cat $f)" 2>&1 >/dev/null | grep "^prompt:" | sed "s|^|$(basename $f) |"
done | tee $R/identity/token-counts.txt
echo "== 1. greedy ids, L x prompt x chunk x binary (single-chunk run = no tail reference)"
for L in 257 2049 4097; do for S in 1 2; do
  P=$R/prompts/p${L}s${S}.txt
  for C in 256 2048 $L; do for B in base after; do
    eval BIN=\$BIN_$B
    $BIN generate --model $M --backend metal --max-seq 8192 --profile-every 0 --temperature 0 --max-tokens 48 --ignore-eos --show-ids \
      --prefill-chunk $C --prompt "$(cat $P)" > $R/identity/$B-L$L-s$S-c$C.out 2> $R/identity/$B-L$L-s$S-c$C.err
    echo "$B L=$L s=$S chunk=$C exit $?"
  done; done
done; done
echo "== 2. accuracy (spec off), base vs after"
for B in base after; do
  eval BIN=\$BIN_$B
  $BIN accuracy --model $M --backend metal --spec off --golden $G/qwen3-8b-fp32.safetensors --mode both --decode-positions 16 --max-seq 4096 > $R/accuracy/$B-short.log 2>&1; echo "acc $B short exit $?"
  $BIN accuracy --model $M --backend metal --spec off --golden $G/qwen3-8b-long-fp32.safetensors --mode all --decode-positions 64 --max-seq 8192 > $R/accuracy/$B-long.log 2>&1; echo "acc $B long exit $?"
  $BIN accuracy --model $M --backend metal --spec off --golden $G/qwen3-8b-4k-fp32-top20.safetensors --mode all --decode-positions 64 --max-seq 8192 --min-top1 0 > $R/accuracy/$B-4k.log 2>&1; echo "acc $B 4k exit $?"
done
echo "== 3. qualify --backend metal (after)"
$BIN_after qualify --model $M --backend metal --golden-dir $G --out $R/qualify > $R/qualify-console.log 2>&1; echo "qualify exit $?"
echo "== 4. cargo test metal (release, workspace crate list)"
cargo test -p intarsia-backend-metal --release -- --test-threads=1 > $R/cargo-test-metal.txt 2>&1; echo "cargo test metal exit $?"
grep -E "^test result|FAILED|panicked" $R/cargo-test-metal.txt | sort | uniq -c
echo GATES_DONE

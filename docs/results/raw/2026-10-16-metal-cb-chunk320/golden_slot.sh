#!/bin/zsh
# Inside one exclusive endor slot: (1) serving-path invariance 320 vs 2048 (and 256 as the known-good control) with the
# chunkgolden harness, (2) Metal + workspace tests in release with the real model.
#   usage: golden_slot.sh OUTDIR
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a slot" >&2; exit 2; }
source ${0:A:h}/guard.sh
O=${1:?outdir}; mkdir -p $O; O=${O:A}
W=/opt/code/intarsiaml-chunk320; H=$W/docs/results/raw/2026-10-15-metal-chunk-golden
R=$W/docs/results/raw/2026-10-16-metal-cb-chunk320
M=$HOME/models/Qwen3-8B
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
export INTARSIA_CPU_FREE_MEM_GB=60
cd $W
echo "slot $FLEETQ_JOB $(date); git $(git rev-parse --short HEAD) harness $(shasum -a 256 $H/harness/target/release/chunkgolden | cut -c1-16)"
LENS=2,3,255,256,257,258,320,321,511,512,513,514,640,641,961,1000,2047,2048,2049,2050,3841,4095,4096,4097,4098,4200
trap "pkill -P \$\$ 2>/dev/null" EXIT
trap "exit 143" INT TERM
start_watchdog
for C in 2048 320 256; do
  past_soft && { echo "SKIP harness $C (past soft deadline)"; continue; }
  mkdir -p $O/dump/c$C
  $H/harness/target/release/chunkgolden --model $M --backend metal --max-seq 8192 --profile-every 0 --prefill-chunk $C \
    --docs tools/prompts/corpus_docs.txt --code tools/prompts/corpus_code.txt --golden $H/golden_ids.jsonl --lens $LENS --n 32 \
    --out $O/chunk$C.jsonl --dump-dir $O/dump/c$C > $O/chunk$C.log 2>&1
  echo "harness chunk=$C exit $? $(date +%T)"
done
python3 $R/compare_chunks.py $O/chunk2048.jsonl $O/chunk320.jsonl > $O/compare-320-vs-2048.md; tail -1 $O/compare-320-vs-2048.md
python3 $R/compare_chunks.py $O/chunk2048.jsonl $O/chunk256.jsonl > $O/compare-256-vs-2048.md; tail -1 $O/compare-256-vs-2048.md
python3 $R/compare_chunks.py $O/chunk256.jsonl $O/chunk320.jsonl > $O/compare-320-vs-256.md; tail -1 $O/compare-320-vs-256.md
rm -rf $O/dump
echo "== metal + workspace tests (release, real model)"
export INTARSIA_TEST_REAL_MODEL=1
past_soft || cargo test -p intarsia-backend-metal --release > $O/cargo-test-metal.log 2>&1; echo "cargo test -p intarsia-backend-metal --release: exit $?"
past_soft || cargo test --workspace --release --no-fail-fast > $O/cargo-test-workspace.log 2>&1; echo "cargo test --workspace --release: exit $?"
grep -E "^test result|FAILED|failed" $O/cargo-test-workspace.log | sort | uniq -c | head -20
echo "done $(date)"

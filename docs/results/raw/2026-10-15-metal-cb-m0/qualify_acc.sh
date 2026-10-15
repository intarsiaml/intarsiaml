#!/bin/zsh
# usage: qualify_acc.sh <outdir>   (inside an exclusive endor slot): fresh qualify on the stamped 8254311 binary, then
# intarsia accuracy (spec off) on the short, long and 4k goldens with the new binary and the 56ec2d5 binary
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a slot" >&2; exit 2; }
[ -d "$DEVELOPER_DIR" ] || { echo "DEVELOPER_DIR missing" >&2; exit 3; }
O=$1; mkdir -p $O
B=/private/tmp/claude-501/-opt-code-intarsiaml/0e8026c5-6dd0-4b83-b375-5ff0a66894fc/scratchpad/bin
NEW=$B/intarsia-cbm0-8254311; BASE=$B/intarsia-base-56ec2d5
M=/Users/josef/models/Qwen3-8B; G=/opt/code/intarsiaml/golden
cd /opt/code/intarsiaml-cb-m0
$NEW qualify --model $M --backend metal --golden-dir $G --out $O/qualify > $O/qualify-console.log 2>&1; echo "qualify exit $?"
for v in new base; do
  if [ $v = new ]; then BIN=$NEW; X=(--max-batch 1 --spec off); else BIN=$BASE; X=(--spec off); fi
  $BIN accuracy --model $M --backend metal $X --golden $G/qwen3-8b-fp32.safetensors --mode both --decode-positions 16 --max-seq 4096 > $O/acc-$v-short.log 2>&1; echo "acc $v short exit $?"
  $BIN accuracy --model $M --backend metal $X --golden $G/qwen3-8b-long-fp32.safetensors --mode all --decode-positions 64 --max-seq 8192 > $O/acc-$v-long.log 2>&1; echo "acc $v long exit $?"
  $BIN accuracy --model $M --backend metal $X --golden $G/qwen3-8b-4k-fp32-top20.safetensors --mode all --decode-positions 64 --max-seq 8192 --min-top1 0 > $O/acc-$v-4k.log 2>&1; echo "acc $v 4k exit $?"
done

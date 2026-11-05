#!/bin/zsh
# MTP on the branch binary: 256 greedy ids, --spec off vs --spec mtp k2 vs default, 35B at exposed prompt lengths.
set -u
[ -n "${FLEETQ_JOB:-}" ] || exit 2
cd /opt/code/intarsiaml/.worktrees/m35-peel
O=$PWD/.scratch/logs/gates; mkdir -p $O/mtp
M=/Users/josef/models/Qwen3.6-35B-A3B-NVFP4
BIN=$PWD/.scratch/intarsia.peel
for key in ${=1}; do
  python3 -c "import json,sys; sys.stdout.write(json.load(open('.scratch/logs/gates/prompts-text35.json'))['$key'])" > $O/mtp/p$key.txt
  for mode in off k2 default; do
    case $mode in off) A=(--spec off);; k2) A=(--spec mtp --spec-k 2);; default) A=();; esac
    $BIN generate --model $M --backend metal --max-seq 8192 --temperature 0 --max-tokens 256 --ignore-eos --profile-every 0 --show-ids $A --prompt "$(cat $O/mtp/p$key.txt)" > $O/mtp/$key-$mode.out 2> $O/mtp/$key-$mode.err
    echo "generate $key $mode exit $?"
  done
  cmp -s $O/mtp/$key-off.out $O/mtp/$key-k2.out && echo "mtp $key: k2 == off" || echo "mtp $key: k2 DIFFERS"
  cmp -s $O/mtp/$key-off.out $O/mtp/$key-default.out && echo "mtp $key: default == off" || echo "mtp $key: default DIFFERS"
done

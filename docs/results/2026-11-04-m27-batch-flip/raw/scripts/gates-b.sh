#!/bin/zsh
# 256 ids: main vs peel on non-exposed lengths, solo == behind on exposed lengths; then MTP k2 == off on the peel binary at an exposed length.
set -u
[ -n "${FLEETQ_JOB:-}" ] || exit 2
cd /opt/code/intarsiaml/.worktrees/m27-flip
O=$PWD/.scratch/logs/gates; mkdir -p $O/mtp
M=/Users/josef/models/Qwen3.8-27B-Quark-AWQ-MXFP4; PORT=18478
LENS=48,108,233,465,772,1272,2055,2861,4099,5770,258,259,260,262,265,266,273,320,516,1028,1088,1412,1540
for t in main peel; do
  BIN=$PWD/.scratch/intarsia.$t
  INTARSIA_CB_TRACE=1 $BIN serve --backend metal --model $M --served-model-name intarsia --max-seq 12000 --max-batch 2 --spec off --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port $PORT > $O/ids-$t-server.out 2> $O/ids-$t-server.err < /dev/null &
  SP=$!
  trap 'kill $SP 2>/dev/null; sleep 2; kill -9 $SP 2>/dev/null' EXIT
  up=0; for i in {1..240}; do curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { up=1; break; }; kill -0 $SP 2>/dev/null || break; sleep 1; done
  [ $up = 1 ] || { echo "SERVER FAILED $t"; exit 1; }
  python3 .scratch/ids256.py $PORT $O/ids.json $t $LENS; echo "ids $t exit $?"
  kill $SP 2>/dev/null; sleep 2; kill -9 $SP 2>/dev/null; wait $SP 2>/dev/null
done
trap - EXIT
BIN=$PWD/.scratch/intarsia.peel
for key in 772 1028; do
  python3 -c "import json,sys; sys.stdout.write(json.load(open('.scratch/logs/gates/prompts-text.json'))['$key'])" > $O/mtp/p$key.txt
  for mode in off k2 default; do
    case $mode in off) A=(--spec off);; k2) A=(--spec mtp --spec-k 2 --mtp-draft-vocab 65536);; default) A=();; esac
    $BIN generate --model $M --backend metal --max-seq 4096 --temperature 0 --max-tokens 256 --ignore-eos --profile-every 0 --show-ids $A --prompt "$(cat $O/mtp/p$key.txt)" > $O/mtp/$key-$mode.out 2> $O/mtp/$key-$mode.err
    echo "generate $key $mode exit $?"
  done
  cmp -s $O/mtp/$key-off.out $O/mtp/$key-k2.out && echo "mtp $key: k2 == off" || echo "mtp $key: k2 DIFFERS"
  cmp -s $O/mtp/$key-off.out $O/mtp/$key-default.out && echo "mtp $key: default == off" || echo "mtp $key: default DIFFERS"
done

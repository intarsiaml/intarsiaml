#!/bin/zsh
# merged head: greedy ids gate (off / default / k2 arms) + no-flag combination checks. Inside a fleetq slot.
[ -n "${FLEETQ_JOB:-}" ] || exit 2
W=/opt/code/intarsiaml/.worktrees/flip-metal-mtp; BIN=$W/.scratch/intarsia.merged; O=$W/.scratch/logs/comb; mkdir -p $O
M=/Users/josef/models/Qwen3.6-35B-A3B-NVFP4
PS="Write a short paragraph about the history of the printing press."
echo "#### ids (merged)"; BIN=$BIN OUT=ids-merged zsh $W/.scratch/ids.sh 2>&1 | tee $W/.scratch/logs/ids-merged.summary | tail -3
echo "#### generate, no flags, T0.7"
$BIN generate --backend metal --model $M --max-seq 4096 --temperature 0.7 --seed 3 --max-tokens 96 --prompt "$PS" > $O/gen-sampled.out 2> $O/gen-sampled.err
grep -a -E "default of recipe|sampled|^spec:" $O/gen-sampled.err | cut -c1-260
PORT=18493
for arm in b1 b4; do
  XA=(); [ $arm = b4 ] && XA=(--max-batch 4)
  echo "#### serve $arm, no --spec"
  $BIN serve --backend metal --model $M --served-model-name intarsia --max-seq 8192 $XA --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port $PORT > $O/serve-$arm.out 2> $O/serve-$arm.err < /dev/null &
  SP=$!; for i in {1..300}; do curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; sleep 1; done
  grep -a -E "default of recipe|sampled|speculative|batched" $O/serve-$arm.err | cut -c1-260
  for T in 0 0.7; do
    curl -s -m 120 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d "{\"model\":\"intarsia\",\"temperature\":$T,\"seed\":3,\"max_tokens\":96,\"messages\":[{\"role\":\"user\",\"content\":\"$PS\"}]}" > $O/serve-$arm-T$T.json
    echo "T=$T: $(python3 -c "import json;d=json.load(open('$O/serve-$arm-T$T.json'));print(d['usage'],d['choices'][0]['message']['content'][:70].replace(chr(10),' '))")"
  done
  sleep 1; grep -a -E "spec|verify" $O/serve-$arm.err | tail -4 | cut -c1-240
  kill -INT $SP; sleep 3; kill $SP 2>/dev/null; wait $SP 2>/dev/null
done
echo "#### done"

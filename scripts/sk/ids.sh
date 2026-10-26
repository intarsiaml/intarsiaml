#!/usr/bin/env bash
# Lane sk: greedy ids of the 35B on spark2 for several arms and prompt sizes; the first arm is the reference.
#   fleetq run --host spark2 --mem 40 --expect 15 --label "sk: ids" -- scripts/sk/ids.sh <outname> <arm> ...      arm = name:tree[:ENV=V,ENV=V]
# Prompts: the first CHARS (default 3300 13600 54700 = ~1k 4k 16k tokens) chars of tools/prompts/corpus_docs.txt; 256 ids each (--chat --no-think --ignore-eos).
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
OUTN=$1; shift; ARMS=("$@"); HOST=${HOST:-spark2}
RAW=/opt/code/intarsiaml/.worktrees/sk2/docs/results/2026-10-25-sk2-serve-macro/raw/$OUTN; mkdir -p "$RAW"
for a in "${ARMS[@]}"; do
  IFS=: read -r name tree envs <<< "$a"; envs=${envs//,/ }
  for chars in ${CHARS:-3300 13600 54700}; do
    ssh -n -o BatchMode=yes "$HOST" "export FLEETQ_JOB=$FLEETQ_JOB; P=\$HOME/intarsiaml-sk-$tree; M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
      env $envs timeout 900 \$P/target/acc/release/intarsia generate -m \$M35 --backend cuda --mem-budget 36 --max-seq 20480 --chat --no-think --profile-every 0 --ignore-eos --show-ids -p \"\$(head -c $chars \$HOME/intarsiaml-sk-main/tools/prompts/corpus_docs.txt)\" --max-tokens ${NIDS:-256} < /dev/null" > "$RAW/$name-$chars.out" 2> "$RAW/$name-$chars.err"
    grep -a '^ids:' "$RAW/$name-$chars.err" > "$RAW/$name-$chars.ids"
    echo "$name $chars: $(grep -aE 'finish=' "$RAW/$name-$chars.err" | head -1 | cut -c1-140) ids=$(tr ',' '\n' < "$RAW/$name-$chars.ids" | wc -l) md5=$(md5 -q "$RAW/$name-$chars.ids" 2>/dev/null || md5sum < "$RAW/$name-$chars.ids" | cut -c1-8)"
  done
done
ref=${ARMS[0]%%:*}
for a in "${ARMS[@]:1}"; do n=${a%%:*}; for chars in ${CHARS:-3300 13600 54700}; do
  if cmp -s "$RAW/$ref-$chars.ids" "$RAW/$n-$chars.ids" && [[ -s "$RAW/$ref-$chars.ids" ]]; then echo "IDS $chars $ref vs $n: identical"; else echo "IDS $chars $ref vs $n: DIFFER"; fi
done; done

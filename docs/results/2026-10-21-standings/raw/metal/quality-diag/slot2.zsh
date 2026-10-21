#!/usr/bin/env zsh
O=/private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/scratchpad/qd
BIN=/opt/code/intarsiaml/.claude/worktrees/agent-a1dec08a3ccbd4d11/target/release/intarsia
M=/Users/josef/models/Qwen3-8B; G=/Users/josef/models/Qwen3-8B-GGUF/Qwen3-8B-BF16.gguf
export INTARSIA_CPU_FREE_MEM_GB=60
run() { l=$1 p=$2 h=$3 m=$4 st=$5; shift 5
  "$@" > $O/$l-$st-s2-server.out 2> $O/$l-$st-s2-server.err & P=$!
  for i in {1..180}; do curl -sf http://127.0.0.1:$p$h >/dev/null 2>&1 && break; sleep 1; done
  rm -f $O/dump-$l-$st.jsonl
  (cd /opt/code/deepeval && DUMP=$O/dump-$l-$st.jsonl uv run python $O/dump.py quality --url http://127.0.0.1:$p/v1 --model $m --bench gsm8k --limit 3 --thinking off --structured $st --label dump-$l-$st > $O/dump-$l-$st.out 2>&1)
  kill $P; sleep 3; kill -9 $P 2>/dev/null; }
INT=($BIN serve --model $M --backend metal --device 0 --host 127.0.0.1 --port 8093 --served-model-name intarsia-metal --max-seq 20480 --profile-every 0 --no-supervisor --no-ui)
run intarsia 8093 /health intarsia-metal json_schema $INT
run intarsia 8093 /health intarsia-metal prompt $INT
run mlx 8092 /v1/models $M json_schema /Users/josef/.local/bin/mlx_lm.server --model $M --port 8092 --chat-template-args '{"enable_thinking":false}' --prompt-cache-size 1
run llama 8091 /health $G json_schema /opt/homebrew/bin/llama-server -m $G -ngl 999 -fa on -c 20480 --cache-ram 0 --port 8091
echo done > $O/slot2.done

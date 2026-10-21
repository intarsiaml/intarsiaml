#!/usr/bin/env zsh
W=/opt/code/intarsiaml/.claude/worktrees/agent-a1dec08a3ccbd4d11
O=/private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/scratchpad/qd
cd $W
echo "head $(git rev-parse HEAD) $(date -Iseconds)" > $O/slot.log
cargo test --workspace > $O/cargo-test.log 2>&1; echo "cargo test exit $?" >> $O/slot.log
cargo build --release -p intarsia-cli --features metal > $O/build.log 2>&1; echo "build exit $?" >> $O/slot.log
BIN=$W/target/release/intarsia; ls -l $BIN >> $O/slot.log
OUT=$O/diag BIN=$BIN zsh docs/results/2026-10-21-standings/raw/metal/resume-quality-diag.sh > $O/diag-summary.log 2>&1; echo "diag exit $?" >> $O/slot.log
M=/Users/josef/models/Qwen3-8B; G=/Users/josef/models/Qwen3-8B-GGUF/Qwen3-8B-BF16.gguf
run() { # label port health mid cmd...
  l=$1 p=$2 h=$3 m=$4; shift 4
  "$@" > $O/$l-server.out 2> $O/$l-server.err & P=$!
  for i in {1..180}; do curl -sf http://127.0.0.1:$p$h >/dev/null 2>&1 && break; sleep 1; done
  (cd /opt/code/deepeval && uv run localeval quality --url http://127.0.0.1:$p/v1 --model $m --bench gsm8k --limit 10 --thinking off --label $l-gsm8k > $O/$l-gsm8k.out 2>&1)
  echo "$l gsm8k exit $?" >> $O/slot.log
  kill $P; sleep 3; kill -9 $P 2>/dev/null
}
run mlx 8092 /v1/models $M /Users/josef/.local/bin/mlx_lm.server --model $M --port 8092 --chat-template-args '{"enable_thinking":false}' --prompt-cache-size 1
run llama 8091 /health $G /opt/homebrew/bin/llama-server -m $G -ngl 999 -fa on -c 20480 --cache-ram 0 --port 8091
echo done >> $O/slot.log

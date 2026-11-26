#!/usr/bin/env bash
# From the Mac: copy the worktree (origin/main) to tatooine's lane directory, plus the three vLLM Inkling goldens (gitignored).
set -uo pipefail
cd /opt/code/intarsiaml/.worktrees/ink-hip-g2
rsync -a --delete --exclude /target --exclude /.git --exclude /.worktrees --exclude /logs --exclude /.scratch/bin ./ tatooine:intarsiaml-ink-hip-g2/ < /dev/null
G=/opt/code/intarsiaml/golden
rsync -L $G/inkling-2-mini-vllm-top20.safetensors tatooine:intarsiaml-ink-hip-g2/golden/inkling2-mini-vllm-short-top20.safetensors < /dev/null
rsync -L $G/inkling-2-mini-vllm-long-top20.safetensors tatooine:intarsiaml-ink-hip-g2/golden/inkling2-mini-vllm-long-top20.safetensors < /dev/null
rsync -L $G/inkling-2-mini-vllm-4k-top20.safetensors tatooine:intarsiaml-ink-hip-g2/golden/inkling2-mini-vllm-4k-top20.safetensors < /dev/null

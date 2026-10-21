#!/usr/bin/env bash
# M3: does vLLM reproduce its own golden rows? 4 light prompt_logprobs requests against spark1 guidance (stays up).
set -u
D=/private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/scratchpad/m3
G=/opt/code/intarsiaml/golden/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors
R="uv run -q --with safetensors --with numpy python $D/vllm_requery.py"
S=m3-$(date +%s)
$R $G $D/rq/salt1.safetensors chat_reuse_4k,long_docs_4k --salt $S-a
$R $G $D/rq/salt1b.safetensors chat_reuse_4k --salt $S-a
$R $G $D/rq/salt2.safetensors chat_reuse_4k --salt $S-b
$R $G $D/rq/nosalt.safetensors chat_reuse_4k

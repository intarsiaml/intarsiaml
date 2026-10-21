#!/usr/bin/env bash
# M3: vLLM run-to-run spread on the whole 35b-long set (and 35b-short), stored ids, salted (no prefix-cache reads).
set -u
D=/private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/scratchpad/m3
GL=/opt/code/intarsiaml/golden/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors
GS=/opt/code/intarsiaml/golden/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors
R="uv run -q --with safetensors --with numpy python $D/vllm_requery.py"
S=m3all-$(date +%s)
L=long_docs_4k,long_code_4k,chat_reuse_4k,long_docs_12k,long_code_12k
Q=count,story,code,science,chat_short,chat_system,chat_multiturn,chat_reasoning,long_prose_300,long_prose_700
$R $GL $D/rq/long-A.safetensors $L --salt $S-A
$R $GL $D/rq/long-B.safetensors $L --salt $S-B
$R $GS $D/rq/short-A.safetensors $Q --salt $S-sA
$R $GS $D/rq/short-B.safetensors $Q --salt $S-sB

#!/usr/bin/env bash
# ink-serve: copy the Inkling-2-Mini snapshot (text model only, no MTP / tiktoken) from this box (endor) to spark1 ~/intarsiaml-ink-serve/model. Run under:
#   fleetq run --host endor,spark1 --load net,disk --expect 15 --label "intarsiaml inkling serve: copy checkpoint" -- bash scripts/inkling2-spark-copy-model.sh
set -euo pipefail
S=$HOME/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/b85c3cfa615fc413cc9af3379f491b963f3b586a
ssh -n -o BatchMode=yes spark1 'mkdir -p ~/intarsiaml-ink-serve/model'
cd "$S"
rsync -aL --partial -e "ssh -o BatchMode=yes" config.json hf_quant_config.json generation_config.json tokenizer.json tokenizer_config.json special_tokens_map.json chat_template.jinja model.safetensors.index.json model-00001-of-00001.safetensors spark1:intarsiaml-ink-serve/model/
ssh -n -o BatchMode=yes spark1 'ls -laL ~/intarsiaml-ink-serve/model; df -h ~ | tail -1'

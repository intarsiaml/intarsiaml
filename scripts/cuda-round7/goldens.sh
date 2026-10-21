#!/usr/bin/env bash
# Round 7: copy spark1's two 35B vLLM goldens to DEST (default scratch dir given as $1) and print what the live vLLM is
# (version, cache config, container). No GPU work; vLLM stays up.
#   fleetq run --host spark1 --mem 0 --label "..." -- scripts/cuda-round7/goldens.sh <dest dir>
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
HOST=${HOST:-spark1}
DEST=${1:?dest dir}
mkdir -p "$DEST"
for f in qwen36-35b-a3b-nvfp4-vllm-top20.safetensors qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors; do
  rsync -a -e "ssh -o BatchMode=yes" --rsync-path="FLEETQ_JOB=$FLEETQ_JOB rsync" "$HOST:intarsiaml-golden/$f" "$DEST/"
done
ssh -o BatchMode=yes "$HOST" "FLEETQ_JOB=$FLEETQ_JOB bash -s" <<'EOF'
ls -la ~/intarsiaml-golden/ | head -30
echo "== vLLM version: $(curl -s localhost:8000/version)"
curl -s localhost:8000/metrics | grep -E '^vllm:cache_config_info' | head -2
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv
docker ps --format '{{.Names}} {{.Image}}'
EOF

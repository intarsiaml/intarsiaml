#!/usr/bin/env bash
# Read-only: what the vLLM `guidance` container on spark1 runs for the 35B (KV cache dtype, attention backend, split
# settings) and the 35B's attention dims from its config.json. No container is started or stopped.
#   fleetq run --host spark1 --mem 0 --label "round5: read vLLM config" --wait-timeout 60 -- scripts/cuda-round5/vllm-info.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
HOST=${HOST:-spark1}; C=${VLLM_CONTAINER:-missioncntrl-guidance}
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
rsh() { ssh -n -o BatchMode=yes "$HOST" "export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
echo "== container state / image / command"
rsh "docker inspect $C --format '{{.State.Status}} started={{.State.StartedAt}} image={{.Config.Image}}'; docker inspect $C --format '{{json .Config.Entrypoint}} {{json .Config.Cmd}} {{json .Args}}'"
echo "== container env (VLLM_* / FLASHINFER* / CUDA*, secrets filtered)"
rsh "docker inspect $C --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -E '^(VLLM_|FLASHINFER|CUDA_|TORCH_CUDA)' | grep -viE 'token|key|secret|pass'"
echo "== mounts"
rsh "docker inspect $C --format '{{range .Mounts}}{{.Source}} -> {{.Destination}}{{println}}{{end}}'"
echo "== startup log lines"
rsh "docker logs $C 2>&1 | grep -nE 'non-default args|kv_cache_dtype|Using .*[Bb]ackend|ttention backend|FlashInfer|FLASHINFER|FLASH_ATTN|FlashAttention|TRITON_ATTN|kv cache|KV cache|fp8|FP8|split|partition|cudagraph|CUDA graph|max_num_seqs|max_model_len|quantization|Initializing a V1|Maximum concurrency|version' | tail -80 | cut -c1-400"
echo "== vllm / flashinfer versions in the image (only if the container is running)"
rsh "docker exec $C python3 -c 'import vllm,torch; print(vllm.__version__, torch.__version__)' 2>&1 | tail -2; docker exec $C python3 -c 'import flashinfer; print(\"flashinfer\", flashinfer.__version__)' 2>&1 | tail -1" || true
echo "== 35B config.json / hf_quant_config.json"
rsh "python3 - <<'PY'
import json, os
d = '$M35'
c = json.load(open(os.path.join(d, 'config.json')))
tc = c.get('text_config', c)
for k in ['num_hidden_layers','full_attention_interval','num_attention_heads','num_key_value_heads','head_dim','hidden_size','max_position_embeddings']:
    print(k, tc.get(k))
lt = tc.get('layer_types')
if lt: print('layer_types', {t: lt.count(t) for t in set(lt)})
qc = c.get('quantization_config')
print('quantization_config', json.dumps(qc)[:700] if qc else None)
p = os.path.join(d, 'hf_quant_config.json')
if os.path.exists(p): print('hf_quant_config', json.dumps(json.load(open(p)))[:700])
PY"

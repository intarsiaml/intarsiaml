#!/usr/bin/env bash
# read-only probe of spark1: 35B quant config, tensor names, vLLM container startup log
set -u
: "${FLEETQ_JOB:?}"
L=$(dirname "$0")/probe1_remote.sh; cat > $L <<'EOS'
M=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
ls $M
python3 - <<'PY'
import json
M='/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5'
c=json.load(open(M+'/config.json'))
q=c.get('quantization_config',{})
print(json.dumps(q,indent=1)[:4000])
tc=c.get('text_config',c)
for k in ['num_hidden_layers','hidden_size','layer_types','full_attention_interval','num_attention_heads','num_key_value_heads','head_dim','linear_num_key_heads','linear_num_value_heads','linear_key_head_dim','linear_value_head_dim','moe_intermediate_size','shared_expert_intermediate_size','num_experts','num_experts_per_tok']:
  print(k, tc.get(k))
idx=json.load(open(M+'/model.safetensors.index.json'))['weight_map']
for k in sorted(idx):
  if ('layers.0.' in k or 'layers.3.' in k) and 'layers.30.' not in k and 'layers.10.' not in k:
    if 'experts.' in k and 'experts.0.' not in k: continue
    print(k)
PY
docker logs missioncntrl-guidance 2>&1 | grep -iE 'kernel|cutlass|scaled_mm|fp8|linear|quant|flashinfer|marlin|backend|nvfp4|modelopt' | head -80
docker inspect missioncntrl-guidance --format '{{.Config.Image}} {{json .Config.Cmd}} {{json .Args}} {{json .Config.Env}}' | head -c 3000
EOS
scp -q $L spark1:/tmp/probe1_remote_$FLEETQ_JOB.sh
ssh -n -o BatchMode=yes spark1 "export FLEETQ_JOB=$FLEETQ_JOB; bash /tmp/probe1_remote_$FLEETQ_JOB.sh; rm -f /tmp/probe1_remote_$FLEETQ_JOB.sh"
rm -f $L

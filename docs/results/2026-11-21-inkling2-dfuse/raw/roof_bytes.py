"""Bytes read per decode token of Inkling-2-Mini by roofline class, at native widths from the safetensors header (bf16 dense,
NVFP4 / MXFP8 experts as stored), like tools/inkling2_floor.py; P12 (lossless 12-bit, the tip's default for the dense
projections and the unembed) at 12/16 of bf16 plus its escape/aux stream (taken from the P12 bench's bytes when given).
usage: python3 -I roof_bytes.py [safetensors]"""
import sys, json, struct, math
fn = sys.argv[1] if len(sys.argv) > 1 else '/Users/josef/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/b85c3cfa615fc413cc9af3379f491b963f3b586a/model-00001-of-00001.safetensors'
f = open(fn, 'rb'); n = struct.unpack('<Q', f.read(8))[0]; h = json.loads(f.read(n)); h.pop('__metadata__')
B = lambda k: h[k]['data_offsets'][1] - h[k]['data_offsets'][0]
P = 'model.llm.layers.'
inproj = ['attn.wq_du.weight', 'attn.wr_du.weight', 'attn.gate_attn.weight', 'attn.attn_sink_proj.weight', 'attn.wk_dv.weight', 'attn.wv_dv.weight']
visits = [(i, 0) for i in range(6)] + [(i, p) for p in range(3) for i in range(6, 18)] + [(i, 0) for i in range(18, 24)]
E, K = 128, 8
r = dict(in_proj=0, wo=0, experts_nv=0, experts_mx=0, router=0, ctrl=0, small=0)
for i, p in visits:
    r['in_proj'] += sum(B(P + f'{i}.{a}') for a in inproj)
    r['wo'] += B(P + f'{i}.attn.wo_ud.weight')
    r['router'] += B(P + f'{i}.mlp.gate.weight') + B(P + f'{i}.mlp.gate.bias')
    eb = sum(B(P + f'{i}.mlp.experts.{w}_weight') + B(P + f'{i}.mlp.experts.{w}_weight_scale') for w in ('w13', 'w2')) / E
    r['experts_mx' if i >= 20 else 'experts_nv'] += K * eb
    if p > 0:
        r['ctrl'] += sum(B(P + f'{i}.controllers.{c}.{w}_weight') for c in ('attn', 'k', 'v', 'mlp') for w in ('down', 'gain', 'forget'))
    for k in h:
        if k.startswith(P + f'{i}.') and ('sconv' in k or 'norm' in k or 'clip' in k or 'rel_logits' in k or 'controllers' in k and 'query' in k):
            r['small'] += B(k)
r['unembed'] = B('model.llm.unembed.weight') * 200058 / 201024
tot = sum(r.values())
for k, v in r.items():
    print(f"{k:12s} {v / 1e6:9.1f} MB/token native ({v / 614e9 * 1e3:6.3f} ms at 614 GB/s)")
print(f"{'total':12s} {tot / 1e6:9.1f} MB/token native ({tot / 614e9 * 1e3:6.3f} ms at 614 GB/s)")

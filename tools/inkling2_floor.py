#!/usr/bin/env python3
"""Byte floor and FLOPs per decode token of Inkling-2-Mini from the real safetensors header (spec section g). stdlib only:
python3 -I tools/inkling2_floor.py [path/to/model-00001-of-00001.safetensors]"""
import sys
import json,struct
fn=sys.argv[1] if len(sys.argv) > 1 else '/Users/josef/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/b85c3cfa615fc413cc9af3379f491b963f3b586a/model-00001-of-00001.safetensors'
f=open(fn,'rb'); n=struct.unpack('<Q',f.read(8))[0]; h=json.loads(f.read(n)); h.pop('__metadata__')
B=lambda k:(h[k]['data_offsets'][1]-h[k]['data_offsets'][0])
import math
def numel(k): return math.prod(h[k]['shape'])
P='model.llm.layers.'
attn_names=['attn.wq_du.weight','attn.wr_du.weight','attn.gate_attn.weight','attn.attn_sink_proj.weight','attn.wk_dv.weight','attn.wv_dv.weight','attn.wo_ud.weight']
small=['attn.q_l2_clip.scale','attn.rel_l2_clip.scale','attn.rel_logits_proj.proj','attn_sconv.weight','mlp_sconv.weight','mlp_norm.weight','attn_norm.weight','q_input_norm.weight','k_input_norm.weight','v_input_norm.weight','attn.k_sconv.weight','attn.v_sconv.weight','attn.k_sconv.static_w','attn.v_sconv.static_w','attn.k_sconv.weight_proj','attn.v_sconv.weight_proj','mlp.gate.bias']
visits=[(i,0) for i in range(6)]+[(i,p) for p in range(3) for i in range(6,18)]+[(i,0) for i in range(18,24)]
E=128;K=8
def expert_bytes(i):
    s=0
    for w in ('w13','w2'):
        k=P+f'{i}.mlp.experts.{w}_weight'; s+=B(k)+B(k+'_scale')
    return s/E
def tot(attn_bpp=None, unemb_bpp=None, mx_to_nv=False):
    r=dict(attn=0,experts=0,router=0,ctrl=0,small=0,unembed=0)
    for i,p in visits:
        for a in attn_names:
            k=P+f'{i}.{a}'; r['attn']+= B(k) if attn_bpp is None else numel(k)*attn_bpp
        r['router']+=B(P+f'{i}.mlp.gate.weight')
        eb=expert_bytes(i)
        if mx_to_nv and i>=20: eb=expert_bytes(0)
        r['experts']+=K*eb
        for s_ in small:
            k=P+f'{i}.{s_}'
            if k in h: r['small']+=B(k)
        if p>0:
            for c in ('attn','k','v','mlp'):
                for w in ('down','gain','forget'):
                    r['ctrl']+=B(P+f'{i}.controllers.{c}.{w}_weight')
    U='model.llm.unembed.weight'; real=200058/201024
    r['unembed']=B(U)*real if unemb_bpp is None else numel(U)*real*unemb_bpp
    r['total']=sum(r.values()); return r
GB=1e9
for name,kw in [('native',{}),('fp8 attn',dict(attn_bpp=1.0+1/128)),('fp8 attn+fp8 unembed',dict(attn_bpp=1+1/128,unemb_bpp=1+1/128)),('nvfp4 attn',dict(attn_bpp=0.5625)),('nvfp4 attn+unembed',dict(attn_bpp=0.5625,unemb_bpp=0.5625)),('native, MXFP8->NVFP4',dict(mx_to_nv=True)),('nvfp4 attn+unembed, MXFP8->NVFP4',dict(attn_bpp=0.5625,unemb_bpp=0.5625,mx_to_nv=True))]:
    r=tot(**kw); t=r['total']
    print(f"{name:36s} "+' '.join(f"{k}={v/1e6:.0f}MB" for k,v in r.items() if k!='total')+f"  TOTAL {t/GB:.3f} GB  M5Max {614e9/t:.0f} tok/s  GB10 {273e9/t:.0f} tok/s")
print('nvfp4 expert MB', expert_bytes(0)/1e6, 'mxfp8 expert MB', expert_bytes(20)/1e6)
print('attn per visit MB', sum(B(P+f'0.{a}') for a in attn_names)/1e6)
print('ctrl per looped visit MB', sum(B(P+f'6.controllers.{c}.{w}_weight') for c in ('attn','k','v','mlp') for w in ('down','gain','forget'))/1e6)
uniq=sum(B(k) for k in h if k.startswith('model.llm.')); print('unique llm bytes GB', uniq/GB, 'experts', sum(B(k) for k in h if '.experts.' in k)/GB)
# KV per token: per visit K+V; full visits = 8, SWA visits 40 (ring 1024)
kvtok=2*8*128*2
print('KV bytes/token/visit', kvtok)
for ctx in (4096,32768,131072):
    full=8*kvtok*ctx; swa=40*kvtok*min(ctx,1024)
    conv=48*(1024+1024+2304+2304)*3*2
    print(f'ctx {ctx}: KV full {full/1e6:.0f}MB swa {swa/1e6:.0f}MB conv {conv/1e6:.2f}MB total {(full+swa+conv)/1e9:.3f}GB ; decode KV read {full/1e6+swa/1e6:.0f}MB')
# FLOPs per decode token (2*MAC)
macs=0
for i,p in visits:
    macs+=sum(numel(P+f'{i}.{a}') for a in attn_names)+numel(P+f'{i}.mlp.gate.weight')+K*(2304*2304+2304*1152)
    if i in (5,11,17,23): macs+=2*32*2304
    if p>0: macs+=4*(64*4608+2*2304*64)
    macs+= 3*2304*5*2  # attn_res mixes rough
macs+=200058*2304
print('GFLOP/token weights', 2*macs/1e9)
for ctx in (4096,32768,131072):
    att=40*2*2*24*128*min(ctx,1024)+8*2*2*24*128*ctx
    rel=48*24*16*1024*2
    print(f'ctx {ctx} attention GFLOP {att/1e9+rel/1e9:.2f}')

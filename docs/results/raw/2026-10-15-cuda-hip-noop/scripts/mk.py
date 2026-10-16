import json, struct, sys, shutil
src="/Users/josef/models/Qwen3-8B"; out=sys.argv[1]
cfg=json.load(open(src+"/config.json")); cfg["num_hidden_layers"]=2; cfg["vocab_size"]=2048; cfg["max_window_layers"]=2
json.dump(cfg,open(out+"/config.json","w"),indent=1)
for f in ["tokenizer.json","tokenizer_config.json","generation_config.json","vocab.json","merges.txt"]: shutil.copy(f"{src}/{f}",out)
H=4096;I=12288;V=2048
T={"model.embed_tokens.weight":[V,H],"model.norm.weight":[H],"lm_head.weight":[V,H]}
for l in range(2):
    p=f"model.layers.{l}."
    T[p+"input_layernorm.weight"]=[H];T[p+"post_attention_layernorm.weight"]=[H]
    T[p+"self_attn.q_proj.weight"]=[4096,H];T[p+"self_attn.k_proj.weight"]=[1024,H];T[p+"self_attn.v_proj.weight"]=[1024,H];T[p+"self_attn.o_proj.weight"]=[H,4096]
    T[p+"self_attn.q_norm.weight"]=[128];T[p+"self_attn.k_norm.weight"]=[128]
    T[p+"mlp.gate_proj.weight"]=[I,H];T[p+"mlp.up_proj.weight"]=[I,H];T[p+"mlp.down_proj.weight"]=[H,I]
hdr={};off=0
for k,s in T.items():
    n=2
    for d in s:n*=d
    hdr[k]={"dtype":"BF16","shape":s,"data_offsets":[off,off+n]};off+=n
hb=json.dumps(hdr).encode(); hb+=b" "*((8-len(hb)%8)%8)
with open(out+"/model.safetensors","wb") as f:
    f.write(struct.pack("<Q",len(hb)));f.write(hb)
    z=bytes(1<<20)
    # small positive-ish bf16 pattern 0x3c00 (~0.0078)? use 0x3a80 repeated for non-degenerate values
    pat=bytes([0x80,0x3a])*(1<<19)
    left=off
    while left>0:
        n=min(left,len(pat));f.write(pat[:n]);left-=n
print(off)

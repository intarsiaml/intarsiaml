# roofline for Qwen3.6-35B-A3B NVFP4 decode on GB10; all weight sizes from the checkpoint headers (wbytes.txt)
MB=1e6
E_layer_expert = (3*5368.7+3*671.1)*MB/(40*256)        # bytes per expert (gate+up+down, nvfp4 + fp8 block scales)
dense = dict(gdn=(503.3+251.7+251.7+3.9+3.9+2.0)*MB, attn=(167.8+83.9+10.5+10.5)*MB, shared=(3*21.0+3*2.6)*MB,
             router=41.9*MB, head=(254.3+31.8)*MB)
D = sum(dense.values())
print('expert bytes MB', E_layer_expert/MB, 'dense total MB', D/MB, {k:round(v/MB,1) for k,v in dense.items()})
state_rw = 2*30*(98304+2097152)       # read + write of conv+S per sequence per step
print('state r+w per seq MB', state_rw/MB, 'one state MiB', 30*(98304+2097152)/2**20)
def kv(ctx): return ctx*20*1024        # 20 KiB/token read
def distinct(n,uniform=True): return 256*(1-(1-8/256)**n)
BW = None
def step_bytes(n, ctx, mode):
    ex = {'grouped':distinct(n), 'pair':8*n}[mode]*E_layer_expert*40
    return D + ex + n*(state_rw+kv(ctx))
b1 = step_bytes(1,1024,'grouped')
# anchor: graph-on step 12.406 ms (nsys, docs/results/2026-10-14-cuda-decode-fusion.md) -> effective bw
BW = b1/12.406e-3
print('N=1 bytes MB', b1/MB, 'effective GB/s', BW/1e9, 'peak 273 ->', BW/273e9)
# serve overhead: conc 1x 74.2 tok/s -> 13.477 ms/step vs 12.406 gpu
host = 1/74.2 - 12.406e-3
print('host ms/step at N=1', host*1e3)
for ctx in (1024, 4096):
    print('ctx', ctx)
    for mode in ('grouped','pair'):
        for n in (1,2,4,8,16):
            g = step_bytes(n,ctx,mode)/BW
            h = host + 0.05e-3*(n-1)
            st = g+h
            agg = n/st
            print(f' {mode:8s} N={n:2d} bytes={step_bytes(n,ctx,mode)/MB:7.0f} MB  gpu={g*1e3:6.1f} ms  step={st*1e3:6.1f} ms  agg={agg:6.1f} tok/s  x{agg/74.2:4.2f}   distinct/layer={distinct(n):.1f}')

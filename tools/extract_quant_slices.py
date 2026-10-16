# stdlib-only: slice real tensors out of the Qwen3.6-35B-A3B-NVFP4 checkpoint into one safetensors file.
import json, struct, glob, sys
d = glob.glob('/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/*/')[0]
idx = json.load(open(d + 'model.safetensors.index.json'))['weight_map']
hdrs = {}
def hdr(f):
    if f not in hdrs:
        with open(d + f, 'rb') as fh:
            n = struct.unpack('<Q', fh.read(8))[0]
            hdrs[f] = (json.loads(fh.read(n)), 8 + n)
    return hdrs[f]
ESZ = {'BF16': 2, 'F32': 4, 'F8_E4M3': 1, 'U8': 1}
def get(name, rows=None, cols=None):
    """rows=(a,b) cols=(a,b) in elements of the stored dtype; returns (dtype, shape, bytes)"""
    f = idx[name]
    h, base = hdr(f)
    t = h[name]
    dt, shape = t['dtype'], t['shape']
    b0, b1 = t['data_offsets']
    es = ESZ[dt]
    with open(d + f, 'rb') as fh:
        if len(shape) == 0:
            fh.seek(base + b0); return dt, [], fh.read(es)
        R, C = shape[0], (shape[1] if len(shape) > 1 else 1)
        r0, r1 = rows or (0, R)
        c0, c1 = cols or (0, C)
        out = bytearray()
        for r in range(r0, r1):
            fh.seek(base + b0 + (r * C + c0) * es)
            out += fh.read((c1 - c0) * es)
    return dt, ([r1 - r0, c1 - c0] if len(shape) > 1 else [r1 - r0]), bytes(out)
out = {}
def put(key, name, rows=None, cols=None):
    out[key] = get(name, rows, cols)
L = 'model.language_model.layers.%d.'
# --- FP8 linear slices (layer 0 linear-attn z projection, layer 3 attention o_proj)
for key, n, rows in (('fp8_z', L % 0 + 'linear_attn.in_proj_z', (0, 40)), ('fp8_o', L % 3 + 'self_attn.o_proj', (7, 31))):
    put(key + '.weight', n + '.weight', rows)
    put(key + '.weight_scale', n + '.weight_scale')
    put(key + '.input_scale', n + '.input_scale')
# --- NVFP4 linear slices: shared_expert gate_proj (K=2048), down_proj (K=512), lm_head (vocab rows)
for key, n, rows in (('nv_gate', L % 0 + 'mlp.shared_expert.gate_proj', (0, 24)),
                     ('nv_down', L % 0 + 'mlp.shared_expert.down_proj', (100, 130)),
                     ('nv_head', 'lm_head', (1000, 1040))):
    put(key + '.weight', n + '.weight', rows)
    put(key + '.weight_scale', n + '.weight_scale', rows)
    put(key + '.weight_scale_2', n + '.weight_scale_2')
    put(key + '.input_scale', n + '.input_scale')
# --- tiny MoE from real slices: E experts of layer 1, H'=256, I'=64 (valid sub-matrices: K slices keep 16-groups)
E, H, I = 8, 256, 64
mp = L % 1 + 'mlp.'
put('moe.router', mp + 'gate.weight', (0, E), (0, H))
put('moe.shared_gate', mp + 'shared_expert_gate.weight', None, (0, H))
for pre, base in [('moe.shared', mp + 'shared_expert.')] + [('moe.e%d' % e, mp + 'experts.%d.' % e) for e in range(E)]:
    for proj, rows, cols in (('gate_proj', (0, I), (0, H)), ('up_proj', (0, I), (0, H)), ('down_proj', (0, H), (0, I))):
        # packed columns are K/2 bytes, scale columns K/16
        K = cols[1]
        put('%s.%s.weight' % (pre, proj), base + proj + '.weight', rows, (0, K // 2))
        put('%s.%s.weight_scale' % (pre, proj), base + proj + '.weight_scale', rows, (0, K // 16))
        put('%s.%s.weight_scale_2' % (pre, proj), base + proj + '.weight_scale_2')
        put('%s.%s.input_scale' % (pre, proj), base + proj + '.input_scale')
# write safetensors
hd, off, blob = {}, 0, bytearray()
for k, (dt, shape, b) in out.items():
    hd[k] = {'dtype': dt, 'shape': shape, 'data_offsets': [off, off + len(b)]}
    blob += b; off += len(b)
hj = json.dumps(hd).encode()
hj += b' ' * ((8 - len(hj) % 8) % 8)
with open(sys.argv[1], 'wb') as f:
    f.write(struct.pack('<Q', len(hj))); f.write(hj); f.write(blob)
print(len(out), 'tensors', off, 'bytes')

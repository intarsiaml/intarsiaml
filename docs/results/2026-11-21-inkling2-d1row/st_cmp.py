"""Compare two safetensors files tensor by tensor (bytes), ignoring __metadata__. usage: st_cmp.py A B"""
import sys, json, struct
def load(p):
    b = open(p, 'rb').read()
    n = struct.unpack('<Q', b[:8])[0]
    h = json.loads(b[8:8 + n])
    meta = h.pop('__metadata__', {})
    return h, b[8 + n:], meta
ha, da, ma = load(sys.argv[1]); hb, db, mb = load(sys.argv[2])
same = diff = 0
for k in sorted(set(ha) | set(hb)):
    if k not in ha or k not in hb:
        print('only in one:', k); diff += 1; continue
    a0, a1 = ha[k]['data_offsets']; b0, b1 = hb[k]['data_offsets']
    if da[a0:a1] == db[b0:b1] and ha[k]['shape'] == hb[k]['shape']:
        same += 1
    else:
        diff += 1
        if diff <= 5: print('differs:', k, ha[k]['shape'])
print(f'{same} tensors identical, {diff} differ; metadata keys differing: {[k for k in set(ma)|set(mb) if ma.get(k)!=mb.get(k)][:6]}')

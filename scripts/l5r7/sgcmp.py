#!/usr/bin/env python3
"""Compare two safetensors files tensor by tensor, byte for byte (metadata such as the code stamp is ignored). usage: sgcmp.py A B
Exit 0 and print 'IDENTICAL n tensors' when every tensor (name, dtype, shape, bytes) is the same."""
import json, struct, sys

def load(p):
    with open(p, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        h = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for k, v in h.items():
            if k == '__metadata__':
                continue
            a, b = v['data_offsets']
            f.seek(base + a)
            out[k] = (v['dtype'], tuple(v['shape']), f.read(b - a))
        return out

a, b = load(sys.argv[1]), load(sys.argv[2])
bad = [k for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
if bad:
    print('DIFFER in %d of %d tensors, first: %s' % (len(bad), len(set(a) | set(b)), bad[:5]))
    sys.exit(1)
print('IDENTICAL %d tensors, %d bytes' % (len(a), sum(len(v[2]) for v in a.values())))

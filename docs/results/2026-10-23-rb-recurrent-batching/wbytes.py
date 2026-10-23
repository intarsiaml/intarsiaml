# Checkpoint bytes by tensor class, from the safetensors headers only (no tensor data is read).
# usage: python3 wbytes.py ~/Models/Qwen3.6-35B-A3B-NVFP4 > wbytes.txt
import json, struct, glob, re, collections, sys
root = sys.argv[1]
by = collections.Counter(); dt = collections.defaultdict(collections.Counter)
for f in sorted(glob.glob(root + '/model-*.safetensors')):
    fh = open(f, 'rb'); n = struct.unpack('<Q', fh.read(8))[0]; h = json.loads(fh.read(n))
    for k, v in h.items():
        if k == '__metadata__': continue
        a, b = v['data_offsets']
        k2 = re.sub(r'\.\d+\.', '.N.', k); k2 = re.sub(r'experts\.N\.', 'experts.E.', k2)
        k2 = re.sub(r'\.\d+\.', '.N.', k2); k2 = re.sub(r'\.\d+$', '', k2)
        by[k2] += b - a; dt[k2][v['dtype']] += 1
print('total GB', sum(by.values()) / 1e9)
for k, v in sorted(by.items(), key=lambda x: -x[1])[:60]:
    print(f'{v/1e6:10.1f} MB  {k}  {dict(dt[k])}')

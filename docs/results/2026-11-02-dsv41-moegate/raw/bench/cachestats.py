import json, struct, sys, glob, os, collections
d = sys.argv[1]
files = sorted(glob.glob(os.path.join(d, '*.safetensors')))
print('files', len(files), files[:3])
tot = collections.defaultdict(float)
rows = []
for f in files:
    with open(f, 'rb') as fh:
        n = struct.unpack('<Q', fh.read(8))[0]
        hdr = json.loads(fh.read(n))
        base = 8 + n
        for name, v in hdr.items():
            if name.endswith('ffn.experts.meta'):
                s, e = v['data_offsets']
                fh.seek(base + s)
                raw = fh.read(e - s)
                import array
                a = array.array('i'); a.frombytes(raw)
                E = v['shape'][0]
                layer = int(name.split('.')[1])
                # meta [E,8]: gate,up,down offsets, gate,up,down k2
                H, I = 5120, 1152
                gu = 0; dn = 0; hist = collections.Counter()
                for e_ in range(E):
                    k2g, k2u, k2d = a[e_*8+3], a[e_*8+4], a[e_*8+5]
                    gu += (H//16)*(I//16)*4*(k2g+k2u)*4
                    dn += (H//16)*(I//16)*4*k2d*4
                    hist[k2g] += 1
                rows.append((layer, E, gu, dn, dict(hist)))
rows.sort()
for r in rows: print(r)
if rows:
    print('layers', len(rows), 'mean gate+up bytes/layer %.3f GB, down %.3f GB' % (sum(r[2] for r in rows)/len(rows)/1e9, sum(r[3] for r in rows)/len(rows)/1e9))
    h = collections.Counter()
    for r in rows:
        for k, c in r[4].items(): h[k] += c
    print('k2 histogram (gate) over all layers', dict(sorted(h.items())))

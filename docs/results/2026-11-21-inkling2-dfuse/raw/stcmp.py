"""Compare two self-golden safetensors files by tensor bytes (the header metadata carries a build time and run metadata, so
the file md5 always differs). usage: stcmp.py A B -> 'EQUAL' or 'DIFFER: <tensors>'"""
import json, struct, sys
def load(f):
    b = open(f, "rb").read()
    n = struct.unpack("<Q", b[:8])[0]
    h = json.loads(b[8:8 + n])
    h.pop("__metadata__", None)
    return h, b[8 + n:]
(ha, da), (hb, db) = load(sys.argv[1]), load(sys.argv[2])
bad = [k for k in sorted(set(ha) | set(hb)) if k not in ha or k not in hb or ha[k]["shape"] != hb[k]["shape"]
       or da[slice(*ha[k]["data_offsets"])] != db[slice(*hb[k]["data_offsets"])]]
print("EQUAL" if not bad else "DIFFER: " + " ".join(bad[:20]))

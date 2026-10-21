# usage: python3 -I hf_headers.py <hf-repo> <out.json>   (HTTP range reads of safetensors headers only; no weight data is saved)
import json, struct, sys, urllib.request, collections, re
def get(url, rng=None):
    req = urllib.request.Request(url, headers={"Range": rng} if rng else {})
    return urllib.request.urlopen(req, timeout=60).read()
def headers(repo):
    base = f"https://huggingface.co/{repo}/resolve/main/"
    idx = json.loads(get(base+"model.safetensors.index.json"))
    shards = sorted(set(idx["weight_map"].values()))
    out = {}
    for s in shards:
        n = struct.unpack("<Q", get(base+s, "bytes=0-7"))[0]
        h = json.loads(get(base+s, f"bytes=8-{7+n}"))
        h.pop("__metadata__", None)
        out[s] = h
    return idx, out
if __name__ == "__main__":
    repo = sys.argv[1]
    idx, hs = headers(repo)
    flat = {}
    for s, h in hs.items():
        for k, v in h.items():
            flat[k] = dict(v, shard=s, nbytes=v["data_offsets"][1]-v["data_offsets"][0])
    json.dump(flat, open(sys.argv[2], "w"))
    print(len(flat), sum(v["nbytes"] for v in flat.values())/1e9, "GB;", idx.get("metadata"))

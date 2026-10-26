#!/usr/bin/env python3
"""Lane sk2: greedy token ids from a running `intarsia serve` (return_token_ids), alone per size, then all sizes at once.
usage: ids-serve.py <url> <outprefix> [chars,...]   -> <outprefix>-alone-<chars>.ids, <outprefix>-conc-<chars>.ids (one id per line)"""
import json, sys, threading, urllib.request, pathlib
url, out = sys.argv[1], sys.argv[2]
sizes = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "3300,13600,54700").split(",")]
corpus = pathlib.Path(__file__).resolve().parents[2].joinpath("tools/prompts/corpus_docs.txt").read_text()
def ask(chars):
    body = {"model": "intarsia-35b", "prompt": corpus[:chars], "max_tokens": 256, "temperature": 0, "ignore_eos": True, "return_token_ids": True}
    r = urllib.request.Request(url + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    j = json.load(urllib.request.urlopen(r, timeout=900))
    c = j["choices"][0]
    return c["token_ids"], j["usage"]["prompt_tokens"], c["finish_reason"]
def save(tag, chars, res):
    ids, pt, fin = res
    pathlib.Path(f"{out}-{tag}-{chars}.ids").write_text("\n".join(map(str, ids)) + "\n")
    print(f"{tag} {chars}: prompt={pt} ids={len(ids)} finish={fin}", flush=True)
for c in sizes:
    save("alone", c, ask(c))
res = {}
ts = [threading.Thread(target=lambda c=c: res.__setitem__(c, ask(c))) for c in sizes]
[t.start() for t in ts]; [t.join() for t in ts]
for c in sizes:
    save("conc", c, res[c])

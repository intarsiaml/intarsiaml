#!/usr/bin/env python3
"""Lane mfp8split (from mchunk4 / mb16): greedy token ids of 1 / 4 / 6 / 8 / 9 / 12 / 16 concurrent /v1/completions requests on a running `intarsia serve`.
Stream i's prompt is a slice of a long text (prompts(): even i about 100 tokens, odd i about 4100 tokens), raw completion, greedy,
ignore_eos, exactly --gen tokens.
  served_ids.py prompts --text prompt-16384.txt --dir D       # writes D/p<i>.txt (the 16 prompts, for the solo `generate` runs)
  served_ids.py run --url http://127.0.0.1:18675/v1 --text prompt-16384.txt --out arm.json [--gen 256]
  served_ids.py solo D/ids-p<i>.err ... --json solo.json      # collect `generate --show-ids` stderr into {"i": [ids]}
  served_ids.py vs-solo arm.json solo.json                    # every (streams, i) cell equal to stream i's solo generate ids
  served_ids.py compare a.json b.json                         # every (streams, i) cell equal between two arms"""
import argparse, json, re, sys, threading, urllib.request

CHARS = [430, 17800]
COUNTS = (1, 4, 6, 8, 9, 10, 12, 16)


def prompts(text):
    t = open(text).read()
    return [t[1500 * i: 1500 * i + CHARS[i % 2]] for i in range(16)]


def post(url, body):
    req = urllib.request.Request(url + "/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)


def run(url, model, text, out, gen, counts=None, short=False):
    ps = prompts(text)
    if short:  # every stream a ~100-token prompt (the route trace wants all streams decoding together)
        t = open(text).read()
        ps = [t[1500 * i: 1500 * i + CHARS[0]] for i in range(16)]
    res = {}
    for n in counts or COUNTS:
        got = [None] * n

        def one(i):
            ch = post(url, {"model": model, "prompt": ps[i], "max_tokens": gen, "ignore_eos": True, "temperature": 0.0, "return_token_ids": True})["choices"][0]
            got[i] = ch.get("token_ids") or ch["text"]

        th = [threading.Thread(target=one, args=(i,)) for i in range(n)]
        [x.start() for x in th]
        [x.join() for x in th]
        res[str(n)] = got
        print(f"streams {n}: {[len(g) if isinstance(g, list) else 'text' for g in got]}", flush=True)
    json.dump(res, open(out, "w"))


def solo(files, out):
    res = {}
    for f in files:
        i = int(re.search(r"ids-p(\d+)\.err$", f).group(1))
        line = [l for l in open(f, errors="replace") if l.startswith("ids")]
        res[str(i)] = [int(x) for x in re.findall(r"\d+", line[-1].split(":", 1)[1])] if line else None
    json.dump(res, open(out, "w"))
    print(f"solo: {len(res)} prompts, lengths {sorted(set(len(v) for v in res.values() if v))}")


def vs_solo(a, s):
    A, S = json.load(open(a)), json.load(open(s))
    rows = []
    allbad = []
    for n in A:
        bad = [i for i in range(len(A[n])) if A[n][i] != S.get(str(i))]
        allbad += [(n, i) for i in bad]
        rows.append(f"{n}:{len(A[n]) - len(bad)}/{len(A[n])}")
    cells = sum(len(v) for v in A.values())
    print(f"{a.split('/')[-1]} vs solo generate: {'SAME' if not allbad else 'DIFFER'} ({cells - len(allbad)}/{cells} cells; {' '.join(rows)}){'' if not allbad else ' ' + str(allbad)}")


def compare(a, b):
    A, B = json.load(open(a)), json.load(open(b))
    bad = [(n, i) for n in A for i in range(len(A[n])) if A[n][i] != B.get(n, [None] * 99)[i]]
    cells = sum(len(v) for v in A.values())
    print(f"{a.split('/')[-1]} vs {b.split('/')[-1]}: {'SAME' if not bad else 'DIFFER'} ({cells - len(bad)}/{cells} cells equal){'' if not bad else ' ' + str(bad)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("files", nargs="*")
    ap.add_argument("--url")
    ap.add_argument("--model", default="intarsia")
    ap.add_argument("--text")
    ap.add_argument("--dir")
    ap.add_argument("--out")
    ap.add_argument("--json")
    ap.add_argument("--gen", type=int, default=256)
    ap.add_argument("--short", action="store_true", help="all prompts ~100 tokens")
    ap.add_argument("--counts", help="comma list of stream counts, default 1,4,6,8,9,10,12,16")
    a = ap.parse_args()
    if a.cmd == "prompts":
        for i, p in enumerate(prompts(a.text)):
            open(f"{a.dir}/p{i}.txt", "w").write(p)
    elif a.cmd == "run":
        run(a.url, a.model, a.text, a.out, a.gen, [int(x) for x in a.counts.split(",")] if a.counts else None, a.short)
    elif a.cmd == "solo":
        solo(a.files, a.json)
    elif a.cmd == "vs-solo":
        vs_solo(*a.files)
    else:
        compare(*a.files)

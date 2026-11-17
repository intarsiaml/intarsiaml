#!/usr/bin/env python3
"""Lane sjf: served greedy token ids, each request alone vs the same requests contending (a short prompt arriving during a long
prefill), against one `intarsia serve --max-batch 4` started with INTARSIA_CB_NO_REUSE=1 (no idle-slot prefix reuse, so a repeated
prompt is cold both times). Writes {"solo": {case: ids}, "contend": {round: {case: ids}}, "reordered": n}.
  ident.py run --url http://host:port/v1 --metrics http://host:port/metrics --out ids.json
  ident.py compare ids.json [other.json ...]     every contend cell equals its solo cell (and every file's solo equals the first's)
"""
import argparse, json, re, sys, threading, time, urllib.request

HERE = __file__.rsplit("/", 1)[0]
CORPUS = ""


def corpus():
    global CORPUS
    if not CORPUS:
        base = HERE + "/../../tools/prompts/"
        CORPUS = open(base + "corpus_docs.txt").read() + open(base + "corpus_code.txt").read()
    return CORPUS


# case name -> characters of corpus (about 4 chars per token): ~100, ~300, ~1K, ~2.3K, ~6K, ~15K tokens
CASES = {"s100": 400, "s300": 1200, "m1k": 4000, "m2k": 9000, "l6k": 24000, "l15k": 60000}
GEN = 48


def prompt(name, off):
    n = CASES[name]
    c = corpus()
    return f"[{name}] " + c[off:off + n]


def post(url, model, text, gen=GEN):
    body = {"model": model, "prompt": text, "max_tokens": gen, "ignore_eos": True, "temperature": 0, "return_token_ids": True}
    req = urllib.request.Request(url + "/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        ch = json.load(r)["choices"][0]
    return ch.get("token_ids")


def metric(murl, name):
    try:
        t = urllib.request.urlopen(murl, timeout=10).read().decode()
        m = re.search(r"^" + name + r"\s+(\S+)", t, re.M)
        return float(m.group(1)) if m else None
    except Exception:
        return None


# contention rounds: (case, start delay s). The long prompt starts first, the shorter ones arrive while it prefills.
ROUNDS = [
    [("l15k", 0.0), ("s100", 0.4), ("m1k", 0.8), ("s300", 1.2)],
    [("l6k", 0.0), ("l15k", 0.1), ("s100", 0.6), ("m2k", 0.9)],
    [("m2k", 0.0), ("l15k", 0.05), ("s300", 0.5), ("s100", 0.7)],
    [("l15k", 0.0), ("l6k", 0.2), ("m1k", 0.3), ("m2k", 0.4)],
    # several short prompts arriving together behind a long prefill: under shortest they head the queue and pack into one pass
    [("l15k", 0.0), ("s100", 0.6), ("s300", 0.6), ("m1k", 0.6)],
    [("l6k", 0.0), ("m1k", 0.5), ("m2k", 0.5), ("s300", 0.5)],
    [("s100", 0.0), ("s300", 0.0), ("m1k", 0.0), ("m2k", 0.0)],
]


def run(a):
    out = {"solo": {}, "contend": {}}
    offs = {k: 1000 * (i + 1) for i, k in enumerate(CASES)}
    for name in CASES:
        out["solo"][name] = post(a.url, a.model, prompt(name, offs[name]))
        print("solo", name, len(out["solo"][name]), flush=True)
    r0 = metric(a.metrics, "intarsia_batch_prefill_reordered_total")
    k0 = metric(a.metrics, "intarsia_batch_pack_passes_total")
    for ri, rnd in enumerate(ROUNDS * a.repeat):
        res = {}

        def one(name, delay):
            time.sleep(delay)
            res[name] = post(a.url, a.model, prompt(name, offs[name]))

        ths = [threading.Thread(target=one, args=x) for x in rnd]
        [t.start() for t in ths]
        [t.join() for t in ths]
        out["contend"][str(ri)] = res
        print("contend round", ri, {k: len(v) for k, v in res.items()}, flush=True)
        time.sleep(1.0)
    r1 = metric(a.metrics, "intarsia_batch_prefill_reordered_total")
    k1 = metric(a.metrics, "intarsia_batch_pack_passes_total")
    out["reordered"] = None if r0 is None or r1 is None else r1 - r0
    out["packed_passes"] = None if k0 is None or k1 is None else k1 - k0
    print("reordered prefill decisions / packed passes during the contention rounds:", out["reordered"], out["packed_passes"])
    json.dump(out, open(a.out, "w"))


def compare(files):
    docs = [json.load(open(f)) for f in files]
    bad = cells = 0
    for f, d in zip(files, docs):
        for ri, res in d["contend"].items():
            for name, ids in res.items():
                cells += 1
                if ids != d["solo"][name]:
                    bad += 1
                    n = next((i for i, (x, y) in enumerate(zip(ids, d["solo"][name])) if x != y), None)
                    print(f"DIFF {f} round {ri} {name}: first difference at token {n}")
        print(f"{f}: reordered decisions {d.get('reordered')}, packed passes {d.get('packed_passes')}")
    for f, d in zip(files[1:], docs[1:]):
        for name, ids in d["solo"].items():
            cells += 1
            if ids != docs[0]["solo"][name]:
                bad += 1
                print(f"DIFF solo {name} between {files[0]} and {f}")
    print(f"{cells - bad}/{cells} cells identical")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--url", required=True)
    r.add_argument("--metrics", required=True)
    r.add_argument("--model", default="guidance")
    r.add_argument("--out", required=True)
    r.add_argument("--repeat", type=int, default=2)
    c = sp.add_parser("compare")
    c.add_argument("files", nargs="+")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else compare(a.files)

#!/usr/bin/env python3
"""Lane dsv41/tp2: are intarsia's greedy divergences from TensorFold near-ties?

usage: neartie.py TF.st INTARSIA.st BASE_URL MODEL [OUT.md] [--also-at N]

For every prompt whose greedy continuation differs, sends intarsia the prompt ids plus the common prefix up to the first
divergence as a raw completion (max_tokens 1, temperature 0, logprobs 20 (top 5 shown), via the server's legacy `logprobs` field) and
prints intarsia's top-5, TensorFold's token's rank and logprob, and the gap top-1 minus TensorFold's token. Also checks the
position itself: intarsia's top-1 must be intarsia's recorded token there (else the server's numerics changed since the
.st was recorded). Standard library only.
"""
import json
import struct
import sys
import urllib.request


def load(path):
    b = open(path, "rb").read()
    n = struct.unpack("<Q", b[:8])[0]
    h = json.loads(b[8 : 8 + n])
    data = b[8 + n :]
    names = [p["name"] for p in json.loads(h["__metadata__"]["prompts"])]

    def t(key):
        e = h[key]
        assert e["dtype"] in ("I32", "U32"), e["dtype"]
        lo, hi = e["data_offsets"]
        return list(struct.unpack(f"<{(hi - lo) // 4}i", data[lo:hi]))

    return {name: (t(f"ids_{i}"), t(f"cont_{i}")) for i, name in enumerate(names)}


def query(base, model, ids):
    body = {"model": model, "prompt": ids, "max_tokens": 1, "temperature": 0, "logprobs": 20, "return_token_ids": True}
    req = urllib.request.Request(base.rstrip("/") + "/completions", json.dumps(body).encode(), {"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def main():
    tf, ia, base, model = sys.argv[1:5]
    out = sys.argv[5] if len(sys.argv) > 5 and not sys.argv[5].startswith("--") else None
    A, B = load(tf), load(ia)
    rows = []
    for name, (ids_a, ca) in A.items():
        ids_b, cb = B[name]
        assert ids_a == ids_b, f"{name}: prompt ids differ"
        m = min(len(ca), len(cb))
        d = next((i for i in range(m) if ca[i] != cb[i]), None)
        if d is None:
            continue
        r = query(base, model, ids_a + ca[:d])
        lp = r["choices"][0]["logprobs"]
        top = [(int(e[0]), float(e[1])) for e in lp["top_logprobs_ids"][0]]
        toks = [e[2] if len(e) > 2 else "" for e in lp["top_logprobs_ids"][0]]
        got = r["choices"][0].get("token_ids", [None])[0]
        tf_tok, ia_tok = ca[d], cb[d]
        rank = next((k + 1 for k, (i, _) in enumerate(top) if i == tf_tok), None)
        tf_lp = dict(top).get(tf_tok)
        # asks for the top 20 (the server's cap) so TF's rank and logprob are known up to rank 20; the table shows the top 5
        gap = top[0][1] - tf_lp if tf_lp is not None else None
        rows.append(dict(name=name, pos=d, tf=tf_tok, ia=ia_tok, got=got, top=top, toks=toks, rank=rank, tf_lp=tf_lp, gap=gap,
                         same_as_recorded=(got == ia_tok)))
    lines = ["| prompt | pos | intarsia top-5 (id text: logprob) | TF token | TF rank | gap top-1 - TF (nats) | top-1 = recorded intarsia token |", "|---|---|---|---|---|---|---|"]
    for x in rows:
        tops = ", ".join(f"{i} {json.dumps(t)}: {l:.3f}" for (i, l), t in list(zip(x["top"], x["toks"]))[:5])
        gap = f"{x['gap']:.3f}" if x["gap"] is not None else f"> {x['top'][0][1] - x['top'][-1][1]:.2f} (outside top 20)"
        lines.append(f"| {x['name']} | {x['pos']} | {tops} | {x['tf']} | {x['rank'] or '> 20'} | {gap} | {'yes' if x['same_as_recorded'] else 'NO: ' + str(x['got'])} |")
    text = "\n".join(lines)
    print(text)
    if out:
        open(out, "w").write(text + "\n")
        json.dump(rows, open(out.rsplit(".", 1)[0] + ".json", "w"), indent=1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Lane sjf: prefill-order latency scenarios against one running `intarsia serve --max-batch 4`.
  lat.py pair   --url U            a ~40K-token prompt at t=0, a ~1K-token prompt at t=2 s: TTFT of each
  lat.py burst  --url U            4 long (~16K tokens) + 8 short (~1K) submitted 50 ms apart in a fixed interleaved order: TTFT p50 / p99, total
  lat.py decode --url U            4 concurrent streams of a ~100-token prompt, 256 tokens each: aggregate decode tok/s
Every prompt starts with a nonce, so no request shares a prefix with another (idle-slot reuse). Prints one JSON line per scenario run.
"""
import argparse, json, sys, threading, time, urllib.request, uuid

HERE = __file__.rsplit("/", 1)[0]
C = open(HERE + "/../../tools/prompts/corpus_docs.txt").read() + open(HERE + "/../../tools/prompts/corpus_code.txt").read()
CPT = 4.0  # characters per token, replaced by the calibration


def text(tag, nchars):
    body = (C * (nchars // len(C) + 1))[:nchars]
    return f"[{tag}-{uuid.uuid4().hex[:12]}] " + body


def stream(url, model, prompt, gen, t0, res, key):
    body = {"model": model, "prompt": prompt, "max_tokens": gen, "ignore_eos": True, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(url + "/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    first = None
    last = None
    n = 0
    ptok = None
    for line in urllib.request.urlopen(req, timeout=1800):
        if line.startswith(b"data: {"):
            d = json.loads(line[6:])
            if d.get("usage"):
                ptok = d["usage"].get("prompt_tokens")
            ch = d.get("choices") or []
            if ch and ch[0].get("text"):
                now = time.time()
                if first is None:
                    first = now
                last = now
                n += 1
    res[key] = {"ttft": (first - t0) if first else None, "end": (last - t0) if last else None, "tokens": n, "prompt_tokens": ptok,
                "t_first_abs": first, "t_last_abs": last}


def calibrate(url, model):
    r = {}
    t = time.time()
    stream(url, model, text("cal", 20000), 1, t, r, "x")
    return 20000 / r["x"]["prompt_tokens"]


def run_set(url, model, specs):
    """specs: list of (key, delay_s, prompt, gen). Returns {key: result}."""
    res = {}
    t0 = time.time()
    ths = []

    def go(k, d, p, g):
        time.sleep(d)
        stream(url, model, p, g, t0 + d, res, k)  # TTFT from this request's own submit

    for k, d, p, g in specs:
        th = threading.Thread(target=go, args=(k, d, p, g))
        th.start()
        ths.append(th)
    [t.join() for t in ths]
    return res, time.time() - t0


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", choices=["pair", "burst", "decode", "warm"])
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="guidance")
    ap.add_argument("--arm", default="")
    ap.add_argument("--long-tokens", type=int, default=40000)
    ap.add_argument("--delay", type=float, default=2.0)
    ap.add_argument("--burst-long", type=int, default=16000)
    a = ap.parse_args()
    cpt = calibrate(a.url, a.model)
    ch = lambda tok: int(tok * cpt)
    if a.scenario == "warm":
        return
    if a.scenario == "pair":
        res, wall = run_set(a.url, a.model, [("long", 0.0, text("L", ch(a.long_tokens)), 16), ("short", a.delay, text("S", ch(1000)), 16)])
        out = {"scn": "pair", "arm": a.arm, "cpt": round(cpt, 3), "wall": round(wall, 2),
               "short_ttft": res["short"]["ttft"], "long_ttft": res["long"]["ttft"], "short_prompt": res["short"]["prompt_tokens"], "long_prompt": res["long"]["prompt_tokens"],
               "short_end": res["short"]["end"], "long_end": res["long"]["end"]}
    elif a.scenario == "burst":
        order = "LSSLSSLSSLSS"  # fixed, interleaved: the first four slots take L S S L
        specs = []
        for i, k in enumerate(order):
            specs.append((f"{k}{i}", 0.05 * i, text(f"B{i}", ch(a.burst_long if k == "L" else 1000)), 16))
        res, wall = run_set(a.url, a.model, specs)
        allv = [r["ttft"] for r in res.values()]
        sh = [r["ttft"] for k, r in res.items() if k[0] == "S"]
        lo = [r["ttft"] for k, r in res.items() if k[0] == "L"]
        out = {"scn": "burst", "arm": a.arm, "cpt": round(cpt, 3), "total": round(wall, 2), "ttft_p50": round(pct(allv, 0.5), 2), "ttft_p99": round(pct(allv, 0.99), 2),
               "short_p50": round(pct(sh, 0.5), 2), "short_p99": round(pct(sh, 0.99), 2), "short_mean": round(sum(sh) / len(sh), 2),
               "long_p50": round(pct(lo, 0.5), 2), "long_max": round(max(lo), 2), "ttfts": {k: round(r["ttft"], 2) for k, r in sorted(res.items(), key=lambda kv: int(kv[0][1:]))}}
    else:
        specs = [(f"d{i}", 0.0, text(f"D{i}", ch(100)), 256) for i in range(4)]
        res, wall = run_set(a.url, a.model, specs)
        t_first = max(r["t_first_abs"] for r in res.values())  # all four decoding
        t_end = min(r["t_last_abs"] for r in res.values())      # until the first stream ends
        # tokens each stream produced in the window [last stream's first token, first stream's end]: approximate by (tokens-1) over its own window
        per = [(r["tokens"] - 1) / (r["t_last_abs"] - r["t_first_abs"]) for r in res.values()]
        agg = sum(r["tokens"] - 1 for r in res.values()) / (max(r["t_last_abs"] for r in res.values()) - min(r["t_first_abs"] for r in res.values()))
        out = {"scn": "decode", "arm": a.arm, "cpt": round(cpt, 3), "agg_tok_s": round(agg, 2), "per_stream_tok_s": [round(x, 2) for x in per], "wall": round(wall, 2)}
    print(json.dumps(out), flush=True)


main()

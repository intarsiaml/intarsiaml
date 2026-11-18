#!/usr/bin/env python3
"""flipev q4: inter-token latency of ongoing decodes while a 40K-token prompt prefills (stdlib only), `serve --max-batch 4`.

  q4.py solo --url U --out solo.json            each prompt alone: the reference ids
  q4.py run  --url U --arm A1 --solo solo.json --out q4.jsonl --d 4 --scn long|longshort [--rep 1]

Prompts are random token ids (seeded), so every request is cold (start the server with INTARSIA_CB_NO_REUSE=1). D decoders
(100-token prompts, --dec-tokens generated) start at t=0; the long prompt (--long-tokens ids, 16 tokens) is submitted at t=--long-at;
scn longshort adds a ~1K-token prompt at long+1.5 s and another at long+4 s (the second one is the one that finds the long prefill
mid-way again: under shortest each pauses the long prefill, which then resumes).
ITL = gap between two consecutive SSE events of one decoder (an event carries 1 or, with MTP, several tokens; same in both arms).
Window = [long submit, long first token]; base = [all decoders streaming + 0.5 s, long submit].
"""
import argparse, http.client, json, random, threading, time, urllib.parse

DEC_SEEDS = [101, 102, 103, 104]


def ids(seed, n):
    r = random.Random(seed * 7919 + 17)
    return [r.randrange(1000, 100000) for _ in range(n)]


def stream(url, model, prompt, gen, out):
    u = urllib.parse.urlparse(url)
    body = json.dumps({"model": model, "prompt": prompt, "max_tokens": gen, "temperature": 0, "ignore_eos": True, "stream": True,
                       "stream_options": {"include_usage": True}, "return_token_ids": True})
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=900)
    out.update(ev=[], ids=[], err=None, t_submit=time.perf_counter())
    try:
        c.request("POST", "/v1/completions", body, {"Content-Type": "application/json"})
        r = c.getresponse()
        if r.status != 200:
            out["err"] = f"HTTP {r.status}: {r.read()[:200]!r}"
            return
        buf = b""
        while True:
            chunk = r.read1(65536)
            if not chunk:
                break
            buf += chunk
            while b"\n\n" in buf:
                ev, buf = buf.split(b"\n\n", 1)
                for line in ev.split(b"\n"):
                    if not line.startswith(b"data:") or line[5:].strip() == b"[DONE]":
                        continue
                    j = json.loads(line[5:].strip())
                    for ch in j.get("choices", []):
                        t = ch.get("token_ids") or (ch.get("delta") or {}).get("token_ids")
                        if t:
                            out["ev"].append((time.perf_counter(), len(t)))
                            out["ids"].extend(t)
                    if j.get("usage"):
                        out["ptok"] = j["usage"].get("prompt_tokens")
    except Exception as e:  # noqa: BLE001
        out["err"] = repr(e)
    finally:
        c.close()


def prompts(a):
    P = {f"d{i}": (ids(DEC_SEEDS[i], 100), a.dec_tokens) for i in range(4)}
    P["long"] = (ids(201, a.long_tokens), 16)
    for i in range(1, 13):
        P[f"s{i}"] = (ids(300 + i, 1000), 16)
    return P


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))] if v else None


def stats(g):
    return {"n": len(g), "p50": pct(g, .5), "p99": pct(g, .99), "max": max(g) if g else None, "mean": sum(g) / len(g) if g else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["solo", "run"])
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="guidance")
    ap.add_argument("--arm", default="")
    ap.add_argument("--visit", default="")
    ap.add_argument("--d", type=int, default=4)
    ap.add_argument("--scn", default="long", choices=["long", "longshort", "longburst"])
    ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--long-tokens", type=int, default=40000)
    ap.add_argument("--long-at", type=float, default=2.0)
    ap.add_argument("--dec-tokens", type=int, default=1000)
    ap.add_argument("--solo")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    P = prompts(a)
    if a.cmd == "solo":
        ref = {}
        for k, (p, g) in P.items():
            o = {}
            stream(a.url, a.model, p, g, o)
            ref[k] = o["ids"]
            print(k, len(o["ids"]), o["err"], flush=True)
        json.dump(ref, open(a.out, "w"))
        return
    ref = json.load(open(a.solo)) if a.solo else {}
    keys = [f"d{i}" for i in range(a.d)] + ["long"] + (["s1", "s2"] if a.scn == "longshort" else [f"s{i}" for i in range(1, 13)] if a.scn == "longburst" else [])
    delay = {**{f"d{i}": 0.0 for i in range(a.d)}, "long": a.long_at, "s1": a.long_at + 1.5, "s2": a.long_at + 4.0, **{f"s{i}": a.long_at + 1.5 for i in range(3, 13)}}
    res = {k: {} for k in keys}
    t0 = time.perf_counter()

    def go(k):
        time.sleep(max(0, delay[k] - (time.perf_counter() - t0)))
        stream(a.url, a.model, P[k][0], P[k][1], res[k])

    ths = [threading.Thread(target=go, args=(k,)) for k in keys]
    [t.start() for t in ths]
    [t.join() for t in ths]
    L = res["long"]
    if L.get("err") or not L["ev"] or any(res[k].get("err") or not res[k]["ev"] for k in keys):
        print(json.dumps({"scn": a.scn, "arm": a.arm, "d": a.d, "rep": a.rep, "error": {k: res[k].get("err") for k in keys}}))
        return
    w0, w1 = L["t_submit"], L["ev"][0][0]
    b0 = max(res[f"d{i}"]["ev"][0][0] for i in range(a.d)) + 0.5
    gw, gb = [], []
    for i in range(a.d):
        e = [t for t, _ in res[f"d{i}"]["ev"]]
        for x, y in zip(e, e[1:]):
            if w0 <= y <= w1:
                gw.append(y - x)
            elif b0 <= y < w0:
                gb.append(y - x)
    # decode-event gaps that straddle nothing else: also the number of decoder tokens delivered in the window
    tok_w = sum(n for i in range(a.d) for t, n in res[f"d{i}"]["ev"] if w0 <= t <= w1)
    out = {"scn": a.scn, "arm": a.arm, "visit": a.visit, "d": a.d, "rep": a.rep, "t": time.strftime("%H:%M:%S"),
           "long_ttft": round(w1 - w0, 3), "long_prompt": L.get("ptok"),
           "short_ttft": {k: round(res[k]["ev"][0][0] - res[k]["t_submit"], 3) for k in sorted((k for k in res if k[0] == "s"), key=lambda k: int(k[1:]))},
           "window_s": round(w1 - w0, 3), "dec_alive_through_window": all(res[f"d{i}"]["ev"][-1][0] > w1 for i in range(a.d)), "dec_tokens_in_window": tok_w,
           "dec_tok_s_in_window": round(tok_w / (w1 - w0) / a.d, 2),
           "itl_window": stats(gw), "itl_base": stats(gb), "gaps_window": [round(x, 4) for x in gw], "gaps_base": [round(x, 4) for x in gb],
           "ids_n": {k: len(res[k]["ids"]) for k in keys},
           "ids_eq_solo": {k: (res[k]["ids"] == ref[k]) for k in keys} if ref else None,
           "ids": {k: res[k]["ids"] for k in keys}}
    open(a.out, "a").write(json.dumps(out) + "\n")
    print(json.dumps({k: out[k] for k in ("scn", "arm", "d", "rep", "long_ttft", "short_ttft", "dec_tok_s_in_window", "itl_window", "itl_base", "ids_n", "ids_eq_solo")}), flush=True)


main()

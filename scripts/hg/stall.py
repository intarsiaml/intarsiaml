#!/usr/bin/env python3
"""Lane hg: steady-state decode rate and the stall a new prompt causes (stdlib only, run from any machine).

  stall.py --url http://tatooine:8294/v1 --model intarsia --streams 16 --gen 512 --inject 1024,4096 [--out out.json]

Starts N streams (1k prompt, `--gen` tokens each, temperature 0, ignore_eos), waits until every stream has its first
token, then
  * steady: the aggregate token rate over the window from 0.5 s after the last first token to the first stream's end
    (prefill of the burst excluded, nothing else running) -- this is the number to put beside the in-process step test;
  * stall: injects one extra request per size in `--inject` (a fresh prompt of that many tokens, 8 generated tokens), one
    at a time, while the N streams decode. For each, the N streams' per-chunk inter-token latency during the injected
    request's prefill (injection to its first token) is reported: max, p99, mean, against the steady ITL p50/p99.
Chunk timestamps are the SSE `data:` lines' arrival times; a chunk holds one token here (the server streams per token).
"""
import argparse, http.client, json, random, statistics, sys, threading, time, urllib.parse

WORDS = ("kernel cache vector matrix token layer signal buffer thread socket packet router ledger invoice harbor glacier "
         "canyon meadow orchard lantern compass anchor falcon marble copper violet amber crimson silver quiet rapid gentle "
         "narrow distant ancient modern hollow measure gather report observe predict balance recover survey inspect compile").split()


def filler(n, seed):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        w = rng.sample(WORDS, 9)
        out.append(f"Note {i + 1}: the {w[0]} {w[1]} will {w[2]} the {w[3]} {w[4]} near the {w[5]} {w[6]} before {w[7]} {w[8]}.")
    return " ".join(out) + "\n\nContinue these notes with a long, detailed account."


def stream(url, model, prompt, gen, marks, info, nonce):
    u = urllib.parse.urlparse(url)
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=600)
    body = {"model": model, "messages": [{"role": "user", "content": f"[{nonce}] " + prompt}], "max_tokens": gen,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True}, "ignore_eos": True,
            "min_tokens": gen, "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.perf_counter()
    c.request("POST", u.path.rstrip("/") + "/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    info["t0"] = t0
    for line in r:
        now = time.perf_counter()
        if not line.startswith(b"data: ") or line.startswith(b"data: [DONE]"):
            continue
        d = json.loads(line[6:])
        if d.get("usage"):
            info["usage"] = d["usage"]
        ch = d.get("choices") or []
        if ch and (ch[0].get("delta", {}).get("content") or ch[0].get("delta", {}).get("reasoning_content")):
            marks.append(now)
    info["t_end"] = time.perf_counter()
    c.close()


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))] if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="intarsia")
    ap.add_argument("--streams", type=int, default=16)
    ap.add_argument("--gen", type=int, default=512)
    ap.add_argument("--prompt-tokens", type=int, default=1024)
    ap.add_argument("--inject", default="1024,4096")
    ap.add_argument("--out")
    a = ap.parse_args()

    def sentences_for(tokens):  # calibrate once against the server's own prompt_tokens
        m, inf = [], {}
        stream(a.url, a.model, filler(40, 1), 1, m, inf, "cal")
        per = (inf["usage"]["prompt_tokens"] - 20) / 40
        return max(1, round((tokens - 20) / per))

    # calibrate every prompt size BEFORE the streams start: with all slots busy a calibration call would queue behind them
    n_base = sentences_for(a.prompt_tokens)
    n_inj = {int(sz): sentences_for(int(sz)) for sz in a.inject.split(",") if sz}
    base = [(list(), {}) for _ in range(a.streams)]
    th = [threading.Thread(target=stream, args=(a.url, a.model, filler(n_base, 100 + i), a.gen, m, inf, f"b{i}-{time.time()}"))
          for i, (m, inf) in enumerate(base)]
    [t.start() for t in th]
    while not all(m for m, _ in base):
        time.sleep(0.05)
    t_first = max(m[0] for m, _ in base)
    time.sleep(max(0.0, t_first + 0.5 - time.perf_counter()))
    w0 = time.perf_counter()
    time.sleep(2.0)
    w1 = time.perf_counter()
    steady_tokens = sum(sum(1 for t in m if w0 <= t < w1) for m, _ in base)
    steady_itl = [(b - a_) * 1e3 for m, _ in base for a_, b in zip(m, m[1:]) if w0 <= b < w1]
    res = {"streams": a.streams, "gen": a.gen, "prompt_tokens": a.prompt_tokens,
           "steady_window_s": round(w1 - w0, 3), "steady_agg_tok_s": round(steady_tokens / (w1 - w0), 1),
           "steady_itl_ms": {"mean": round(statistics.mean(steady_itl), 2), "p50": round(pct(steady_itl, 50), 2),
                              "p99": round(pct(steady_itl, 99), 2), "max": round(max(steady_itl), 2)},
           "inject": []}
    for size in [int(s) for s in a.inject.split(",") if s]:
        n = n_inj[size]
        m, inf = [], {}
        t_inj = time.perf_counter()
        ti = threading.Thread(target=stream, args=(a.url, a.model, filler(n, 9000 + size), 8, m, inf, f"inj{size}-{time.time()}"))
        ti.start(); ti.join()
        t_first_tok = m[0] if m else time.perf_counter()
        itl = [(b - a_) * 1e3 for bm, _ in base for a_, b in zip(bm, bm[1:]) if t_inj <= b <= t_first_tok + 0.05]
        res["inject"].append({"prompt_tokens": inf.get("usage", {}).get("prompt_tokens"), "ttft_s": round(t_first_tok - t_inj, 3),
                              "base_itl_ms_during": {"n": len(itl), "mean": round(statistics.mean(itl), 2) if itl else None,
                                                     "p99": round(pct(itl, 99), 2) if itl else None,
                                                     "max": round(max(itl), 2) if itl else None}})
        time.sleep(1.0)
    [t.join() for t in th]
    json.dump(res, open(a.out, "w"), indent=1) if a.out else None
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

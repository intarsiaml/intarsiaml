#!/usr/bin/env python3
"""Lane dp: one load driver for both arms of the two-tp1-vs-one-tp2 test (stdlib only).

  load.py --urls http://tatooine:8301/v1,http://tatooine:8302/v1 --streams 1,8,16,32 --out out.json     (arm D: 2 servers)
  load.py --urls http://tatooine:8300/v1 --streams 1,8,16,32 --out out.json                            (arm T: 1 server)
  load.py --urls ... --ident --out ident.json                                                          (identity prompts)
  load.py --urls URL --ctx-probe 16000 --out ctx.json                                                  (long prompt)

N total streams are split evenly over the given servers (stream i -> server i % len(urls); N = 1 uses the first server only; any other
N must be a multiple of len(urls)). Every stream has its own ~1k-token prompt (`[N-burst-i] ` nonce + hg/stall.py's filler seeded by (N, i)),
so the prompt set is the same for every arm. All streams start behind one barrier. Greedy, `ignore_eos`, `min_tokens` = gen, no thinking.
Per N: one discarded warm-up burst (nonce `[warm...]`), then --bursts measured bursts.
Per burst, per server: aggregate tok/s = completion tokens / that server's wall (barrier release to its last stream's end), the same
definition as localeval's `aggregate_decode_tok_s` (prefill included in the wall). The arm's aggregate is the SUM over servers; the
combined figure (all tokens / the longest wall) is kept beside it. Per stream: TTFT, decode tok/s = (tokens - 1) / (end - first token),
sha256 of the streamed text (stand-in for the greedy ids).
"""
import argparse, hashlib, http.client, json, os, statistics, sys, threading, time, urllib.parse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hg"))
from stall import filler  # noqa: E402  (same prompt generator as lane fh's ttft.py and localeval's filler)


def chat_body(model, prompt, gen, stream):
    b = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": gen, "temperature": 0, "stream": stream,
         "ignore_eos": True, "min_tokens": gen, "chat_template_kwargs": {"enable_thinking": False}}
    if stream:
        b["stream_options"] = {"include_usage": True}
    return b


def one(url, model, prompt, gen, barrier, res, i, t0box):
    u = urllib.parse.urlparse(url)
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=900)
    body = json.dumps(chat_body(model, prompt, gen, True))
    try:
        c.connect()
    except Exception as e:      # still reach the barrier, or every other stream would wait for ever
        barrier.wait()
        res[i] = {"server": url, "error": repr(e), "end_s": 0.0}
        return
    barrier.wait()
    t0 = t0box[0]
    try:
        c.request("POST", u.path.rstrip("/") + "/chat/completions", body, {"Content-Type": "application/json"})
        r = c.getresponse()
        first, text, chunks, usage, status = None, [], 0, None, r.status
        for line in r:
            if line.startswith(b"data: {"):
                d = json.loads(line[6:])
                if d.get("usage"):
                    usage = d["usage"]
                ch = d.get("choices") or []
                piece = (ch[0].get("delta") or {}).get("content") if ch else None
                if piece:
                    if first is None:
                        first = time.perf_counter()
                    text.append(piece)
                    chunks += 1
        end = time.perf_counter()
        n = (usage or {}).get("completion_tokens") or gen
        res[i] = {"server": url, "status": status, "ttft_s": None if first is None else first - t0, "end_s": end - t0, "tokens": n, "chunks": chunks,
                  "prompt_tokens": (usage or {}).get("prompt_tokens"),
                  "decode_tok_s": None if first is None or end <= first else (n - 1) / (end - first),
                  "sha": hashlib.sha256("".join(text).encode()).hexdigest()[:16]}
    except Exception as e:  # a failed stream is recorded, not hidden
        res[i] = {"server": url, "error": repr(e), "end_s": time.perf_counter() - t0}


def burst(urls, model, jobs, gen):
    """jobs: list of (server_index, prompt). Returns the per-stream results and the per-server / combined aggregates."""
    n = len(jobs)
    res, t0box = [None] * n, [0.0]
    barrier = threading.Barrier(n + 1, action=lambda: t0box.__setitem__(0, time.perf_counter()))   # t0 is set once, before anyone is released
    ts = [threading.Thread(target=one, args=(urls[s], model, p, gen, barrier, res, i, t0box)) for i, (s, p) in enumerate(jobs)]
    [t.start() for t in ts]
    barrier.wait()
    [t.join() for t in ts]
    per = {}
    for r in res:
        d = per.setdefault(r["server"], {"tokens": 0, "wall": 0.0, "streams": 0, "errors": 0})
        d["streams"] += 1
        if "error" in r or r.get("status") != 200:
            d["errors"] += 1
            continue
        d["tokens"] += r["tokens"]
        d["wall"] = max(d["wall"], r["end_s"])
    agg = {s: (d["tokens"] / d["wall"] if d["wall"] else None) for s, d in per.items()}
    tot = sum(d["tokens"] for d in per.values())
    wall = max((d["wall"] for d in per.values()), default=0)
    return {"streams": res, "per_server": {s: {**per[s], "agg_tok_s": agg[s]} for s in per},
            "agg_sum_tok_s": sum(v for v in agg.values() if v), "agg_combined_tok_s": tot / wall if wall else None, "errors": sum(d["errors"] for d in per.values())}


def split(urls, n):
    if n == 1:
        return 1
    if n % len(urls):
        sys.exit(f"{n} streams do not split evenly over {len(urls)} servers")
    return len(urls)


def prompt_for(n, b, i, sentences):
    return f"[{n}-{b}-{i}] " + filler(sentences, 1000 * n + i)


def metrics(url):
    u = urllib.parse.urlparse(url)
    try:
        c = http.client.HTTPConnection(u.hostname, u.port, timeout=10)
        c.request("GET", "/metrics")
        t = c.getresponse().read().decode()
    except Exception:
        return {}
    out = {}
    for line in t.splitlines():
        for k in ("intarsia_batch_pack_passes_total", "intarsia_batch_packed_chunks_total"):
            if line.startswith(k + " "):
                out[k] = float(line.split()[1])
    return out


def sweep(a, urls):
    out = {"urls": urls, "gen": a.gen, "sentences": a.sentences, "bursts": a.bursts, "rows": []}
    for url in urls:   # warm every server once so no first-use setup lands in a measured burst
        burst([url], a.model, [(0, "[warm0] " + filler(a.sentences, 999))], 8)
    for n in [int(x) for x in a.streams.split(",")]:
        k = split(urls, n)
        use = urls[:k]
        m0 = [metrics(u) for u in use]
        jobs_of = lambda tag: [(i % k, (f"[warm{n}] " + filler(a.sentences, 1000 * n + i)) if tag == "warm" else prompt_for(n, tag, i, a.sentences)) for i in range(n)]
        if not a.no_warm:
            burst(use, a.model, jobs_of("warm"), a.gen)
            time.sleep(1)
        bs = []
        for b in range(a.bursts):
            bs.append(burst(use, a.model, jobs_of(b), a.gen))
            time.sleep(1)
        m1 = [metrics(u) for u in use]
        streams = [s for b in bs for s in b["streams"] if "error" not in s and s.get("ttft_s") is not None]
        row = {"streams": n, "servers": k, "bursts": bs,
               "agg_sum_median": statistics.median(b["agg_sum_tok_s"] for b in bs),
               "agg_combined_median": statistics.median(b["agg_combined_tok_s"] for b in bs),
               "pack_passes": sum(m1[j].get("intarsia_batch_pack_passes_total", 0) - m0[j].get("intarsia_batch_pack_passes_total", 0) for j in range(k)),
               "errors": sum(b["errors"] for b in bs)}
        tt = sorted(s["ttft_s"] for s in streams)
        pct = lambda p: tt[min(len(tt) - 1, round(p / 100 * (len(tt) - 1)))]
        row.update({"ttft_p50": statistics.median(tt), "ttft_p95": pct(95), "ttft_max": tt[-1], "n_streams": len(tt),
                    "stream_decode_median": statistics.median(s["decode_tok_s"] for s in streams if s["decode_tok_s"]),
                    "prompt_tokens": sorted({s["prompt_tokens"] for s in streams if s.get("prompt_tokens")})[:3]})
        out["rows"].append(row)
        print(f"N={n} on {k} server(s): agg(sum) {row['agg_sum_median']:.1f} tok/s (bursts "
              + ", ".join(f"{b['agg_sum_tok_s']:.1f}" for b in bs) + f"), combined {row['agg_combined_median']:.1f}, per-stream decode median "
              f"{row['stream_decode_median']:.1f}, ttft p50 {row['ttft_p50']:.3f} p95 {row['ttft_p95']:.3f}, errors {row['errors']}, prompt_tokens {row['prompt_tokens']}", flush=True)
    return out


def ident(a, urls):
    """Four fixed prompts: solo on each server in turn, then all at once on every server (one stream per prompt per server)."""
    prompts = [("ident%d" % j, f"[ident{j}] " + filler(a.sentences, 7000 + j)) for j in range(4)]
    out = {"solo": {}, "dual": {}}
    for s, url in enumerate(urls):
        out["solo"][url] = []
        for _, p in prompts:
            r = burst([url], a.model, [(0, p)], a.gen)["streams"][0]
            out["solo"][url].append(r.get("sha"))
    jobs = [(s, p) for s in range(len(urls)) for _, p in prompts]
    r = burst(urls, a.model, jobs, a.gen)["streams"]
    for s, url in enumerate(urls):
        out["dual"][url] = [x.get("sha") for x in r[s * len(prompts):(s + 1) * len(prompts)]]
    allsha = {x for v in list(out["solo"].values()) + list(out["dual"].values()) for x in v}
    out["same_across_servers_and_load"] = (len(allsha) == len(prompts) and all(
        out["solo"][urls[0]] == v for v in list(out["solo"].values()) + list(out["dual"].values())))
    print("ident: solo/dual sha per prompt", json.dumps(out["solo"]), json.dumps(out["dual"]), "identical:", out["same_across_servers_and_load"], flush=True)
    return out


def ctx_probe(a, urls):
    """Calibrate sentences -> tokens with the usage field, then send one prompt of about --ctx-probe tokens (32 generated)."""
    url = urls[0]
    r0 = burst([url], a.model, [(0, prompt_for(1, 0, 0, a.sentences))], 4)["streams"][0]
    per = (r0["prompt_tokens"] - 30) / a.sentences     # tokens per sentence (the template adds a few tokens)
    n = max(1, round((a.ctx_probe - 40) / per))
    t = time.perf_counter()
    r = burst([url], a.model, [(0, f"[ctx] " + filler(n, 424242))], 32)["streams"][0]
    out = {"url": url, "target_prompt_tokens": a.ctx_probe, "sentences": n, "ok": "error" not in r and r.get("status") == 200,
           "prompt_tokens": r.get("prompt_tokens"), "completion_tokens": r.get("tokens"), "ttft_s": r.get("ttft_s"), "wall_s": time.perf_counter() - t, "stream": r}
    print("ctx-probe:", json.dumps({k: v for k, v in out.items() if k != "stream"}), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", required=True); ap.add_argument("--model", default="intarsia")
    ap.add_argument("--streams", default="1,8,16,32"); ap.add_argument("--bursts", type=int, default=3)
    ap.add_argument("--gen", type=int, default=128); ap.add_argument("--sentences", type=int, default=45)
    ap.add_argument("--no-warm", action="store_true"); ap.add_argument("--ident", action="store_true")
    ap.add_argument("--ctx-probe", type=int, default=0); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    urls = a.urls.split(",")
    res = ident(a, urls) if a.ident else ctx_probe(a, urls) if a.ctx_probe else sweep(a, urls)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

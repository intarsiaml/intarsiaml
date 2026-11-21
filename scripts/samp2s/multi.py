#!/usr/bin/env python3
"""samp2s: N concurrent streaming requests (sampled or greedy), timed per stream and over their overlap window.

  uv run --project /opt/code/deepeval python scripts/samp2s/multi.py --url URL --model guidance --arm a1-s2-T0.7 --out DIR \
      --streams 2 --temperature 0.7 --top-k 20 --sizes 4096,32768,65536 --reps 8 [--warm] [--metrics URL/metrics]

Per rep: N distinct nonce'd prompts of about SIZE tokens (filler text), all started together; exactly --gen tokens forced (ignore_eos + min_tokens).
Per stream: decode tok/s = (completion_tokens - 1) / (last chunk - first chunk), localeval's definition.
Overlap window = [latest first chunk, earliest last chunk] (all N streams decoding); tokens in it are the stream's completion_tokens times the
share of its characters that arrived inside the window (linear between chunks; MTP chunks carry several tokens). `combined_overlap` = their sum / window.
`combined_whole` = (sum of completion_tokens - N) / (latest last chunk - earliest first chunk), prefill stagger and single-stream tails included.
--warm: every prompt is first sent once with max_tokens 1 (one after the other), then the timed requests start together on the stored prefixes,
so all N streams reach decode at about the same time even at 64k (the prefix store restores them).
`run_streams` is also used by seeded2.py."""
import argparse, http.client, json, pathlib, re, statistics, sys, threading, time, urllib.parse, urllib.request, uuid
from localeval.speed import INSTRUCTION, filler


def post_stream(url, body, res):
    """One streaming chat request; res gets marks [(t, chars)], text, usage, error."""
    u = urllib.parse.urlsplit(url.rstrip("/") + "/chat/completions")
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=1800)
    res.update(marks=[], text=[], usage={}, error=None, finish=None)
    try:
        conn.request("POST", u.path, body=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        if resp.status != 200:
            res["error"] = f"HTTP {resp.status}: {resp.read().decode('utf-8', 'replace')[:600]}"
            return
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            p = line[5:].strip()
            if p == "[DONE]":
                break
            ev = json.loads(p)
            if "error" in ev:
                res["error"] = json.dumps(ev["error"])[:500]
                return
            res["usage"] = ev.get("usage") or res["usage"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                c = (d.get("content") or "") + (d.get("reasoning_content") or d.get("reasoning") or "")
                if c:
                    res["text"].append(c)
                    res["marks"].append((time.perf_counter(), len(c)))
                if ch.get("finish_reason"):
                    res["finish"] = ch["finish_reason"]
    except Exception as e:  # noqa: BLE001
        res["error"] = repr(e)
    finally:
        conn.close()


def body_for(model, prompt, gen, temperature, top_p, top_k, seed, force):
    b = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": gen, "temperature": temperature, "stream": True,
         "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    if top_p is not None:
        b["top_p"] = top_p
    if top_k is not None:
        b["top_k"] = top_k
    if seed is not None:
        b["seed"] = seed
    if force:
        b["ignore_eos"] = True
        b["min_tokens"] = gen
    return b


def run_streams(url, bodies):
    """Start all bodies at once (threads), return one result dict per body."""
    rs = [{} for _ in bodies]
    ts = [threading.Thread(target=post_stream, args=(url, b, r)) for b, r in zip(bodies, rs)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return rs


def cum_chars(marks, t):
    """Characters received by time t: a chunk's characters spread linearly over the gap since the previous chunk (the first chunk is a point at its time)."""
    c, pt = 0.0, None
    for mt, n in marks:
        if pt is None:
            if t < mt:
                return 0.0
            c, pt = float(n), mt
        elif t >= mt:
            c, pt = c + n, mt
        else:
            return c + n * (t - pt) / (mt - pt)
        if t == mt:
            return c
    return c


def analyse(rs):
    ok = all(not r["error"] and r["marks"] and r["usage"].get("completion_tokens") for r in rs)
    if not ok:
        return {"error": [r["error"] for r in rs if r["error"]] or ["no tokens"]}
    first = [r["marks"][0][0] for r in rs]; last = [r["marks"][-1][0] for r in rs]
    ct = [r["usage"]["completion_tokens"] for r in rs]
    per = [(n - 1) / (l - f) if l > f else 0.0 for n, f, l in zip(ct, first, last)]
    lo, hi = max(first), min(last)
    out = {"completion_tokens": ct, "prompt_tokens": [r["usage"].get("prompt_tokens") for r in rs], "per_stream": [round(x, 1) for x in per],
           "combined_whole": round((sum(ct) - len(ct)) / (max(last) - min(first)), 1), "overlap_s": round(max(0.0, hi - lo), 2)}
    if hi - lo >= 1.0:
        toks = []
        for r, n in zip(rs, ct):
            tot = sum(c for _, c in r["marks"])
            toks.append(n * (cum_chars(r["marks"], hi) - cum_chars(r["marks"], lo)) / tot)
        out["per_stream_overlap"] = [round(t / (hi - lo), 1) for t in toks]
        out["combined_overlap"] = round(sum(toks) / (hi - lo), 1)
    return out


COUNTERS = ("intarsia_spec_steps_total", "intarsia_spec_proposed_tokens_total", "intarsia_spec_accepted_tokens_total", "intarsia_spec_verify_topk_solo_total",
            "intarsia_spec_verify_topk_fallback_total", "intarsia_spec_keyed_graph_steps_total")


def scrape(url):
    if not url:
        return {}
    try:
        txt = urllib.request.urlopen(url, timeout=10).read().decode()
    except Exception:
        return {}
    out = {}
    for line in txt.splitlines():
        m = re.match(r"^(intarsia_spec_\w+)\s+([-0-9.e+]+)$", line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def calibrate(url, model, size):
    """Sentence count that gives about `size` prompt tokens (one probe at 24 tokens per sentence, then scaled once)."""
    n = max(1, round(size / 24))
    run = run_streams(url, [body_for(model, filler(n, seed=size) + INSTRUCTION, 1, 0, None, None, None, False)])[0]
    probe = run["usage"]["prompt_tokens"]
    return max(1, round(n * size / probe))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True); ap.add_argument("--model", required=True); ap.add_argument("--arm", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--streams", type=int, default=2); ap.add_argument("--sizes", default="4096"); ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--gen", type=int, default=512); ap.add_argument("--temperature", type=float, default=0.7); ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=None); ap.add_argument("--warm", action="store_true"); ap.add_argument("--metrics", default="")
    a = ap.parse_args()
    top_p = a.top_p if a.temperature > 0 else None
    rows = []
    for size in [int(s) for s in a.sizes.split(",")]:
        n = calibrate(a.url, a.model, size)
        m0 = scrape(a.metrics)
        reps = []
        for rep in range(a.reps):
            # distinct prompts per stream and per rep, each with a nonce first so no stored prefix of another rep matches
            prompts = [f"nonce {uuid.uuid4().hex} " + filler(n, seed=size + 17 * rep + 1000 * k) + INSTRUCTION for k in range(a.streams)]
            if a.warm:
                for p in prompts:
                    w = run_streams(a.url, [body_for(a.model, p, 1, 0, None, None, None, False)])[0]
                    if w["error"]:
                        print(f"[multi] warm error {w['error']}", flush=True)
            bodies = [body_for(a.model, p, a.gen, a.temperature, top_p, a.top_k, None, True) for p in prompts]
            rs = run_streams(a.url, bodies)
            an = analyse(rs)
            reps.append(an)
            print(f"[multi] {a.arm} {size} rep{rep + 1}: " + json.dumps(an), flush=True)
            time.sleep(1.5)
        m1 = scrape(a.metrics)
        d = {k: m1.get(k, 0) - m0.get(k, 0) for k in m1}
        steps = d.get("intarsia_spec_steps_total")
        good = [r for r in reps if "error" not in r]
        row = {"size": size, "streams": a.streams, "reps": reps, "counters": {k: d[k] for k in sorted(d)},
               "tokens_per_verify": round((d.get("intarsia_spec_accepted_tokens_total", 0) + steps) / steps, 3) if steps else None,
               "combined_whole_each": [r["combined_whole"] for r in good],
               "combined_overlap_each": [r["combined_overlap"] for r in good if "combined_overlap" in r],
               "per_stream_each": [x for r in good for x in r["per_stream"]], "errors": [r["error"] for r in reps if "error" in r]}
        rows.append(row)
        print(f"[multi] {a.arm} {size}: whole {statistics.mean(row['combined_whole_each']) if row['combined_whole_each'] else None} "
              f"overlap {statistics.mean(row['combined_overlap_each']) if row['combined_overlap_each'] else None} "
              f"({len(row['combined_overlap_each'])}/{a.reps} reps with overlap) tpv {row['tokens_per_verify']} counters {row['counters']}", flush=True)
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.arm}-multi.json").write_text(json.dumps({"arm": a.arm, "streams": a.streams, "temperature": a.temperature, "top_p": top_p, "top_k": a.top_k,
                                                        "gen": a.gen, "warm": a.warm, "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

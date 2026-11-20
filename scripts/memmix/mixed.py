#!/usr/bin/env python3
"""memgrow: mixed-traffic load for a memory soak (what the replay soaks do not exercise): varied prompt sizes (0.5k to 90k tokens), stream and
non-stream, thinking default and off, temperature 0 / 1, long generations, tool calls, json_schema, multi-turn chats that hit the prefix
store, and client aborts (mid-prefill and mid-decode, by closing the socket).
  cd /opt/code/deepeval && uv run python /path/scripts/memgrow/mixed.py --url http://HOST:PORT --model guidance --minutes 45 --streams 2 --seed 7 --out DIR
Per-request rows go to DIR/mixed-<seed>.jsonl (no reply text), a summary to DIR/mixed-<seed>.summary.json."""
import argparse, collections, http.client, json, random, sys, threading, time, urllib.parse
from localeval.speed import filler

TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}}, "required": ["city"]}}}]
SCHEMA = {"type": "json_schema", "json_schema": {"name": "p", "schema": {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "required": ["name", "age"]}}}
TOK_PER_SENT = 22  # rough tokens per filler sentence (measured only as an order of magnitude; the server reports the real count)


def post(host, port, body, stream, abort_after=None, abort_time=None, timeout=900):
    """returns (status, finish, ptoks, ctoks, ttft, total, note). abort_after = close after this many SSE data chunks; abort_time = close after N s."""
    t0 = time.perf_counter()
    c = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
        if abort_time is not None:
            time.sleep(abort_time)
            c.sock.close() if c.sock else None
            return 0, "aborted_time", 0, 0, None, time.perf_counter() - t0, f"closed after {abort_time:.1f}s"
        r = c.getresponse()
        if not stream:
            data = r.read()
            try:
                j = json.loads(data)
            except Exception:
                return r.status, None, 0, 0, None, time.perf_counter() - t0, data[:120].decode("utf8", "replace")
            u = j.get("usage") or {}
            ch = (j.get("choices") or [{}])[0]
            return r.status, ch.get("finish_reason"), u.get("prompt_tokens", 0), u.get("completion_tokens", 0), None, time.perf_counter() - t0, "" if r.status == 200 else str(j)[:120]
        if r.status != 200:
            return r.status, None, 0, 0, None, time.perf_counter() - t0, r.read()[:120].decode("utf8", "replace")
        n, ttft, fin, pt, ct = 0, None, None, 0, 0
        for line in r:
            if not line.startswith(b"data:"):
                continue
            p = line[5:].strip()
            if p == b"[DONE]":
                break
            try:
                j = json.loads(p)
            except Exception:
                continue
            if ttft is None:
                ttft = time.perf_counter() - t0
            n += 1
            if j.get("usage"):
                pt, ct = j["usage"].get("prompt_tokens", 0), j["usage"].get("completion_tokens", 0)
            if j.get("choices"):
                fin = j["choices"][0].get("finish_reason") or fin
            if abort_after is not None and n >= abort_after:
                r.close(); c.close()
                return 200, "aborted_chunks", pt, ct, ttft, time.perf_counter() - t0, f"closed after {n} chunks"
        return 200, fin, pt, ct, ttft, time.perf_counter() - t0, ""
    except Exception as e:
        return -1, None, 0, 0, None, time.perf_counter() - t0, f"{type(e).__name__}: {e}"[:120]
    finally:
        try:
            c.close()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--minutes", type=float, required=True)
    ap.add_argument("--streams", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--profile", choices=["mixed", "tiny"], default="mixed", help="tiny: many short requests (a per-request leak shows up fast), no long prompts or generations")
    ap.add_argument("--weights", default="", help="memmix: override the request-kind weights, e.g. 'abort_decode=10,abort_prefill=8' (kinds not named get 0); bisect arms")
    a = ap.parse_args()
    tiny = a.profile == "tiny"
    u = urllib.parse.urlparse(a.url)
    host, port = u.hostname, u.port
    import pathlib
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    f = (out / f"mixed-{a.seed}.jsonl").open("w")
    lock = threading.Lock(); stats = collections.Counter(); kinds = collections.Counter(); errs = collections.Counter()
    end = time.time() + a.minutes * 60
    convs = {}  # multi-turn conversations by id (prefix-store hits)

    KINDS = ["chat", "long", "abort_decode", "abort_prefill", "tool", "schema", "longgen", "multiturn"]
    weights = [40, 0, 10, 5, 15, 10, 0, 20] if tiny else [38, 5, 10, 8, 10, 5, 10, 14]
    if a.weights:
        wd = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in a.weights.split(",")}
        assert set(wd) <= set(KINDS), wd
        weights = [wd.get(k, 0) for k in KINDS]

    def one(w, rng, i):
        kind = rng.choices(KINDS, weights)[0]
        think = rng.random() < 0.5
        body = {"model": a.model, "temperature": rng.choice([0.0, 1.0]), "stream": rng.random() < 0.6}
        if not think:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        nonce = f"[req {a.seed}-{w}-{i}-{rng.random():.6f}] "
        stream = body["stream"]; abort_after = abort_time = None
        if kind == "chat":
            n = rng.choice([5, 20, 60] if tiny else [20, 60, 150, 400, 900]); body["max_tokens"] = rng.choice([8, 16, 32] if tiny else [32, 128, 400, 900])
            body["messages"] = [{"role": "user", "content": nonce + filler(n, rng.randrange(10**6)) + "\n\nSummarize the notes in two sentences."}]
        elif kind == "long":
            n = rng.choice([2200, 3000, 4000]); body["max_tokens"] = 32; body["stream"] = stream = True
            body["messages"] = [{"role": "user", "content": nonce + filler(n, rng.randrange(10**6)) + "\n\nHow many notes were there? Answer briefly."}]
        elif kind == "abort_decode":
            n = rng.choice([10, 40] if tiny else [40, 200, 600]); body["max_tokens"] = 1500; body["stream"] = stream = True; abort_after = rng.choice([1, 3, 20, 60])
            body["messages"] = [{"role": "user", "content": nonce + filler(n, rng.randrange(10**6)) + "\n\nWrite a long story about these notes."}]
        elif kind == "abort_prefill":
            n = rng.choice([200, 400] if tiny else [800, 1500, 2500]); body["max_tokens"] = 64; body["stream"] = stream = True; abort_time = rng.choice([0.2, 0.6, 1.5, 3.0])
            body["messages"] = [{"role": "user", "content": nonce + filler(n, rng.randrange(10**6)) + "\n\nSummarize."}]
        elif kind == "tool":
            body["max_tokens"] = 300; body["tools"] = TOOLS; body["tool_choice"] = "auto"
            body["messages"] = [{"role": "user", "content": nonce + f"What is the weather in {rng.choice(['Paris', 'Oslo', 'Lima', 'Kyoto', 'Perth'])} right now?"}]
        elif kind == "schema":
            body["max_tokens"] = 200; body["response_format"] = SCHEMA; body["stream"] = stream = False
            body["messages"] = [{"role": "user", "content": nonce + "Give me a person as JSON with name and age."}]
        elif kind == "longgen":
            body["max_tokens"] = rng.choice([1500, 3000]); body["messages"] = [{"role": "user", "content": nonce + "Write a detailed essay on how bridges are designed, with sections."}]
        else:  # multiturn: grow one of a few conversations (system + notes prefix shared, store hits)
            cid = rng.randrange(6)
            with lock:
                hist = convs.setdefault(cid, [{"role": "system", "content": f"Conversation {a.seed}-{cid}. " + filler(300 + 40 * cid, 900 + cid)}])
                if len(hist) > 14:
                    hist[:] = hist[:1]
                hist.append({"role": "user", "content": f"Turn {len(hist)}: " + filler(rng.choice([20, 80]), rng.randrange(10**6)) + " Reply in one short paragraph."})
                msgs = list(hist)
            body["max_tokens"] = 120; body["messages"] = msgs; body["chat_template_kwargs"] = {"enable_thinking": False}
        if stream:
            body["stream_options"] = {"include_usage": True}
        st, fin, pt, ct, ttft, tot, note = post(host, port, body, stream, abort_after, abort_time)
        if kind == "multiturn" and st == 200:
            with lock:
                if convs.get(cid):
                    convs[cid].append({"role": "assistant", "content": "Understood."})
        row = {"t": round(time.time(), 1), "w": w, "kind": kind, "stream": stream, "think": think, "temp": body["temperature"], "status": st, "finish": fin,
               "prompt": pt, "completion": ct, "ttft": None if ttft is None else round(ttft, 2), "total": round(tot, 2), "note": note}
        with lock:
            f.write(json.dumps(row) + "\n"); f.flush()
            kinds[kind] += 1
            ok = (st == 200) or (kind in ("abort_prefill",) and st == 0)
            stats["ok" if ok else "bad"] += 1
            if not ok:
                errs[f"{kind}:{st}:{note[:60]}"] += 1

    def worker(w):
        rng = random.Random(a.seed * 1000 + w); i = 0
        while time.time() < end:
            one(w, rng, i); i += 1

    ths = [threading.Thread(target=worker, args=(w,)) for w in range(a.streams)]
    [t.start() for t in ths]; [t.join() for t in ths]
    f.close()
    summ = {"seed": a.seed, "profile": a.profile, "weights": a.weights, "minutes": a.minutes, "streams": a.streams, "requests": sum(kinds.values()), "kinds": dict(kinds), "ok": stats["ok"], "bad": stats["bad"], "errors": dict(errs)}
    (out / f"mixed-{a.seed}.summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ))


main()

#!/usr/bin/env python3
"""Lane itl2s: load generator that logs the arrival time of every SSE chunk (stdlib only; runs on the server's own box so the clock
is the server's: its `cb-trace-us t=<epoch us>` lines and these marks line up).

Same prompts, seeds (2000 + i, + 1000 per extra round), sampling and forced length as scripts/flipprep/speed.py (localeval's request
body: ignore_eos + min_tokens, thinking default or off). Streams run in consecutive groups of --streams (all started together).

  python3 itl.py --url http://localhost:8194/v1 --model guidance --visit F.a --set en --temperature 0 --thinking default --streams 2 --out DIR
Writes DIR/itl-<visit>-<set>[-nt]-t<temp>-s<streams>.json: per request {marks: [[epoch_s, chars], ...], t_send, usage, sha256, ...}.
"""
import argparse, hashlib, http.client, json, pathlib, threading, time, urllib.parse

# prompt sets: verbatim from scripts/flipprep/speed.py (kept in sync by copy; the sets are flipprep's)
sys_path = pathlib.Path(__file__).resolve().parents[1] / "flipprep" / "speed.py"


def load_sets():
    """Read the prompt lists out of flipprep's speed.py without importing localeval."""
    import ast
    tree = ast.parse(sys_path.read_text())
    env = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("PROMPTS", "CJK", "CODE", "MIXED"):
            env[node.targets[0].id] = eval(compile(ast.Expression(node.value), "<speed.py>", "eval"), {}, env)
    return env


def one(url, model, prompt, seed, temperature, thinking, gen, out, i):
    u = urllib.parse.urlsplit(url.rstrip("/") + "/chat/completions")
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": gen, "temperature": temperature, "stream": True,
            "stream_options": {"include_usage": True}, "seed": seed, "ignore_eos": True, "min_tokens": gen}
    if thinking != "default":
        body["chat_template_kwargs"] = {"enable_thinking": thinking == "on"}
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=600)
    marks, text, usage = [], [], {}
    try:
        conn.request("POST", u.path, body=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        t_send = time.time()
        resp = conn.getresponse()
        if resp.status != 200:
            out[i] = {"ok": False, "error": f"HTTP {resp.status}: {resp.read()[:300]!r}"}
            return
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            p = line[5:].strip()
            if p == "[DONE]":
                break
            ev = json.loads(p)
            usage = ev.get("usage") or usage
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                c = (d.get("content") or "") + (d.get("reasoning_content") or d.get("reasoning") or "")
                if c or d.get("tool_calls"):
                    marks.append([time.time(), len(c)])
                    text.append(c)
        out[i] = {"ok": True, "t_send": t_send, "marks": marks, "usage": usage, "sha256": hashlib.sha256("".join(text).encode()).hexdigest()[:16], "seed": seed,
                  "prompt_idx": i}
    except Exception as e:  # noqa: BLE001
        out[i] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    finally:
        conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--visit", required=True)
    ap.add_argument("--set", default="en", choices=["en", "cjk", "code", "mixed"])
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--thinking", default="default", choices=["default", "on", "off"])
    ap.add_argument("--streams", type=int, default=2)
    ap.add_argument("--gen", type=int, default=384)
    ap.add_argument("--rounds", type=int, default=2, help="the 8 prompts this many times (seed + 1000 per round); round 0 is the same request set as flipprep's")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    S = load_sets()
    prompts = {"en": S["PROMPTS"], "cjk": S["CJK"], "code": S["CODE"], "mixed": S["MIXED"]}[a.set]
    # warm-up (not recorded): first-request costs
    # (a group of --streams concurrent requests at the cell's temperature, twice: batched verify, sampled top-k rows, graphs)
    for rep in range(2):
        w = {}
        ts = [threading.Thread(target=one, args=(a.url, a.model, prompts[(rep + j) % len(prompts)], 1 + j, a.temperature, a.thinking, 96, w, j)) for j in range(a.streams)]
        [t.start() for t in ts]
        [t.join() for t in ts]
    res = []
    t_start = time.time()
    for rnd in range(a.rounds):
        for lo in range(0, len(prompts), a.streams):
            idx = list(range(lo, min(lo + a.streams, len(prompts))))
            out = {}
            ts = [threading.Thread(target=one, args=(a.url, a.model, prompts[i], 2000 + 1000 * rnd + i, a.temperature, a.thinking, a.gen, out, i)) for i in idx]
            t_group = time.time()
            [t.start() for t in ts]
            [t.join() for t in ts]
            for i in idx:
                r = out[i]
                r.update({"round": rnd, "group": lo // a.streams, "t_group": t_group})
                res.append(r)
    t_end = time.time()
    name = f"itl-{a.visit}-{a.set}{'-nt' if a.thinking == 'off' else ''}-t{a.temperature:g}-s{a.streams}.json"
    p = pathlib.Path(a.out)
    p.mkdir(parents=True, exist_ok=True)
    (p / name).write_text(json.dumps({"visit": a.visit, "set": a.set, "thinking": a.thinking, "temperature": a.temperature, "streams": a.streams, "gen": a.gen,
                                      "t_start": t_start, "t_end": t_end, "requests": res}))
    bad = [r.get("error") for r in res if not r["ok"]]
    toks = sum(r["usage"].get("completion_tokens", 0) for r in res if r["ok"])
    print(f"[itl] {a.visit} {a.set} t={a.temperature} s={a.streams}: {len(res)} requests, {len(bad)} failed, {toks} tokens in {t_end - t_start:.1f}s -> {name}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

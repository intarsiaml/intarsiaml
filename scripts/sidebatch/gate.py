#!/usr/bin/env python3
"""sidebatch: one server's waves of concurrent requests (non-streaming; reply sha per request). One JSON line per request.
  gate.py <port> <cells-dir> <out-dir> <arm> [waves]
waves (comma list, default all): warm (8 short greedy streams), g8 (8 concurrent greedy streams, staggered max_tokens), g8s (8 greedy, starts staggered by 0.35 s,
fresh nonces), s8 (8 sampled: T 0.7, top_p 0.95, seed 1000+i), long (one ~16K-token prompt, 512 tokens, greedy), longmix (that long
prompt beside 4 decoding streams). Prompts: scripts/mcopy cells; every request gets a '[wave-i] ' nonce prefix (same in every arm, so
no request is a prefix-store hit of an earlier one). Thinking off."""
import hashlib, json, pathlib, sys, threading, time, urllib.request

port, cells, out, arm = sys.argv[1], pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]), sys.argv[4]
waves = (sys.argv[5] if len(sys.argv) > 5 else "g8,g8s,s8,long,longmix").split(",")
out.mkdir(parents=True, exist_ok=True)
NAMES = ["copy3k", "agent-edit", "agent-batch", "json-xform", "chat-short", "reason", "code-gen", "code-gen"]
MAXT = [600, 450, 520, 400, 300, 500, 700, 350]
lock = threading.Lock()
LOG = open(out / f"{arm}.jsonl", "a")


def msgs_of(name, tag):
    d = json.load(open(cells / f"{name}.json"))
    m = json.loads(json.dumps(d["messages"]))
    first = next(x for x in m if x["role"] == "user")
    first["content"] = f"[{tag}] " + first["content"]
    return m


def long_msgs(tag):
    src = ""
    for f in ["docs/tuning.md"]:
        src += (cells.parent / "src" / f).read_text()
    src = src[:52000]
    return [{"role": "user", "content": f"[{tag}] Below is a long reference document.\n\n" + src +
             "\n\nWrite a detailed, structured summary of the document: list the main sections and, for each, the knobs it describes."}]


def one(wave, i, messages, max_tokens, sampled, delay=0.0):
    time.sleep(delay)
    body = {"model": "m", "stream": False, "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": False}, "messages": messages}
    if sampled:
        body.update({"temperature": 0.7, "top_p": 0.95, "seed": 1000 + i})
    else:
        body["temperature"] = 0
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"})
    t0 = time.time()
    try:
        d = json.load(urllib.request.urlopen(req, timeout=1500))
        ch = d["choices"][0]
        txt = ch["message"].get("content") or ""
        rec = {"arm": arm, "wave": wave, "i": i, "sha": hashlib.sha1(txt.encode()).hexdigest()[:12], "toks": d["usage"]["completion_tokens"],
               "ptoks": d["usage"]["prompt_tokens"], "finish": ch.get("finish_reason"), "wall": round(time.time() - t0, 2)}
        (out / f"{arm}-{wave}-{i}.txt").write_text(txt)
    except Exception as e:  # noqa: BLE001
        rec = {"arm": arm, "wave": wave, "i": i, "error": str(e)[:200]}
    with lock:
        print(json.dumps(rec), file=LOG, flush=True)
        print(json.dumps(rec), flush=True)


def run(w, jobs):
    global wave
    wave = w
    t0 = time.time()
    th = [threading.Thread(target=one, args=(w, *j)) for j in jobs]
    [t.start() for t in th]
    [t.join() for t in th]
    print(json.dumps({"arm": arm, "wave": w, "wall_total": round(time.time() - t0, 2)}), file=LOG, flush=True)


wave = ""
for w in waves:
    if w == "g8":
        run(w, [(i, msgs_of(NAMES[i], f"g8-{i}"), MAXT[i], False) for i in range(8)])
    elif w == "g8s":
        run(w, [(i, msgs_of(NAMES[i], f"g8s-{i}"), MAXT[7 - i], False, 0.35 * i) for i in range(8)])
    elif w == "s8":
        run(w, [(i, msgs_of(NAMES[i], f"s8-{i}"), MAXT[i], True) for i in range(8)])
    elif w == "warm":
        run(w, [(i, msgs_of("chat-short", f"warm-{i}"), 40, False) for i in range(8)])
    elif w == "long":
        run(w, [(0, long_msgs("long-0"), 512, False)])
    elif w == "longmix":
        run(w, [(0, long_msgs("longmix-0"), 512, False)] + [(i, msgs_of(NAMES[i], f"longmix-{i}"), 400, False, 0.2) for i in range(1, 5)])

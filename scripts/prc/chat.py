#!/usr/bin/env python3
"""Lane prc: served identity of 2-turn chats and turn-2 time to first token, against one running `serve` (not restarted between
requests, so a follow-up turn finds its earlier slot). Chat requests at temperature 0, thinking off, 48 forced tokens.
  chat.py run URL OUT        identity set: CHATS conversations of 2 turns one at a time (SEQ), repeated once (a repeated turn 1 and
                             turn 2), then all at once (CONC). Saves [{phase, chat, turn, text, cached, prompt}].
  chat.py compare REF OUT    REF is a run of a server with INTARSIA_CB_NO_REUSE=1 (every request cold)
  chat.py ttft URL N [CHARS] N conversations (default 6000 chars, ~1.5k tokens, of the corpus at different offsets): turn 1, then
                             turn 2 streamed; prints turn 2's time to first token and cached tokens per conversation
"""
import json, os, sys, threading, time, urllib.request

CORPUS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'tools', 'prompts', 'corpus_docs.txt')
FIRST = ["What is the capital of Australia, and why was it chosen?", "Explain, step by step, why the sky is blue.",
         "Write a haiku about autumn, then explain its imagery.", "Describe how TCP differs from UDP."]
SECOND = "Please continue, and explain it in more detail."


def chat(url, messages, stream=False, max_tokens=48):
    body = {"model": "intarsia", "messages": messages, "max_tokens": max_tokens, "min_tokens": max_tokens, "ignore_eos": True,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}, "stream": stream,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(url + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    if not stream:
        with urllib.request.urlopen(req, timeout=900) as r:
            j = json.load(r)
        u = j.get("usage") or {}
        return j["choices"][0]["message"]["content"] or "", (u.get("prompt_tokens_details") or {}).get("cached_tokens"), u.get("prompt_tokens"), None
    t0, first, text, usage = time.time(), None, "", {}
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            j = json.loads(line[5:])
            if j.get("usage"):
                usage = j["usage"]
            for c in j.get("choices") or []:
                d = (c.get("delta") or {}).get("content")
                if d:
                    first = first or time.time() - t0
                    text += d
    return text, (usage.get("prompt_tokens_details") or {}).get("cached_tokens"), usage.get("prompt_tokens"), first


def conv(url, i, tag, phase, recs, first_user=None):
    """Two turns of chat i: turn 2 carries turn 1's answer as history."""
    q = first_user or FIRST[i % len(FIRST)]
    h = [{"role": "user", "content": q}]
    t1, c1, p1, _ = chat(url, h)
    recs.append({"phase": phase, "chat": i, "turn": 1, "text": t1, "cached": c1, "prompt": p1})
    h += [{"role": "assistant", "content": t1}, {"role": "user", "content": SECOND}]
    t2, c2, p2, _ = chat(url, h)
    recs.append({"phase": phase, "chat": i, "turn": 2, "text": t2, "cached": c2, "prompt": p2})


def run(url, out):
    recs = []
    for rnd in ("SEQ", "REPEAT"):
        for i in range(len(FIRST)):
            conv(url, i, 0, rnd, recs)
            print(rnd, i, recs[-2]["cached"], recs[-1]["cached"], flush=True)
    for rnd in range(2):
        got = [[] for _ in FIRST]
        ths = [threading.Thread(target=conv, args=(url, i, 0, f"CONC{rnd}", got[i])) for i in range(len(FIRST))]
        [t.start() for t in ths]; [t.join() for t in ths]
        recs += [r for g in got for r in g]
        print("CONC", rnd, flush=True)
    json.dump(recs, open(out, "w"), indent=1)


def compare(ref, out):
    R, O = json.load(open(ref)), json.load(open(out))
    want = {}
    for r in R:
        want.setdefault((r["chat"], r["turn"]), r["text"])
        assert want[(r["chat"], r["turn"])] == r["text"], f"reference not stable for chat {r['chat']} turn {r['turn']}"
    bad = 0
    for r in O:
        same = r["text"] == want[(r["chat"], r["turn"])]
        bad += not same
        print(f"{r['phase']:7s} chat {r['chat']} turn {r['turn']} prompt {r['prompt']:4d} cached {r['cached']}: {'IDENTICAL' if same else 'DIFFER'}")
    print(f"{len(O) - bad}/{len(O)} requests identical to the cold reference")
    return bad


def ttft(url, n, chars):
    corpus = open(CORPUS).read()
    for k in range(n):
        doc = corpus[1000 * k + 5000:1000 * k + 5000 + chars]
        h = [{"role": "user", "content": f"Read this passage ({k}):\n\n{doc}\n\nWhat is it about?"}]
        t1, c1, p1, f1 = chat(url, h, stream=True)
        h += [{"role": "assistant", "content": t1}, {"role": "user", "content": SECOND}]
        t2, c2, p2, f2 = chat(url, h, stream=True)
        print(f"run {k}: turn 1 prompt {p1} cached {c1} ttft {f1 * 1000:.0f} ms; turn 2 prompt {p2} cached {c2} ttft {f2 * 1000:.0f} ms", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "ttft":
        ttft(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 6000)
    else:
        sys.exit(1 if compare(sys.argv[2], sys.argv[3]) else 0)

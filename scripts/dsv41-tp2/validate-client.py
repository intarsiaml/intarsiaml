#!/usr/bin/env python3
"""Lane dsv41/validate: clients and analysis for scripts/dsv41-tp2/validate-test.sh (docs/results/2026-11-05-dsv41-validate.md).

  predict                       print the cached-token prediction table of check 1 (a Python port of the engine's reuse rules)
  c1 URL OUT.json ARM TOK       check 1 against a running server (ids sent to /v1/completions): cold and hit B's, the cancel cases
  c1cmp ARM=OUT.json ...        check 1 analysis (pass rules in the doc)
  c3 URL OUT.json ARM MAXTOK ITEMS..   check 3 (thinking on, chat endpoint, token ids): ITEMS are label=file.jsonl:N (first N rows)
  c3cmp A.json B.json           check 3 analysis
  c4 URL OUT.json ARM ITEMS     check 4 (sampled decoding, temperature 0.7, fixed seeds)
  c4cmp A.json B.json           check 4 analysis
  c2cmp A.json B.json           check 2 analysis of two localeval result.json (FINAL parse, differing replies)
  qfirst SRC.jsonl N OUT.jsonl  turn the first N rows of a frozen chunk32k-2 shard into question-first prompts

Stdlib only, plus `tokenizers` for c1 (uv run --no-project --with tokenizers python validate-client.py ...).
"""
import json
import os
import re
import socket
import statistics as st
import sys
import time
import urllib.request
import urllib.parse

WINDOW, MARGIN, CHUNK = 128, 16, 2048
CODE_RE = re.compile(r"(?<![A-Z0-9])[A-Z]{2}-\d{4}-[A-Z]{2}(?![A-Z0-9]|-[A-Z0-9])")
FINAL_RE = re.compile(r"FINAL:\s*(.+?)\s*$", re.M)
DOC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../tools/prompts/corpus_docs.txt")
NA = 6000          # prompt A, tokens (BOS included): chunks 2048 / 2048 / 1904, floor 5872 in chunk 3


# ------------------------------------------------------------------ the engine's reuse rules (ported; see docs, "pass rules")
def floor(total, window=WINDOW):
    return max(total - window, 0)


def decoder_dead(end, ahead, chunk=CHUNK, window=WINDOW):          # replay.rs decoder_dead
    if ahead == 0 or chunk < window or window == 0:
        return False
    last = end + ahead
    lo = max(last - window, 0)
    e = end + max(-(-ahead // chunk) - 1, 0) * chunk
    while e > end:
        if e > lo:
            lo = min(lo, max(e - window, 0))
        e -= chunk
    return end <= lo


def simulate(prompt, replay, skip, upto=None):
    """The state (len, ring_high, replay_from) after prefilling `prompt` cold in chunks of CHUNK, or only the chunks ending at or
    before `upto` (a cancel between chunks). Each chunk is one forward call whose floor is its own end - window (runtime.rs
    replay_floor); a dead chunk runs encoder-only and the state is valid from its end, any other chunk from its floor."""
    rf, pos = 0, 0
    while pos < len(prompt):
        c = min(CHUNK, len(prompt) - pos)
        end, ahead = pos + c, len(prompt) - pos - c
        if upto is not None and end > upto:
            break
        if replay:
            f = floor(end)
            if skip and c > WINDOW and c <= CHUNK and decoder_dead(end, ahead):
                rf = end
            elif c >= 2 and f > 0:
                rf = f
        pos = end
    return {"len": pos, "high": pos, "rf": rf}


def keep_for(state, tokens_state, prompt, replay):
    common = min(lcp(tokens_state, prompt), len(prompt) - 1)
    want = min(common, state["len"])
    if not (want + MARGIN >= state["len"] and want + MARGIN + 1 >= state["high"]):
        return 0
    if replay and want and not (want <= floor(len(prompt)) or state["rf"] <= floor(len(prompt))):
        return 0
    return want


def predict(prompt_a, bs, replay, skip, cancel_at=None):
    """cached_tokens predicted for each B after A served with max_tokens 1, or cancelled with chunks 1 and 2 run (cancel_at rows)."""
    state = simulate(prompt_a, replay, skip, upto=cancel_at)
    return {name: keep_for(state, prompt_a[:state["len"]], b, replay) for name, b in bs.items()}


def variants(ids):
    """The B prompts: literal ones of the task, the cancel-landing one, and the in-margin hit that depends on cache history."""
    a = ids[:NA]
    x = ids[NA:]                                   # fresh text for the new tails
    return a, {
        "B1_A+300": a + x[:300],
        "B1m_A[:-8]+50": a[:NA - 8] + x[400:450],
        "B2_A[:4100]+50": a[:4100] + x[500:550],
        "B3_A[:2040]+50": a[:2040] + x[600:650],
        "B3p_A[:4088]+50": a[:4088] + x[700:750],
    }


# ------------------------------------------------------------------ http
def post(url, path, body, timeout=1800):
    req = urllib.request.Request(url.rstrip("/") + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def get(url, path, timeout=30):
    with urllib.request.urlopen(url.rstrip("/") + path, timeout=timeout) as r:
        return r.read().decode()


def completion(url, ids, n, **kw):
    body = {"model": "dsv41-intarsia", "prompt": ids, "max_tokens": n, "temperature": 0, "return_token_ids": True}
    body.update(kw)
    t0 = time.time()
    r = post(url, "/v1/completions", body)
    ch = r["choices"][0]
    return {"ids": [int(t) for t in (ch.get("token_ids") or r.get("token_ids") or [])], "text": ch.get("text", ""),
            "cached": int(((r.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens") or 0),
            "prompt_tokens": (r.get("usage") or {}).get("prompt_tokens"), "completion_tokens": (r.get("usage") or {}).get("completion_tokens"),
            "finish": ch.get("finish_reason"), "wall_s": round(time.time() - t0, 2)}


def metrics(url):
    try:
        t = get(url, "/metrics")
    except Exception as e:  # noqa
        return {"error": str(e)}
    return {l.split()[0]: l.split()[-1] for l in t.splitlines() if l and not l.startswith("#") and re.search(r"cancel|cached|prompt_tokens|gen_tokens|finished|completion", l)}


def stream_cancel(url, ids, delay):
    """Send a streaming completion and close the socket after `delay` seconds (the client going away mid-prefill)."""
    u = urllib.parse.urlparse(url)
    body = json.dumps({"model": "dsv41-intarsia", "prompt": ids, "max_tokens": 1, "temperature": 0, "stream": True}).encode()
    s = socket.create_connection((u.hostname, u.port), timeout=30)
    s.sendall((f"POST /v1/completions HTTP/1.1\r\nHost: {u.hostname}:{u.port}\r\nContent-Type: application/json\r\n"
               f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n").encode() + body)
    time.sleep(delay)
    try:
        s.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    s.close()


def degenerate(ids):
    """True when a reply looks like garbage: short of distinct tokens, or one 4-gram filling more than half of it."""
    if len(ids) < 8:
        return True
    grams = [tuple(ids[i:i + 4]) for i in range(len(ids) - 3)]
    rep = 1 - len(set(grams)) / max(len(grams), 1)
    return len(set(ids)) < max(6, len(ids) // 8) or rep > 0.5


def looping(ids, n=16, times=6):
    """Reply stuck in a loop: some n-gram of n tokens occurs `times` times or more. Quoting repeated document lines is not a loop
    (the ids inside the quoted lines differ); degenerate() is the pre-registered, looser proxy and misfires on long reasoning."""
    import collections
    c = collections.Counter(tuple(ids[i:i + n]) for i in range(len(ids) - n + 1))
    return bool(c) and max(c.values()) >= times


def lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


# ------------------------------------------------------------------ check 1
def c1(url, out, arm, tok_path):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tok_path)
    bos = tok.token_to_id("<｜begin▁of▁sentence｜>")
    text = open(DOC, encoding="utf-8").read()
    ids = [bos] + tok.encode(text, add_special_tokens=False).ids
    assert len(ids) > NA + 1000, len(ids)
    a, bs = variants(ids)
    nonce = [0]

    def evict():
        nonce[0] += 1
        e = tok.encode(f"zq{nonce[0]} unrelated separator request number {nonce[0]}: nothing here shares a prefix. " * 4, add_special_tokens=False).ids
        completion(url, e[:64], 2)          # no BOS: shares nothing with any B, so the cached sequence is replaced

    res = {"arm": arm, "NA": NA, "metrics0": metrics(url), "cold": {}, "hit": {}, "cancel": {}, "notes": []}
    order = list(bs)
    # 1. truly fresh: the first request of a freshly started server is cold B1
    res["cold_fresh_B1"] = completion(url, bs[order[0]], 64, ignore_eos=True)
    for n in order:
        evict()
        res["cold"][n] = completion(url, bs[n], 64, ignore_eos=True)
    evict()
    res["cold_again_B1"] = completion(url, bs[order[0]], 64, ignore_eos=True)
    # 2. hits: A served with one token (the state is exactly A), then B
    for n in order[:3]:
        evict()
        ra = completion(url, a, 1)
        res["hit"][n] = completion(url, bs[n], 64, ignore_eos=True)
        res["hit"][n]["A_cached"] = ra["cached"]
    # 3. cancel: close the client mid-prefill of A; calibrate the delay with a probe A[:4096]+200 (its cached_tokens = state length)
    probe = a[:4096] + ids[NA + 800:NA + 1000]
    chosen = None
    res["cancel"]["calibration"] = []
    for d in (0.8, 1.2, 0.5, 1.6, 2.4):
        evict()
        m0 = metrics(url)
        stream_cancel(url, a, d)
        time.sleep(8)
        m1 = metrics(url)
        p = completion(url, probe, 1)
        res["cancel"]["calibration"].append({"delay_s": d, "probe_cached": p["cached"], "probe_wall_s": p["wall_s"],
                                             "metrics_delta": {k: (m1.get(k), m0.get(k)) for k in m1 if m1.get(k) != m0.get(k)}})
        if p["cached"] in (2048, 4096):
            chosen = (d, p["cached"])
            break
    res["cancel"]["landing"] = chosen
    if chosen:
        for n in ("B3p_A[:4088]+50", "B3_A[:2040]+50"):
            evict()
            m0 = metrics(url)
            stream_cancel(url, a, chosen[0])
            time.sleep(8)
            m1 = metrics(url)
            r = completion(url, bs[n], 64, ignore_eos=True)
            r["metrics_delta"] = {k: (m1.get(k), m0.get(k)) for k in m1 if m1.get(k) != m0.get(k)}
            res["cancel"][n] = r
    else:
        res["notes"].append("no calibration delay landed the cancel at a chunk boundary (probe cached_tokens 2048 or 4096): cancel cases not run")
    res["metrics1"] = metrics(url)
    json.dump(res, open(out, "w"), indent=1)
    print("c1 done", arm, "landing", chosen)


def c1cmp(args):
    arms = {}
    for a in args:
        n, f = a.split("=", 1)
        arms[n] = json.load(open(f))
    NAME = {"default": (True, True), "off": (False, False), "skip0": (True, False)}
    ok = True
    from tokenizers import Tokenizer  # noqa
    tokp = os.environ["TOKENIZER"]
    tok = Tokenizer.from_file(tokp)
    bos = tok.token_to_id("<｜begin▁of▁sentence｜>")
    ids = [bos] + tok.encode(open(DOC, encoding="utf-8").read(), add_special_tokens=False).ids
    a, bs = variants(ids)
    print("# check 1\n")
    for n, r in arms.items():
        replay, skip = NAME[n]
        print(f"## arm {n}\n")
        fr, ag = r["cold_fresh_B1"], r["cold_again_B1"]
        same = fr["ids"] == ag["ids"]
        print(f"- cold B1 on the fresh server vs cold B1 after a separator request: ids identical = {same} ({len(fr['ids'])} tokens), cached {fr['cached']} / {ag['cached']}")
        ok &= same and fr["cached"] == 0
        pred = predict(a, bs, replay, skip)
        print("\n| B | predicted cached | measured cached | ids equal to cold B | equal prefix (of 64) | garbage |\n|---|---|---|---|---|---|")
        for nme in list(bs)[:3]:
            h, c = r["hit"][nme], r["cold"][nme]
            exp = pred[nme]
            g = degenerate(h["ids"]) or degenerate(c["ids"]) or len(h["ids"]) != 64
            hit_ok = abs(h["cached"] - exp) <= 1
            ok &= (not g) and hit_ok and (exp > 0 or h["ids"] == c["ids"])
            print(f"| {nme} | {exp} | {h['cached']}{'' if hit_ok else ' MISS'} | {h['ids'] == c['ids']} | {lcp(h['ids'], c['ids'])} | {g} |")
        cl = r["cancel"]
        print(f"\ncancel landing: {cl.get('landing')}; calibration: {json.dumps(cl.get('calibration'))[:900]}")
        if cl.get("landing"):
            at = cl["landing"][1]
            pc = predict(a, bs, replay, skip, cancel_at=at)
            for nme in ("B3p_A[:4088]+50", "B3_A[:2040]+50"):
                h, c = cl[nme], r["cold"][nme]
                exp = pc[nme]
                g = degenerate(h["ids"])
                hit_ok = abs(h["cached"] - exp) <= 1
                ok &= (not g) and hit_ok and (exp > 0 or h["ids"] == c["ids"])
                print(f"- after the cancel (state {at}): {nme}: predicted cached {exp}, measured {h['cached']}{'' if hit_ok else ' MISS'}, ids equal to cold {h['ids'] == c['ids']}, equal prefix {lcp(h['ids'], c['ids'])}, garbage {g}")
        print()
    # default vs off hit replies
    if "default" in arms and "off" in arms:
        print("## default against off, same B\n\n| B | hit ids equal | equal prefix | cold ids equal |\n|---|---|---|---|")
        for nme in list(bs)[:3]:
            d, o = arms["default"]["hit"][nme], arms["off"]["hit"][nme]
            dc, oc = arms["default"]["cold"][nme], arms["off"]["cold"][nme]
            print(f"| {nme} | {d['ids'] == o['ids']} | {lcp(d['ids'], o['ids'])} | {dc['ids'] == oc['ids']} |")
    print("\nPASS" if ok else "\nFAIL (see the rows marked MISS / False in the pass rule's terms)")
    return 0 if ok else 1


# ------------------------------------------------------------------ chat clients (checks 3, 4)
def chat(url, content, **kw):
    body = {"model": "dsv41-intarsia", "messages": [{"role": "user", "content": content}], "return_token_ids": True}
    body.update(kw)
    t0 = time.time()
    r = post(url, "/v1/chat/completions", body)
    ch = r["choices"][0]
    m = ch["message"]
    u = r.get("usage") or {}
    return {"ids": [int(t) for t in (ch.get("token_ids") or r.get("token_ids") or [])], "content": m.get("content") or "",
            "reasoning": m.get("reasoning_content") or "", "finish": ch.get("finish_reason"),
            "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
            "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            "cached": int((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0), "wall_s": round(time.time() - t0, 1)}


def rows(spec):
    label, rest = spec.split("=", 1)
    f, n = rest.rsplit(":", 1)
    return label, [json.loads(l) for l in open(f) if l.startswith("{")][:int(n)]


def score(reply, expected):
    """FINAL: <answer> items (chunk32k-2) or a bare code (recent-window items): the answer is the last FINAL line / the last code."""
    if expected.startswith("FINAL:"):
        m = FINAL_RE.findall(reply)
        return bool(m) and m[-1].strip() == expected[len("FINAL:"):].strip()
    m = CODE_RE.findall(reply)
    return bool(m) and m[-1] == expected


def c3(url, out, arm, specs, max_tokens):
    res = {"arm": arm, "max_tokens": max_tokens, "items": []}
    for sp in specs:
        label, rs = rows(sp)
        for row in rs:
            r = chat(url, row["input"], temperature=0, max_tokens=max_tokens, chat_template_kwargs={"enable_thinking": True})
            r.update(set=label, name=row["name"], expected=row["expected_output"])
            r["correct"] = score(r["content"], row["expected_output"])
            res["items"].append(r)
            print(arm, label, row["name"], r["prompt_tokens"], r["completion_tokens"], r["reasoning_tokens"], r["finish"], r["correct"], flush=True)
            json.dump(res, open(out, "w"), indent=1)


def c3cmp(fa, fb):
    A, B = json.load(open(fa)), json.load(open(fb))
    by = {i["name"]: i for i in B["items"]}
    print(f"# check 3 (thinking on): {A['arm']} against {B['arm']}, max_tokens {A['max_tokens']}\n")
    print("| set | n | finished (not length) A / B | correct A / B | ids identical | mean equal-prefix fraction | thinking seen A / B |\n|---|---|---|---|---|---|---|")
    ok = True
    for s in sorted({i["set"] for i in A["items"]}):
        xs = [i for i in A["items"] if i["set"] == s]
        ys = [by[i["name"]] for i in xs]
        fin = lambda L: sum(1 for i in L if i["finish"] != "length")   # noqa
        cor = lambda L: sum(1 for i in L if i["correct"])              # noqa
        same = sum(1 for x, y in zip(xs, ys) if x["ids"] and x["ids"] == y["ids"])
        fr = st.mean(lcp(x["ids"], y["ids"]) / max(len(x["ids"]), len(y["ids"]), 1) for x, y in zip(xs, ys))
        th = lambda L: sum(1 for i in L if (i["reasoning"] or i["reasoning_tokens"]))   # noqa
        proxy = [i["name"] for i in xs + ys if degenerate(i["ids"])]
        loops = [i["name"] for i in xs + ys if looping(i["ids"])]
        print(f"| {s} | {len(xs)} | {fin(xs)} / {fin(ys)} | {cor(xs)} / {cor(ys)} | {same} | {fr:.2f} | {th(xs)} / {th(ys)} |")
        if proxy:
            print(f"\n  pre-registered proxy (distinct ids / half of the 4-grams repeated) flagged: {proxy}; loops (a 16-gram 6 or more times): {loops}\n")
        if loops:
            ok = False
        ok &= th(xs) > 0 and abs(cor(xs) - cor(ys)) <= 2     # pass rule: thinking really on, accuracy within 2 items, nothing degenerate
    print("\n" + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def c4(url, out, arm, items_file):
    items = [json.loads(l) for l in open(items_file) if l.startswith("{")][:16]
    free = [f"Write about {t} in about two hundred words. Run id {i}." for i, t in enumerate(
        ["the history of the lighthouse", "how bread rises", "a walk through an old harbour town", "why the sky is blue",
         "the life of a beekeeper", "the rules of chess", "how a river forms a delta", "an old library at night"])]
    res = {"arm": arm, "items": []}
    for i, row in enumerate(items):
        r = chat(url, row["input"], temperature=0.7, seed=1000 + i, max_tokens=60, chat_template_kwargs={"enable_thinking": False})
        r.update(kind="answer", name=row["name"], expected=row["expected_output"], correct=score(r["content"], row["expected_output"]))
        res["items"].append(r)
    for i, p in enumerate(free):
        r = chat(url, p, temperature=0.7, seed=2000 + i, max_tokens=256, chat_template_kwargs={"enable_thinking": False})
        r.update(kind="free", name=f"free-{i}", expected="", correct=None)
        res["items"].append(r)
    json.dump(res, open(out, "w"), indent=1)
    print("c4 done", arm)


def c4l(url, out, arm, items_file):
    """Check 4, long prompts: 16 free-form requests (temperature 0.7, seed 3000 + i, 256 tokens) after different 4K prompts of the
    recent-window set (rows 16..31; the final question is replaced by a request for a summary), so the replay prefill and the decode
    window ring after it are exercised by a long sampled reply. Unique first lines: no prefix hit."""
    rows_ = [json.loads(l) for l in open(items_file) if l.startswith("{")][16:32]
    res = {"arm": arm, "items": []}
    for i, row in enumerate(rows_):
        p = row["input"]
        k = p.rindex("Question:")
        assert "Reply with only the code." in p[k:], row["name"]
        p = p[:k] + "Question: summarize the log above in about 150 words, naming one vault and its code."
        r = chat(url, p, temperature=0.7, seed=3000 + i, max_tokens=256, chat_template_kwargs={"enable_thinking": False})
        r.update(kind="freelong", name=row["name"], expected="", correct=None)
        res["items"].append(r)
        print(arm, row["name"], r["prompt_tokens"], r["completion_tokens"], r["finish"], flush=True)
    json.dump(res, open(out, "w"), indent=1)


def rep4(ids):
    g = [tuple(ids[i:i + 4]) for i in range(len(ids) - 3)]
    return 1 - len(set(g)) / len(g) if g else 0.0


def c4cmp(fa, fb):
    A, B = json.load(open(fa)), json.load(open(fb))
    print(f"# check 4 (temperature 0.7, seeds fixed per request): {A['arm']} against {B['arm']}\n")
    ok = True
    print("| kind | arm | n | correct | mean tokens | mean repeated-4-gram fraction | max token run | degenerate | finish=length |\n|---|---|---|---|---|---|---|---|---|")
    stats = {}
    for kind in ("answer", "free", "freelong"):
        for R in (A, B):
            L = [i for i in R["items"] if i["kind"] == kind]
            if not L:
                continue
            run = max((max(len(list(g)) for _, g in __import__("itertools").groupby(i["ids"])) if i["ids"] else 0) for i in L)
            deg = [i["name"] for i in L if (degenerate(i["ids"]) if kind != "answer" else (len(i["ids"]) == 0))]
            cor = sum(1 for i in L if i["correct"]) if kind == "answer" else "-"
            m = (st.mean(len(i["ids"]) for i in L), st.mean(rep4(i["ids"]) for i in L))
            stats[(kind, R["arm"])] = (cor, m, deg)
            print(f"| {kind} | {R['arm']} | {len(L)} | {cor} | {m[0]:.1f} | {m[1]:.3f} | {run} | {len(deg)} | {sum(1 for i in L if i['finish'] == 'length')} |")
            ok &= not deg and run <= 40
    hits = [i["name"] for R in (A, B) for i in R["items"] if i.get("cached")]
    print(f"\nprefix-cache hits in either arm (must be none): {hits}")
    ok &= not hits
    fk = "free" if ("free", A["arm"]) in stats else "freelong"
    ca, cb = (stats[("answer", A["arm"])][0], stats[("answer", B["arm"])][0]) if ("answer", A["arm"]) in stats else (0, 0)
    fa_, fb_ = stats[(fk, A["arm"])][1], stats[(fk, B["arm"])][1]
    same = sum(1 for x, y in zip(A["items"], B["items"]) if x["ids"] == y["ids"])
    print(f"\nidentical replies (same seed) {same} of {len(A['items'])}; correct {ca} vs {cb}; free-form length ratio {fa_[0] / max(fb_[0], 1e-9):.2f}, repeated-4-gram {fa_[1]:.3f} vs {fb_[1]:.3f}")
    ok &= abs(ca - cb) <= 2 and 0.75 <= fa_[0] / max(fb_[0], 1e-9) <= 1.33 and abs(fa_[1] - fb_[1]) <= 0.10
    print("\n" + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


# ------------------------------------------------------------------ check 2
def c2cmp(fa, fb, ma=None):
    A, B = json.load(open(fa)), json.load(open(fb))
    ca = {c["name"]: c for d in A["datasets"] for c in d["cases"]}
    cb = {c["name"]: c for d in B["datasets"] for c in d["cases"]}
    print("# check 2: question first, 32K\n\n| item | expected | A reply FINAL | B reply FINAL | same reply |\n|---|---|---|---|---|")
    n = right_a = right_b = same = bad = 0
    for k in ca:
        a, b = ca[k], cb.get(k)
        if not b:
            continue
        n += 1
        ea = a["expected_output"]
        fa_ = (FINAL_RE.findall(a["actual_output"] or "") or [""])[-1]
        fb_ = (FINAL_RE.findall(b["actual_output"] or "") or [""])[-1]
        ra, rb = fa_.strip() == ea[len("FINAL:"):].strip(), fb_.strip() == ea[len("FINAL:"):].strip()
        right_a += ra
        right_b += rb
        s = (a["actual_output"] or "") == (b["actual_output"] or "")
        same += s
        bad += (not (a["actual_output"] or "").strip()) + (not (b["actual_output"] or "").strip())
        print(f"| {k} | {ea} | {fa_} {'ok' if ra else 'WRONG'} | {fb_} {'ok' if rb else 'WRONG'} | {s} |")
    print(f"\nitems {n}; exact FINAL right: A {right_a}, B {right_b}; identical replies {same}; empty replies {bad}")
    print("\nDIFFERING REPLIES (A, then B; first 400 chars)\n")
    for k in ca:
        b = cb.get(k)
        if b and (ca[k]["actual_output"] or "") != (b["actual_output"] or ""):
            print(f"--- {k}\nA: {(ca[k]['actual_output'] or '')[:400]!r}\nB: {(b['actual_output'] or '')[:400]!r}\n")
    ok = bad == 0 and n >= 11 and right_a >= right_b - 2
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def select(final, step6, out12, out8):
    """12 items for check 2 (question first) and the first 8 of them for check 3 (question last, thinking on): two per kind of the
    frozen set, taken from the items the OFF arm AND the ON arm both answered right with the question last (step 6, exact FINAL)."""
    ex = json.load(open(step6))
    allrows = [json.loads(l) for l in open(final) if l.startswith("{")]
    groups = {}
    for r in allrows:
        if r["name"].endswith("32k-" + r["name"].rsplit("-", 1)[1]) or "-32k-" in r["name"]:
            k = "-".join(r["name"].split("-")[:2])
            if ex["off"].get(r["name"]) and ex["on"].get(r["name"]):
                groups.setdefault(k, []).append(r)
    order = ["hop2-L3", "hop3-L3", "agg_sum-L3", "agg_sum-L4", "agg_max-L3", "repo_sum-L4"]
    pick = []
    for rnd in range(2):
        for k in order:
            if k == "agg_sum-L3" and not groups.get(k):
                k = "agg_sum-L4"
            g = groups[k]
            pick.append(g[rnd])
    names = [p["name"] for p in pick]
    assert len(set(names)) == len(names) == 12, names
    tmp = out12 + ".src"
    with open(tmp, "w") as f:
        for p in pick:
            f.write(json.dumps(p, ensure_ascii=False, sort_keys=True) + "\n")
    qfirst(tmp, 12, out12)
    os.remove(tmp)
    with open(out8, "w") as f:
        for p in pick[:8]:
            f.write(json.dumps(p, ensure_ascii=False, sort_keys=True) + "\n")
    print("selected", names)


def qfirst(src, n, out):
    rows_ = [json.loads(l) for l in open(src) if l.startswith("{")][:int(n)]
    outp = []
    for r in rows_:
        p = r["input"]
        m = re.match(r"(Run id [^\n]*\.)\n\nBelow is a long document\. Read it carefully, then answer the question after it\.\n\n(<document>\n.*\n</document>)\n\n(Question: .*?)\n\n(Use only the document\..*)\Z", p, re.S)
        assert m, r["name"]
        head, doc, q, instr = m.groups()
        r2 = dict(r)
        r2["name"] = r["name"] + "-qfirst"
        r2["input"] = f"{head}\n\nFirst read the question, then the document below it, then answer.\n\n{q}\n\n{instr}\n\n{doc}"
        assert r2["input"].endswith("</document>") and r2["input"].index(q) < r2["input"].index("<document>")
        outp.append(r2)
    with open(out, "w") as f:
        for r in outp:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"wrote {len(outp)} question-first items to {out}")


def main(a):
    c = a[0]
    if c == "predict":
        os.environ.setdefault("TOKENIZER", "")
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(a[1])
        bos = tok.token_to_id("<｜begin▁of▁sentence｜>")
        ids = [bos] + tok.encode(open(DOC, encoding="utf-8").read(), add_special_tokens=False).ids
        aa, bs = variants(ids)
        print(f"corpus {len(ids)} tokens; A = {len(aa)} tokens (chunks 2048 2048 {len(aa) - 4096}, floor {floor(len(aa))})")
        print("| B | tokens | default | REPLAY=0 | REPLAY_SKIP=0 |\n|---|---|---|---|---|")
        for sect, cancel in (("A completed (max_tokens 1)", None), ("A cancelled, state 4096", 4096)):
            p = [predict(aa, bs, *m, cancel_at=cancel) for m in ((True, True), (False, False), (True, False))]
            for n in bs:
                if cancel and not (n.startswith("B3")):
                    continue
                print(f"| {n} ({sect}) | {len(bs[n])} | {p[0][n]} | {p[1][n]} | {p[2][n]} |")
        return 0
    if c == "c1":
        return c1(a[1], a[2], a[3], a[4])
    if c == "c1cmp":
        return c1cmp(a[1:])
    if c == "c3":
        return c3(a[1], a[2], a[3], a[5:], int(a[4]))
    if c == "c3cmp":
        return c3cmp(a[1], a[2])
    if c == "c4":
        return c4(a[1], a[2], a[3], a[4])
    if c == "c4l":
        return c4l(a[1], a[2], a[3], a[4])
    if c == "c4cmp":
        return c4cmp(a[1], a[2])
    if c == "c2cmp":
        return c2cmp(a[1], a[2])
    if c == "select":
        return select(a[1], a[2], a[3], a[4])
    if c == "qfirst":
        return qfirst(a[1], a[2], a[3])
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]) or 0)

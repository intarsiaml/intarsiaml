#!/usr/bin/env python3
"""Goldens and noise measurements from MiaAI-Lab's TensorFold-Zig server (DeepSeek-V4.1-Flash EXL3 2.9 bpw, 2 x Spark).

Why this tool exists: the TensorFold server REFUSES log-probs today (HTTP 400 "logprobs are not supported by this model
or backend", zig/src/server/fields.zig `probabilityOptions`), `echo` / `prompt_logprobs` are never read, and prefill computes
the head for the LAST prompt position only. So `probe` tells which of the three ways below the connected server offers, `topk`
uses the best one, and `tokens` (greedy token ids via `return_token_ids`) works on the unpatched server. See
docs/research/2026-10-31-dsv41-golden-plan.md. This tool never starts or stops anything; it only sends HTTP requests, and a
request to a fleet server must run inside the fleetq slot of the takedown window.

Modes
    probe                  GET /health, /v1/models, /metrics; POST /v1/completions with logprobs / prompt_logprobs / echo;
                           prints a verdict and which top-k strategy is possible.
    topk   --out F         [--ids-from GOLDEN] `intarsia-golden-topk-v1` file (docs/accuracy.md), strategies in order of preference:
                             prompt_logprobs  one request per prompt, vLLM style (like tools/golden_vllm.py)
                             echo             one request per prompt, OpenAI style `echo` + `logprobs`
                             continuation     SLOW FALLBACK, generation-time log-probs only: rows exist ONLY for the positions
                                              that generate the `continue` tokens; every other row is padded (-1 / -1e30) and the
                                              metadata key `rows_valid` lists the valid row ranges. NOT a full teacher-forced
                                              golden: do not hand it to `intarsia accuracy` (it knows no `rows_valid`).
    tokens --out F         greedy continuation token ids (temperature 0, `draft: false`) for the same prompt set. Works today.
    compare A B            token files: first divergence and top-1 (leading-match) agreement per prompt.
    compare-topk A B       two topk files over the same ids: KL(A||B), KL(B||A), top-1 agreement; applies an absolute KL floor.

Format `intarsia-golden-tokens-v1` (safetensors, small): per prompt i
    ids_i   I32 [n_i]   prompt token ids (as the server's /tokenize gave them, or tiled corpus ids for the long prompts)
    cont_i  I32 [c_i]   the greedy continuation ids (c_i < the requested count only when the server stopped early)
  metadata (all strings): format, engine, engine_version, server, served_model, max_model_len, sampling (JSON), launch_flags,
  note, server_meta (JSON of /health, /v1/models and tensorfold lines of /metrics), prompt_files, skipped (JSON),
  prompts (JSON list {name, kind, n_tokens, n_prompt, continuation, text}), built_unix.

Format `intarsia-golden-topk-v1`: exactly as tools/golden_vllm.py writes it (ids_i, topk_ids_i, topk_logprobs_i, and the
metadata of docs/accuracy.md) plus `source: tensorfold`, `strategy`, `cached_tokens` (JSON, per prompt, what the server
says it served from its prompt cache) and, for the continuation fallback only, `coverage` and `rows_valid`.

Long prompts: `--long-prompts 32768,65536` builds one prompt per length from tools/prompts/corpus_code.txt (tokenised once
by the server, the ids tiled and trimmed to exactly N). Nothing big is written to disk except the golden itself.

Only the standard library, numpy and safetensors are needed:
    uv run --with numpy --with safetensors python tools/dsv41/golden_tf.py probe --base-url http://spark1:8888/v1
"""
import argparse
import json
import os
import pathlib
import re
import sys
import time
import urllib.error
import urllib.request

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

TOOLS = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS))
import goldenlib as gl  # noqa: E402

PAD_LP = -1e30
FORMAT_TOKENS = "intarsia-golden-tokens-v1"
DEFAULT_PROMPTS = [str(TOOLS / "prompts" / "golden.jsonl"), str(TOOLS / "prompts" / "golden_long.jsonl")]
KL_FLOOR = 1e-3


# ---------------------------------------------------------------------------------------------------- http
class Server:
    def __init__(self, base, model, key=None, timeout=3600, draft_field=True):
        self.base = base.rstrip("/")
        self.root = self.base[:-3] if self.base.endswith("/v1") else self.base
        self.model, self.key, self.timeout, self.draft_field = model, key, timeout, draft_field
        self._max_len = None

    def _req(self, url, body=None, timeout=None):
        h = {"Content-Type": "application/json"}
        if self.key:
            h["Authorization"] = "Bearer " + self.key
        req = urllib.request.Request(url, None if body is None else json.dumps(body).encode(), h)
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
                return r.status, r.read().decode()
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode(errors="replace")
        except (urllib.error.URLError, OSError) as e:
            return 0, str(e)

    def get(self, path, root=True, timeout=30):
        return self._req((self.root if root else self.base) + path, None, timeout)

    def post(self, path, body, root=False, timeout=None):
        """(status, parsed json or raw text). Non-2xx is returned, not raised."""
        st, txt = self._req((self.root if root else self.base) + path, body, timeout)
        try:
            return st, json.loads(txt)
        except ValueError:
            return st, txt

    def post_ok(self, path, body, what, root=False):
        st, r = self.post(path, body, root)
        if st != 200:
            raise SystemExit(f"{what}: HTTP {st}: {str(r)[:400]}")
        return r

    def tokenize(self, text, special=True):
        r = self.post_ok("/tokenize", {"model": self.model, "prompt": text, "add_special_tokens": special}, "/tokenize", root=True)
        if r.get("max_model_len"):
            self._max_len = int(r["max_model_len"])
        return [int(t) for t in r["tokens"]]

    def max_len(self):
        """The window a request may use (prompt + reply). TensorFold's /tokenize reports it as `max_model_len`."""
        if self._max_len is None:
            self.tokenize("hi")
        if self._max_len is None:
            st, r = self.get("/models", root=False)
            try:
                d = json.loads(r)["data"][0]
                self._max_len = int(d.get("max_model_len") or d.get("context_window") or 0) or None
            except Exception:
                pass
        return self._max_len

    def completion(self, ids, n, extra=None):
        body = {"model": self.model, "prompt": ids, "max_tokens": n, "temperature": 0}
        if self.draft_field:
            body["draft"] = False  # TensorFold: no DSpark drafts for this request (drafted replies are exact, but keep the path plain)
        body.update(extra or {})
        return self.post("/completions", body)


def token_ids_of(resp):
    """Generated token ids of a completions reply: TensorFold's `tensorfold.token_ids`, else a top-level / choice `token_ids`."""
    ch = resp["choices"][0]
    for src in ((resp.get("tensorfold") or {}).get("token_ids"), ch.get("token_ids"), resp.get("token_ids")):
        if src is not None:
            return [int(t) for t in src]
    toks = (ch.get("logprobs") or {}).get("tokens") or []
    out = []
    for t in toks:
        m = re.fullmatch(r"token_id:(\d+)", t)
        if not m:
            raise SystemExit("cannot read generated token ids (no tensorfold.token_ids / token_ids / token_id: tokens); "
                             "send return_token_ids on a server that supports it")
        out.append(int(m.group(1)))
    return out


def cached_tokens_of(resp):
    return int(((resp.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens") or 0)


# ---------------------------------------------------------------------------------------------------- prompts
class Prompts:
    def __init__(self, srv, files, long_lengths, long_continue):
        self.srv = srv
        self.list = gl.load_prompts(files) if files else []
        self.files = list(files)
        for n in long_lengths:
            self.list.append({"name": f"long_code_{n}", "kind": "raw", "long_tokens": n, "continue": long_continue})
        names = [p["name"] for p in self.list]
        if len(set(names)) != len(names):
            raise SystemExit(f"duplicate prompt names {names}")
        self._corpus = None
        self.skipped = []

    def ids(self, p):
        """(ids, description text) of a prompt. Unlike goldenlib.prompt_ids there is no 16384 cap (the server's window is the cap)."""
        if "long_tokens" in p:
            n = int(p["long_tokens"])
            if self._corpus is None:
                text = (gl.PROMPT_DIR / "corpus_code.txt").read_text()
                self._corpus = (self.srv.tokenize(text, True), self.srv.tokenize(text, False))
            first, body = self._corpus
            ids = list(first)
            while len(ids) < n:  # tiles join at token level: not re-tokenised across joins; the golden stores the ids
                ids += body
            return ids[:n], f"tools/prompts/corpus_code.txt tiled to exactly {n} tokens (first copy with special tokens)"
        text = gl.prompt_text(p, "chatml")
        ids = self.srv.tokenize(text, True)
        if "file" in p:
            skip = int(p.get("skip_tokens", 0))
            ids = ids[skip:skip + int(p["max_tokens"])]
        if len(ids) < gl.MIN_TOK:
            raise SystemExit(f"prompt {p['name']} has {len(ids)} tokens, want >= {gl.MIN_TOK}")
        return ids, text

    def fitting(self, extra_tokens=2, default_continue=0):
        """Yield (p, ids, text, want) for prompts that fit the window; print SKIPPED for the others (like golden_vllm)."""
        max_len = self.srv.max_len()
        print(f"server max_model_len={max_len}", flush=True)
        for p in self.list:
            ids, text = self.ids(p)
            want = int(p.get("continue", default_continue))
            if max_len is not None and len(ids) + want + extra_tokens > max_len:
                print(f"SKIPPED {p['name']}: {len(ids)}+{want} tokens do not fit max_model_len {max_len}", flush=True)
                self.skipped.append(p["name"])
                continue
            yield p, ids, text, want


def add_prompt_args(ap):
    ap.add_argument("--prompts", nargs="*", default=DEFAULT_PROMPTS,
                    help="prompt-set files (default: golden.jsonl, golden_long.jsonl; `--prompts` with no file = only --long-prompts)")
    ap.add_argument("--long-prompts", default="", metavar="N,N", help="also build prompts of exactly N tokens from corpus_code.txt, e.g. 32768,65536")
    ap.add_argument("--long-continue", type=int, default=16, help="greedy continuation tokens appended to each long prompt (default 16)")


def add_server_args(ap):
    ap.add_argument("--base-url", default="http://127.0.0.1:8888/v1", help="OpenAI-compatible base URL (the recipe serves port 8888)")
    ap.add_argument("--model", default="DeepSeek-V4.1-Flash-1017", help="served model name (recipe SERVED_NAME)")
    ap.add_argument("--api-key-file", default=None, help="file with the bearer key (recipe API_KEY=1 keeps it in $STATE_DIR/api-key); or env TF_API_KEY")
    ap.add_argument("--timeout", type=int, default=3600, help="per request seconds (a 64K prefill takes ~30 s, a 512K one ~5 min)")
    ap.add_argument("--no-draft-field", action="store_true", help="do not send `draft: false` (use against a non-TensorFold server that rejects unknown fields)")
    ap.add_argument("--note", default="", help="free text for the metadata: launch flags, recipe commit, TF_REF + patch hash, env of both ranks, KV dtype")


def make_server(a):
    key = os.environ.get("TF_API_KEY")
    if a.api_key_file:
        key = pathlib.Path(a.api_key_file).read_text().strip()
    return Server(a.base_url, a.model, key, a.timeout, not a.no_draft_field)


def longs(a):
    return [int(x) for x in a.long_prompts.split(",") if x.strip()]


def server_meta(srv):
    """What the API shows about the server. Version / patch hash are looked for in /metrics lines that mention them."""
    out = {}
    for name, path, root in (("health", "/health", True), ("models", "/models", False)):
        st, txt = srv.get(path, root)
        out[name] = {"status": st, "body": txt[:2000]}
    st, txt = srv.get("/metrics")
    lines = [l for l in txt.splitlines() if st == 200 and not l.startswith("#") and re.search(r"info|version|patch|engine|build|cache_config|cached", l, re.I)]
    out["metrics_lines"] = lines[:40]
    return out


# ---------------------------------------------------------------------------------------------------- probe
def probe(srv, verbose=True):
    """Returns {"logprobs", "prompt_logprobs", "echo", "details"}; prints the verdict when verbose."""
    say = print if verbose else (lambda *x, **k: None)
    caps = {"logprobs": False, "prompt_logprobs": False, "echo": False, "details": {}}
    st, txt = srv.get("/health")
    say(f"GET /health          -> {st} {txt[:200]!r}")
    if st != 200:
        raise SystemExit("server not healthy; nothing to probe")
    st, txt = srv.get("/models", root=False)
    say(f"GET /v1/models       -> {st} {txt[:300]!r}")
    st, txt = srv.get("/metrics")
    say(f"GET /metrics         -> {st} ({len(txt)} bytes)")
    ids = srv.tokenize("The quick brown fox jumps over the lazy dog.")
    say(f"/tokenize            -> {len(ids)} ids, max_model_len={srv.max_len()}")

    def show(label, st, r):
        msg = r["error"].get("message") if isinstance(r, dict) and isinstance(r.get("error"), dict) else str(r)[:200]
        say(f"{label:<20} -> {st} {msg if st != 200 else 'ok'}")
        return msg

    # plain generation + token ids (what `tokens` needs)
    st, r = srv.completion(ids, 3, {"return_token_ids": True})
    try:
        got = token_ids_of(r) if st == 200 and isinstance(r, dict) else None
    except SystemExit:
        got = None
    caps["details"]["return_token_ids"] = got is not None
    show("plain + token ids", st, r)
    say(f"  generated ids      : {got}")

    # generation-time log-probs: legacy int, then bool (TensorFold's field check wants a boolean)
    for label, extra in (("logprobs=1 (int)", {"logprobs": 1}), ("logprobs=true", {"logprobs": True, "top_logprobs": 1})):
        st, r = srv.completion(ids, 2, extra)
        msg = show(label, st, r)
        caps["details"][label] = {"status": st, "message": msg if st != 200 else ""}
        if st == 200 and isinstance(r, dict) and (r["choices"][0].get("logprobs") or None):
            caps["logprobs"] = True
            break
    # prompt_logprobs (vLLM): supported only when the reply carries a list with one entry per prompt token
    st, r = srv.completion(ids, 1, {"prompt_logprobs": 1})
    show("prompt_logprobs=1", st, r)
    pl = r["choices"][0].get("prompt_logprobs") if st == 200 and isinstance(r, dict) else None
    caps["prompt_logprobs"] = isinstance(pl, list) and len(pl) >= len(ids)
    if st == 200 and not caps["prompt_logprobs"]:
        say("  field accepted but IGNORED (no prompt_logprobs in the reply)")
    # echo + logprobs (OpenAI)
    if caps["logprobs"]:
        st, r = srv.completion(ids, 1, {"echo": True, "logprobs": 1, "return_tokens_as_token_ids": True})
        show("echo+logprobs", st, r)
        lp = (r["choices"][0].get("logprobs") or {}) if st == 200 and isinstance(r, dict) else {}
        caps["echo"] = len(lp.get("top_logprobs") or []) >= len(ids)
    else:
        say("echo+logprobs       -> not testable: logprobs themselves are refused")

    say("")
    if caps["prompt_logprobs"]:
        say("VERDICT: SUPPORTED, strategy prompt_logprobs (one request per prompt, teacher-forced rows for every position).")
    elif caps["echo"]:
        say("VERDICT: SUPPORTED, strategy echo (one request per prompt, rows for every position).")
    elif caps["logprobs"]:
        say("VERDICT: PARTIAL. Generation-time log-probs only: `topk --strategy continuation` covers the continuation positions only.")
    else:
        why = caps["details"].get("logprobs=true", {}).get("message") or caps["details"].get("logprobs=1 (int)", {}).get("message")
        say(f"VERDICT: REFUSED. The server returns no log-probs ({why!r}). Use `tokens` / `compare`; "
            "top-k goldens need the engine patch (docs/research/2026-10-31-dsv41-golden-plan.md).")
    return caps


def best_strategy(caps):
    return "prompt_logprobs" if caps["prompt_logprobs"] else "echo" if caps["echo"] else "continuation" if caps["logprobs"] else None


# ---------------------------------------------------------------------------------------------------- topk
def entry_items(entry, K):
    """[(id, logprob)] best first, at most K, from one position's entry. Handles vLLM prompt_logprobs ({"123": {logprob, rank}}),
    OpenAI top_logprobs ({"token_id:123": -1.5}) and chat `top_logprobs` lists ([{token: "token_id:123", logprob}])."""
    items = []
    pairs = [(e.get("token"), e) for e in entry] if isinstance(entry, list) else list(entry.items())
    for k, v in pairs:
        lp, rank = (v["logprob"], v.get("rank", 0)) if isinstance(v, dict) else (v, 0)
        m = re.fullmatch(r"(?:token_id:)?(\d+)", str(k))
        if not m:
            raise SystemExit(f"top-logprob key {k!r} is not a token id; the server needs return_tokens_as_token_ids (or ids as keys)")
        if rank <= K:  # vLLM adds the actual prompt token with rank > K when it is outside the top-K
            items.append((int(m.group(1)), float(lp)))
    items.sort(key=lambda x: -x[1])
    return items[:K]


def fill_rows(tid, tlp, rows, start, K):
    for t, entry in enumerate(rows, start):
        for j, (k, lp) in enumerate(entry_items(entry, K)):
            tid[t, j], tlp[t, j] = k, lp


def topk_rows(srv, strategy, ids, want, K, name):
    """(ids_full, tid [n,K], tlp [n,K], valid_range, cached_tokens). Row t = distribution over token t+1 after ids[..=t]."""
    n_prompt = len(ids)
    if strategy == "continuation":
        if want < 1:
            return None
        st, r = srv.completion(ids, want, {"logprobs": K, "min_tokens": want, "ignore_eos": True, "return_token_ids": True,
                                           "return_tokens_as_token_ids": True})
        if st != 200:
            raise SystemExit(f"{name}: HTTP {st}: {str(r)[:300]}")
        cont = token_ids_of(r)
        if len(cont) != want:
            raise SystemExit(f"{name}: asked for {want} continuation tokens, got {len(cont)}")
        lp = r["choices"][0]["logprobs"]
        rows = lp.get("top_logprobs") or (lp.get("content") and [c["top_logprobs"] for c in lp["content"]])
        if not rows or len(rows) != want:
            raise SystemExit(f"{name}: no per-token top_logprobs in the reply")
        full = ids + cont
        n = len(full)
        tid, tlp = np.full((n, K), -1, np.int32), np.full((n, K), PAD_LP, np.float32)
        fill_rows(tid, tlp, rows, n_prompt - 1, K)  # generation i predicts cont[i] at position n_prompt-1+i
        return full, tid, tlp, (n_prompt - 1, n_prompt - 1 + want), cached_tokens_of(r)
    # full-coverage strategies: continuation first (plain greedy), then ONE scoring request on ids + cont + dummy token
    full = list(ids)
    if want:
        st, r = srv.completion(ids, want, {"min_tokens": want, "ignore_eos": True, "return_token_ids": True})
        if st != 200:
            raise SystemExit(f"{name}: continuation HTTP {st}: {str(r)[:300]}")
        cont = token_ids_of(r)
        if len(cont) != want:
            raise SystemExit(f"{name}: asked for {want} continuation tokens, got {len(cont)}")
        full += cont
    n = len(full)
    body = {"prompt_logprobs": K} if strategy == "prompt_logprobs" else {"echo": True, "logprobs": K, "return_tokens_as_token_ids": True}
    st, r = srv.completion(full + [full[-1]], 1, body)  # dummy last token: gives the row of the final real position
    if st != 200:
        raise SystemExit(f"{name}: HTTP {st}: {str(r)[:300]}")
    ch = r["choices"][0]
    pl = ch.get("prompt_logprobs") if strategy == "prompt_logprobs" else (ch.get("logprobs") or {}).get("top_logprobs")
    if pl is None or len(pl) < n + 1 or pl[0] is not None:
        raise SystemExit(f"{name}: expected >= {n + 1} entries with a null first one, got {None if pl is None else len(pl)}")
    tid, tlp = np.full((n, K), -1, np.int32), np.full((n, K), PAD_LP, np.float32)
    fill_rows(tid, tlp, pl[1:n + 1], 0, K)
    if (tid[:, 0] < 0).any():
        raise SystemExit(f"{name}: empty log-prob entry at row {int(np.argmax(tid[:, 0] < 0))}")
    return full, tid, tlp, (0, n), cached_tokens_of(r)


def cmd_topk(a):
    srv = make_server(a)
    caps = probe(srv, verbose=False)
    strategy = a.strategy if a.strategy != "auto" else best_strategy(caps)
    if strategy is None:
        raise SystemExit("the server refuses log-probs (run `probe`): no top-k golden can be built. Use `tokens` for the token-id noise proxy.")
    if strategy == "continuation":
        print("WARNING: continuation strategy: rows exist only for the continuation positions (metadata rows_valid); "
              "this is not a teacher-forced golden", flush=True)
    K = a.topk
    ps = Prompts(srv, a.prompts, longs(a), a.long_continue)
    tensors, meta_prompts, valid, cached = {}, [], [], []
    t0 = time.time()
    if a.ids_from:  # re-score the exact ids (prompt + recorded continuation) of an existing golden: a second config on identical positions
        if strategy == "continuation":
            raise SystemExit("--ids-from needs a full-coverage strategy (prompt_logprobs / echo)")
        src = safe_open(a.ids_from, "np")
        src_meta = json.loads((src.metadata() or {})["prompts"])
        items = [({"name": m["name"], "kind": m.get("kind", "raw"), "_meta": m}, [int(t) for t in src.get_tensor(f"ids_{j}")],
                  m.get("text", ""), 0) for j, m in enumerate(src_meta)]
        a.prompts = [f"ids from {pathlib.Path(a.ids_from).name}"]
    else:
        items = ps.fitting()
    for p, ids, text, want in items:
        if a.ids_from and srv.max_len() is not None and len(ids) + 2 > srv.max_len():
            raise SystemExit(f"{p['name']}: {len(ids)} tokens do not fit the server's max_model_len {srv.max_len()}")
        st, _ = srv.get("/health")
        if st != 200:
            raise SystemExit(f"/health {st} before {p['name']}")
        res = topk_rows(srv, strategy, ids, want, K, p["name"])
        if res is None:
            print(f"SKIPPED {p['name']}: continuation strategy needs a `continue` count", flush=True)
            continue
        full, tid, tlp, rng, ct = res
        i = len(meta_prompts)
        tensors[f"ids_{i}"], tensors[f"topk_ids_{i}"], tensors[f"topk_logprobs_{i}"] = np.array(full, np.int32), tid, tlp
        meta_prompts.append(p["_meta"] if "_meta" in p else gl.meta_prompt(p, text, len(full), len(ids), gl.reuse_split(p, ids, srv.tokenize)))
        valid.append(list(rng))
        cached.append(ct)
        warn = f"  CACHED {ct} prompt tokens (start the server with --prompt-cache-gib 0)" if ct else ""
        print(f"[{i}] {p['name']:<18} {len(full):6d} tokens ({len(ids)}+{len(full) - len(ids)}) rows {rng[0]}..{rng[1]} "
              f"top1_last={int(tid[rng[1] - 1, 0])} lp={float(tlp[rng[1] - 1, 0]):.3f} {time.time() - t0:.0f}s{warn}", flush=True)
    if not tensors:
        raise SystemExit("no prompt was built (all skipped?)")
    meta = {
        "format": gl.FORMAT_TOPK, "model": a.model, "model_revision": "not observable (see note)", "source": "tensorfold",
        "engine": "tensorfold-native", "engine_version": "not observable through the API (see server_meta; pass --note with TF_REF + patch hash)",
        "server": a.base_url, "served_model": a.model, "dtype": "server", "device": "server",
        "kv_dtype": "unknown (TensorFold; recipe docs: compressed FP4 KV)",
        "cache_config_info": "{}", "max_model_len": str(srv.max_len()),
        "launch_flags": "not observable through the API" + (f"; note: {a.note}" if a.note else ""),
        "template": "chatml", "add_special_tokens": "true", "strategy": strategy,
        "sampling": json.dumps({strategy: K, "temperature": 0, "draft": False if srv.draft_field else "not sent",
                                "continuation": "greedy, ignore_eos, from the same server",
                                "prefix_cache": "uncontrolled unless the server runs --prompt-cache-gib 0; see cached_tokens"}),
        "cached_tokens": json.dumps(cached), "server_meta": json.dumps(server_meta(srv)),
        "prompt_files": json.dumps([pathlib.Path(x).name for x in a.prompts] + [f"long:{n}" for n in longs(a)]),
        "ids_from": a.ids_from or "", "skipped": json.dumps(ps.skipped), "topk": str(K), "prompts": json.dumps(meta_prompts),
        "built_unix": str(int(time.time())),
        "note": "row t = distribution over token t+1 given ids[..=t]; logprobs are full-vocab as reported by the server",
    }
    if strategy == "continuation":
        meta["coverage"] = "continuation-only"
        meta["rows_valid"] = json.dumps(valid)  # per prompt [first_row, end_row); other rows are padded
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out), metadata=meta)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB, {len(meta_prompts)} prompts) in {time.time() - t0:.0f}s")


# ---------------------------------------------------------------------------------------------------- tokens
def cmd_tokens(a):
    srv = make_server(a)
    ps = Prompts(srv, a.prompts, longs(a), a.long_continue)
    tensors, mp = {}, []
    t0 = time.time()
    for p, ids, text, want in ps.fitting(default_continue=a.gen_tokens):
        st, _ = srv.get("/health")
        if st != 200:
            raise SystemExit(f"/health {st} before {p['name']}")
        st, r = srv.completion(ids, want, {"min_tokens": want, "ignore_eos": True, "return_token_ids": True})
        if st != 200:
            raise SystemExit(f"{p['name']}: HTTP {st}: {str(r)[:300]}")
        cont = token_ids_of(r)
        i = len(mp)
        tensors[f"ids_{i}"], tensors[f"cont_{i}"] = np.array(ids, np.int32), np.array(cont, np.int32)
        mp.append(gl.meta_prompt(p, text, len(ids) + len(cont), len(ids), None))
        print(f"[{i}] {p['name']:<18} {len(ids):6d} prompt tokens, {len(cont)} generated, cached={cached_tokens_of(r)} "
              f"head={cont[:6]} {time.time() - t0:.0f}s", flush=True)
    if not tensors:
        raise SystemExit("no prompt was run")
    meta = {
        "format": FORMAT_TOKENS, "engine": "tensorfold-native", "engine_version": "not observable through the API (pass --note)",
        "server": a.base_url, "served_model": a.model, "max_model_len": str(srv.max_len()),
        "sampling": json.dumps({"temperature": 0, "draft": False if srv.draft_field else "not sent", "ignore_eos": True,
                                "return_token_ids": True}),
        "launch_flags": "not observable through the API" + (f"; note: {a.note}" if a.note else ""),
        "note": a.note, "server_meta": json.dumps(server_meta(srv)),
        "prompt_files": json.dumps([pathlib.Path(x).name for x in a.prompts] + [f"long:{n}" for n in longs(a)]),
        "skipped": json.dumps(ps.skipped), "prompts": json.dumps(mp), "built_unix": str(int(time.time())),
    }
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out), metadata=meta)
    print(f"wrote {out} ({out.stat().st_size / 1e3:.1f} kB, {len(mp)} prompts) in {time.time() - t0:.0f}s")


def load_tokens(path):
    f = safe_open(path, "np")
    m = f.metadata() or {}
    if m.get("format") != FORMAT_TOKENS:
        raise SystemExit(f"{path}: format {m.get('format')!r}, want {FORMAT_TOKENS}")
    names = [p["name"] for p in json.loads(m["prompts"])]
    return {nm: (f.get_tensor(f"ids_{i}"), f.get_tensor(f"cont_{i}")) for i, nm in enumerate(names)}, m


def cmd_compare(a):
    A, _ = load_tokens(a.A)
    B, _ = load_tokens(a.B)
    tot = agree = 0
    diverged = []
    for nm in A:
        if nm not in B:
            print(f"{nm:<18} only in A")
            continue
        (ia, ca), (ib, cb) = A[nm], B[nm]
        if not np.array_equal(ia, ib):
            print(f"{nm:<18} prompt ids differ ({len(ia)} vs {len(ib)}): not comparable")
            continue
        n = min(len(ca), len(cb))
        neq = np.nonzero(ca[:n] != cb[:n])[0]
        first = int(neq[0]) if len(neq) else n
        same = len(ca) == len(cb) and not len(neq)
        pos = float((ca[:n] == cb[:n]).mean()) if n else 1.0
        what = "IDENTICAL" if same else f"first divergence at continuation token {first}"
        print(f"{nm:<18} n_prompt={len(ia):6d} cont={len(ca)}/{len(cb)}  {what}  leading-match {first}/{n}  "
              f"positionwise-equal {100 * pos:.1f}%")
        tot += n
        agree += first
        if not same:
            diverged.append(nm)
    print(f"OVERALL leading-match {agree}/{tot} ({100 * agree / max(tot, 1):.2f}%)  prompts diverged: {len(diverged)} {diverged}")
    print("NOTE: after the first divergence the two continuations are no longer on the same text, so only the leading-match "
          "fraction is a top-1 agreement of the same positions; positionwise-equal is shown for orientation. Not a KL.")
    if a.expect_identical and diverged:
        raise SystemExit(1)


# ---------------------------------------------------------------------------------------------------- compare-topk
def rowwise_kl(ia, la, ib, lb):
    """KL(A||B) per row on A's support; B's log-prob where B lists the token, else B's K-th log-prob (a bound, as golden_cmp.py).
    The support mass is renormalised (sum over A's listed tokens)."""
    la, lb = la.astype(np.float64), lb.astype(np.float64)
    floor = np.where(lb > -1e29, lb, np.inf).min(axis=1)
    floor = np.where(np.isfinite(floor), floor, -50.0)
    eq = (ia[:, :, None] == ib[:, None, :]) & (ib[:, None, :] >= 0)
    has = eq.any(axis=2)
    lbv = np.where(has, np.where(eq, lb[:, None, :], -np.inf).max(axis=2), floor[:, None])
    mask = ia >= 0
    pa = np.where(mask, np.exp(la), 0.0)
    num = (pa * np.where(mask, la - lbv, 0.0)).sum(axis=1)
    return num / np.maximum(pa.sum(axis=1), 1e-300)


def cmd_compare_topk(a):
    fa, fb = safe_open(a.A, "np"), safe_open(a.B, "np")
    ma, mb = fa.metadata() or {}, fb.metadata() or {}
    for m, p in ((ma, a.A), (mb, a.B)):
        if m.get("format") != gl.FORMAT_TOPK:
            raise SystemExit(f"{p}: format {m.get('format')!r}, want {gl.FORMAT_TOPK}")
    pa_, pb_ = json.loads(ma["prompts"]), json.loads(mb["prompts"])
    nb = {p["name"]: j for j, p in enumerate(pb_)}
    va, vb = json.loads(ma.get("rows_valid", "null")), json.loads(mb.get("rows_valid", "null"))
    allkl, allkl_r, a1, n_all = [], [], 0, 0
    for j, p in enumerate(pa_):
        k = nb.get(p["name"])
        if k is None:
            print(f"{p['name']:<18} only in A")
            continue
        ids_a, ids_b = fa.get_tensor(f"ids_{j}"), fb.get_tensor(f"ids_{k}")
        if not np.array_equal(ids_a, ids_b):
            print(f"{p['name']:<18} ids differ ({len(ids_a)} vs {len(ids_b)}): not comparable (both runs must score the same ids; "
                  "use the `tokens` files first to see whether the continuations agree)")
            continue
        ia, ib = fa.get_tensor(f"topk_ids_{j}"), fb.get_tensor(f"topk_ids_{k}")
        la, lb = fa.get_tensor(f"topk_logprobs_{j}"), fb.get_tensor(f"topk_logprobs_{k}")
        ok = (ia[:, 0] >= 0) & (ib[:, 0] >= 0)  # padded rows of the continuation fallback are skipped
        for v, idx in ((va, j), (vb, k)):
            if v:
                r = np.zeros(len(ok), bool)
                r[v[idx][0]:v[idx][1]] = True
                ok &= r
        if not ok.any():
            print(f"{p['name']:<18} no common valid rows")
            continue
        kl = rowwise_kl(ia[ok], la[ok], ib[ok], lb[ok])
        klr = rowwise_kl(ib[ok], lb[ok], ia[ok], la[ok])
        ag = int((ia[ok, 0] == ib[ok, 0]).sum())
        n = int(ok.sum())
        print(f"{p['name']:<18} n={n:6d} top1-agree {100 * ag / n:7.3f}%  KL(A||B) mean {kl.mean():.3e} p99 {np.percentile(kl, 99):.3e} "
              f"max {kl.max():.3e}  KL(B||A) mean {klr.mean():.3e}")
        allkl.append(kl)
        allkl_r.append(klr)
        a1 += ag
        n_all += n
    if not n_all:
        raise SystemExit("nothing compared")
    kl, klr = np.concatenate(allkl), np.concatenate(allkl_r)
    sym = 0.5 * (kl.mean() + klr.mean())
    print(f"OVERALL n={n_all} top1-agree {100 * a1 / n_all:.3f}%  mean KL(A||B) {kl.mean():.3e}  mean KL(B||A) {klr.mean():.3e}  "
          f"sym {sym:.3e}  disagree-rows {n_all - a1}")
    if kl.mean() == 0.0 and a1 == n_all:
        print("RESULT: the two files are numerically identical on these rows (bit-deterministic pair, or the same run twice).")
    print(f"reference-noise budget for gating a port = max(measured sym KL {sym:.3e}, absolute floor {a.floor:.0e}) = "
          f"{max(sym, a.floor):.3e}  (plan: 2x this, docs/plans/2026-10-31-dsv41-unpark.md phase 3)")
    print("CAVEAT: KL is on A's top-K support with B floored at its K-th log-prob (a bound, not exact); fp rounding of the "
          "reported log-probs is not renormalised away here.")


# ---------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe", help="detect what the server supports")
    add_server_args(p)
    p = sub.add_parser("topk", help="build an intarsia-golden-topk-v1 file")
    add_server_args(p)
    add_prompt_args(p)
    p.add_argument("--out", required=True)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--strategy", choices=["auto", "prompt_logprobs", "echo", "continuation"], default="auto")
    p.add_argument("--ids-from", default=None, metavar="GOLDEN",
                   help="score the exact ids (prompt + recorded continuation) of an existing topk golden instead of building prompts; "
                        "use it for every config after the first so all files cover identical positions (as golden_vllm.py)")
    p = sub.add_parser("tokens", help="greedy continuation token ids (works on the unpatched server)")
    add_server_args(p)
    add_prompt_args(p)
    p.add_argument("--out", required=True)
    p.add_argument("--gen-tokens", type=int, default=64, help="continuation length for prompts without a `continue` key (default 64)")
    p = sub.add_parser("compare", help="compare two token files")
    p.add_argument("A")
    p.add_argument("B")
    p.add_argument("--expect-identical", action="store_true", help="exit 1 when any prompt diverges (determinism check)")
    p = sub.add_parser("compare-topk", help="compare two topk files")
    p.add_argument("A")
    p.add_argument("B")
    p.add_argument("--floor", type=float, default=KL_FLOOR, help=f"absolute KL floor of the noise budget (default {KL_FLOOR})")
    a = ap.parse_args()
    if a.cmd == "probe":
        probe(make_server(a))
    elif a.cmd == "topk":
        cmd_topk(a)
    elif a.cmd == "tokens":
        cmd_tokens(a)
    elif a.cmd == "compare":
        cmd_compare(a)
    else:
        cmd_compare_topk(a)


if __name__ == "__main__":
    main()

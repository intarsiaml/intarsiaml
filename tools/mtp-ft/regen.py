"""Greedy regeneration of prompts by the target model, tokenised for `intarsia mtp-dump --input`.

For every prompt (chat messages) the target endpoint generates at temperature 0 and the script writes
  tokens.jsonl   {"id": "<category>__<n>", "tokens": [prompt ids..., response ids...]}      <- exactly what mtp-dump parses
  meta.json      {id: {"prompt_len", "category", "name", "n_response", "finish_reason"}}      <- train.py / accept.py use prompt_len
so the head trains only on rows whose target token is the target model's own greedy argmax.

Tokens = tokenize(chat template(messages, add_generation_prompt=True)) + the generated ids (+ `<|im_end|>` if generation stopped on
EOS and the id is not already last). The assistant turn is never re-rendered through the template (Qwen templates rewrite
`<think>` content in history). Thinking stays on by default (the template's `<think>\n` generation prompt), as in the eval runs;
`--no-thinking` renders `enable_thinking=false` instead. Generation uses `/v1/completions` with the pre-rendered prompt, which
bypasses any reasoning parser; generated ids come from `choices[0].token_ids` when the server returns them (`return_token_ids`),
else the response text is re-tokenised (reported as `retokenised`).

Prompts are excluded when their user text equals (whitespace-normalised, case-folded) the user text of any eval prompt in
`--exclude` (default scripts/dfh/cats/*.jsonl: what scripts/dt1/cat-window.sh runs; its comment says dt1/cats but it reads ../dfh/cats).

Prompt JSONL lines: {"messages":[{"role","content"},...]} or {"input"|"prompt"|"text": "<user text>"[, "system": "..."]};
optional "category" and "name". Needs `tokenizers` and `jinja2` (both in the nvcr pytorch image), no other dependency.

  python regen.py --ckpt <model dir> --prompts p.jsonl --endpoint http://host:8000 --out-dir regen/ \
      [--max-tokens 1024 --concurrency 16 --category prose --model <id> --resume] [--dry-run]
  python regen.py --selftest --ckpt <model dir>         (stub server round trip, no GPU, no network)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import jinja2
from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))


# ---------------------------------------------------------------------------------------------------------------------
# tokenizer + chat template (what transformers' apply_chat_template does, without transformers)


class Renderer:
    def __init__(self, ckpt: str):
        self.tok = Tokenizer.from_file(os.path.join(ckpt, "tokenizer.json"))
        tpl_path = os.path.join(ckpt, "chat_template.jinja")
        if os.path.exists(tpl_path):
            tpl = open(tpl_path).read()
        else:
            tpl = json.load(open(os.path.join(ckpt, "tokenizer_config.json")))["chat_template"]

        def raise_exception(msg):
            raise jinja2.exceptions.TemplateError(msg)

        def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
            return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
        env.filters["tojson"] = tojson
        env.globals["raise_exception"] = raise_exception
        env.globals["strftime_now"] = lambda fmt: time.strftime(fmt)
        self.tpl = env.from_string(tpl)
        self.im_end = self.tok.token_to_id("<|im_end|>")
        self.eos_ids = {i for i in (self.im_end, self.tok.token_to_id("<|endoftext|>")) if i is not None}

    def render(self, messages, thinking: bool = True) -> str:
        kw = {} if thinking else {"enable_thinking": False}
        return self.tpl.render(messages=messages, add_generation_prompt=True, tools=None, bos_token="", eos_token="<|im_end|>", **kw)

    def encode(self, text: str) -> list:
        return self.tok.encode(text, add_special_tokens=False).ids


# ---------------------------------------------------------------------------------------------------------------------
# prompts


def norm(s: str) -> str:
    return " ".join(s.split()).casefold()


def user_texts(rec: dict) -> list:
    if rec.get("messages"):
        return [m["content"] if isinstance(m.get("content"), str) else json.dumps(m.get("content")) for m in rec["messages"] if m.get("role") == "user"]
    for k in ("input", "prompt", "text"):
        if isinstance(rec.get(k), str):
            return [rec[k]]
    return []


def to_messages(rec: dict) -> list:
    if rec.get("messages"):
        return rec["messages"]
    msgs = []
    if rec.get("system"):
        msgs.append({"role": "system", "content": rec["system"]})
    msgs.append({"role": "user", "content": user_texts(rec)[0]})
    return msgs


def eval_set(paths) -> set:
    out = set()
    for p in paths:
        for line in open(p):
            if line.strip():
                out.update(norm(t) for t in user_texts(json.loads(line)))
    return out


def default_exclude() -> list:
    pats = [os.path.join(REPO, "scripts", "dfh", "cats", "*.jsonl"), os.path.join(REPO, "scripts", "dt1", "cats", "*.jsonl")]
    return sorted(f for p in pats for f in glob.glob(p))


# ---------------------------------------------------------------------------------------------------------------------
# endpoint


def http_json(url: str, body: dict | None = None, timeout: float = 900.0) -> dict:
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def generate(endpoint: str, model: str, prompt: str, max_tokens: int, retries: int = 2) -> dict:
    body = dict(model=model, prompt=prompt, max_tokens=max_tokens, temperature=0, stream=False, skip_special_tokens=False,
                return_token_ids=True, add_special_tokens=False)
    err = None
    for i in range(retries + 1):
        try:
            return http_json(endpoint.rstrip("/") + "/v1/completions", body)["choices"][0]
        except Exception as e:  # noqa: BLE001 - network / server errors: retry, then give up on this prompt
            err = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"generation failed: {err}")


def response_ids(r: Renderer, choice: dict) -> tuple:
    """(generated ids, how) from a completions choice."""
    ids = choice.get("token_ids")
    how = "server ids"
    if not ids:
        ids = r.encode(choice.get("text", ""))
        how = "retokenised"
    ids = list(ids)
    if choice.get("finish_reason") == "stop" and not (ids and ids[-1] in r.eos_ids):
        ids.append(r.im_end)
    return ids, how


# ---------------------------------------------------------------------------------------------------------------------


def run(a) -> int:
    r = Renderer(a.ckpt)
    exclude = a.exclude if a.exclude is not None else default_exclude()
    banned = eval_set(exclude)
    print(f"exclusion set: {len(banned)} eval user texts from {len(exclude)} files", flush=True)

    recs, skipped = [], 0
    for pf in a.prompts:
        stem = os.path.splitext(os.path.basename(pf))[0]
        for line in open(pf):
            if not line.strip():
                continue
            rec = json.loads(line)
            if any(norm(t) in banned for t in user_texts(rec)):
                skipped += 1
                continue
            rec["_cat"] = (a.category or rec.get("category") or stem).replace("__", "_")
            recs.append(rec)
    if a.limit:
        recs = recs[: a.limit]
    print(f"{len(recs)} prompts kept, {skipped} excluded as eval prompts", flush=True)
    if not recs:
        return 1

    jobs = []
    counters = {}
    for rec in recs:
        c = rec["_cat"]
        counters[c] = counters.get(c, -1) + 1
        pid = f"{c}__{counters[c]:06d}"
        text = r.render(to_messages(rec), thinking=not a.no_thinking)
        jobs.append((pid, rec, text, r.encode(text)))
    plens = [len(j[3]) for j in jobs]
    print(f"prompt tokens: min {min(plens)} mean {sum(plens)/len(plens):.0f} max {max(plens)}; first prompt ends: {jobs[0][2][-80:]!r}", flush=True)
    if a.dry_run:
        print("dry run: nothing generated or written")
        return 0

    os.makedirs(a.out_dir, exist_ok=True)
    tok_path, meta_path = os.path.join(a.out_dir, "tokens.jsonl"), os.path.join(a.out_dir, "meta.json")
    meta = json.load(open(meta_path)) if (a.resume and os.path.exists(meta_path)) else {}
    done = set(meta)
    out = open(tok_path, "a" if a.resume else "w")
    model = a.model
    try:
        ms = http_json(a.endpoint.rstrip("/") + "/v1/models", timeout=30)
        ids = [m.get("id") for m in ms.get("data", [])]
        print(f"endpoint {a.endpoint} serves: {ids}", flush=True)
        model = model or (ids[0] if ids else "default")
    except Exception as e:  # noqa: BLE001
        print(f"warning: /v1/models failed ({e}); using --model {model!r}", flush=True)
        model = model or "default"

    lock = threading.Lock()
    stats = dict(ok=0, failed=0, truncated=0, retok=0, resp_tokens=0)
    t0 = time.time()

    def work(job):
        pid, rec, text, pids = job
        ch = generate(a.endpoint, model, text, a.max_tokens)
        rids, how = response_ids(r, ch)
        return pid, rec, pids, rids, how, ch.get("finish_reason")

    todo = [j for j in jobs if j[0] not in done]
    with ThreadPoolExecutor(a.concurrency) as ex:
        futs = {ex.submit(work, j): j for j in todo}
        for f in as_completed(futs):
            try:
                pid, rec, pids, rids, how, fin = f.result()
            except Exception as e:  # noqa: BLE001
                stats["failed"] += 1
                print(f"{futs[f][0]}: {e}", file=sys.stderr, flush=True)
                continue
            if fin == "length":
                stats["truncated"] += 1
                if a.drop_truncated:
                    continue
            with lock:
                out.write(json.dumps({"id": pid, "tokens": pids + rids}) + "\n")
                out.flush()
                meta[pid] = dict(prompt_len=len(pids), category=rec["_cat"], name=rec.get("name"), n_response=len(rids), finish_reason=fin)
                stats["ok"] += 1
                stats["retok"] += how == "retokenised"
                stats["resp_tokens"] += len(rids)
                if stats["ok"] % 25 == 0:
                    json.dump(meta, open(meta_path, "w"))
                    print(f"{stats['ok']}/{len(todo)} done, {stats['resp_tokens']/(time.time()-t0):.0f} resp tok/s", flush=True)
    out.close()
    json.dump(meta, open(meta_path, "w"))
    print(f"done: {stats}; wrote {tok_path} and {meta_path}", flush=True)
    return 0 if stats["ok"] else 1


def selftest(ckpt: str) -> int:
    """Stub OpenAI-style server: answers /v1/completions with a canned reply (server ids for one prompt, text only for another)."""
    import http.server
    import tempfile

    r = Renderer(ckpt)

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send(200, {"data": [{"id": "stub-model"}]})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert body["temperature"] == 0 and body["prompt"].endswith("<think>\n"), body["prompt"][-40:]
            reply = "Let me think.\n</think>\n\nHello there, this is a stub answer."
            ids = r.encode(reply)
            if "no-ids" in body["prompt"]:
                self.send(200, {"choices": [{"text": reply, "finish_reason": "stop"}]})
            else:
                self.send(200, {"choices": [{"text": reply, "token_ids": ids, "finish_reason": "length" if "long" in body["prompt"] else "stop"}]})

        def send(self, code, obj):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    with tempfile.TemporaryDirectory() as td:
        pf = os.path.join(td, "p.jsonl")
        evalf = os.path.join(td, "evalset.jsonl")
        open(evalf, "w").write(json.dumps({"name": "e", "input": "What is  the capital of France?"}) + "\n")
        open(pf, "w").write("\n".join(json.dumps(x) for x in [
            {"input": "what is the capital of  france?"},  # same as the eval prompt after normalisation -> excluded
            {"input": "Say hi."}, {"input": "no-ids please"}, {"input": "long one"},
            {"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Two plus two?"}], "category": "chat"},
        ]) + "\n")
        a = argparse.Namespace(ckpt=ckpt, prompts=[pf], endpoint=f"http://127.0.0.1:{srv.server_address[1]}", out_dir=os.path.join(td, "o"),
                               exclude=[evalf], category=None, limit=0, no_thinking=False, dry_run=False, resume=False, model=None,
                               max_tokens=64, concurrency=3, drop_truncated=False)
        assert run(a) == 0
        toks = [json.loads(l) for l in open(os.path.join(td, "o", "tokens.jsonl"))]
        meta = json.load(open(os.path.join(td, "o", "meta.json")))
        assert len(toks) == 4 and len(meta) == 4, (len(toks), len(meta))
        for t in toks:
            m = meta[t["id"]]
            assert set(t) == {"id", "tokens"}
            assert t["tokens"][m["prompt_len"] - 1] == r.encode("<think>\n")[-1]
            assert len(t["tokens"]) == m["prompt_len"] + m["n_response"]
            resp = t["tokens"][m["prompt_len"]:]
            assert r.tok.decode(resp, skip_special_tokens=False).startswith("Let me think."), r.tok.decode(resp)
            assert (resp[-1] == r.im_end) == (m["finish_reason"] == "stop"), (resp[-3:], m)
        assert sum(m["finish_reason"] == "length" for m in meta.values()) == 1
        print("selftest ok:", {k: (v["prompt_len"], v["n_response"], v["finish_reason"]) for k, v in meta.items()})
    srv.shutdown()
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint dir (tokenizer.json, chat_template.jinja)")
    ap.add_argument("--prompts", nargs="+")
    ap.add_argument("--endpoint")
    ap.add_argument("--model", default=None)
    ap.add_argument("--out-dir", default="regen")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--category", default=None, help="category for every prompt (default: the line's, else the file name)")
    ap.add_argument("--exclude", nargs="*", default=None, help="eval prompt JSONL files (default scripts/dfh/cats/*.jsonl); give none to disable")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--drop-truncated", action="store_true", help="drop responses that hit --max-tokens")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="render, tokenise, exclude and report; no requests")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest(a.ckpt))
    if not a.prompts or (not a.endpoint and not a.dry_run):
        ap.error("--prompts and --endpoint are required (or --dry-run / --selftest)")
    sys.exit(run(a))


if __name__ == "__main__":
    main()

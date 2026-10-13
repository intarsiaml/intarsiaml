#!/usr/bin/env python3
"""Build a top-k golden from an OpenAI-compatible vLLM server (prompt_logprobs is a vLLM extension).

For every prompt of the prompt set the script tokenises the text with the server's own `/tokenize`,
then sends ONE `/v1/completions` request (`max_tokens=1`, `prompt_logprobs=K`) with the token ids plus one
extra dummy token at the end. vLLM returns, for each prompt position j >= 1, the log-probs of the top-K
tokens of the distribution that predicts token j given tokens < j. Row t of the golden (the distribution
over token t+1 after ids[..=t]) is therefore `prompt_logprobs[t+1]`, and the dummy last token exists only to
get the row for the final real position (causal attention: it cannot change earlier rows).

Output: format `intarsia-golden-topk-v1` (docs/accuracy.md): per prompt i
    ids_i          I32 [n]        token ids
    topk_ids_i     I32 [n, K]     the K most likely ids per row, best first (-1 padded)
    topk_logprobs_i F32 [n, K]    their FULL-vocabulary log-probs as the server reports them (-1e30 padded)

    python3 tools/golden_vllm.py --base-url http://spark-4e77:8000/v1 --model guidance \\
        --out golden/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors [--prompts a.jsonl ...] [--topk 20]

Prompts with `"continue": N` get N tokens of the server's own greedy continuation appended (see goldenlib.py), so
the teacher-forced decode window is on the reference's distribution. Prompts that do not fit the server's
`max_model_len` (plus the continuation and the dummy token) are skipped with a message, never sent; the golden
then has fewer prompts. The golden's metadata records what the API shows about the reference: vLLM version,
`vllm:cache_config_info` (KV cache dtype, block size, prefix caching), `max_model_len`, the model root and
revision. Launch flags are NOT observable through the API and are recorded as such; pass `--note` for anything
you know (e.g. the flags the server was started with).

Only the standard library, numpy and safetensors are needed. Requests to a fleet vLLM server must run inside
`fleetq run --uses <role> ...`.
"""
import argparse
import json
import pathlib
import re
import time
import urllib.request

import numpy as np
from safetensors.numpy import save_file

import goldenlib as gl

HERE = pathlib.Path(__file__).resolve().parent
PAD_LP = -1e30


def post(base, path, body, timeout=1800):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def get_text(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode()


def cache_config_info(root):
    """Labels of the `vllm:cache_config_info` metric ({} when /metrics has none)."""
    try:
        txt = get_text(root + "/metrics")
    except Exception as e:  # metrics are optional
        return {"error": str(e)}
    for line in txt.splitlines():
        if line.startswith("vllm:cache_config_info{"):
            return dict(re.findall(r'(\w+)="([^"]*)"', line[line.index("{"):line.index("}")]))
    return {}


def continuation_ids(base, model, ids, n):
    """The server's own greedy continuation of `ids`: n token ids (fewer if the server stops early)."""
    r = post(base, "/completions", {"model": model, "prompt": ids, "max_tokens": n, "min_tokens": n, "temperature": 0,
                                    "ignore_eos": True, "logprobs": 0, "return_tokens_as_token_ids": True})
    ch = r["choices"][0]
    if ch.get("token_ids"):
        return [int(t) for t in ch["token_ids"]]
    toks = (ch.get("logprobs") or {}).get("tokens") or []
    out = []
    for t in toks:
        m = re.fullmatch(r"token_id:(\d+)", t)
        if not m:
            raise SystemExit(f"cannot read the continuation token ids from the response (got {t!r}); "
                             "server lacks return_tokens_as_token_ids")
        out.append(int(m.group(1)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://spark-4e77:8000/v1", help="OpenAI-compatible base URL (ends in /v1)")
    ap.add_argument("--model", default="guidance", help="served model name")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", nargs="+",
                    default=[str(HERE / "prompts" / "golden.jsonl"), str(HERE / "prompts" / "golden_long.jsonl")])
    ap.add_argument("--topk", type=int, default=20, help="vLLM prompt_logprobs (top-K per position, max 20 by default)")
    ap.add_argument("--dump-first", action="store_true", help="print the raw response of the first prompt and exit")
    ap.add_argument("--note", default="", help="free text stored in the metadata (e.g. the server's launch flags)")
    ap.add_argument("--cache-salt", default=None, metavar="SALT",
                    help="send vLLM's per-request `cache_salt` (SALT-<prompt index>-<n tokens>) so the golden request cannot read prefix-cache "
                         "blocks of an earlier request (Phase 1b review finding 10b: the prompt+continuation is re-sent to a server with "
                         "prefix caching on, and whether prompt_logprobs requests bypass the fp8 KV blocks is unverified for 0.26). "
                         "NOT TESTED against a live server; the value is recorded in the metadata, and a server that does not know the field "
                         "answers 4xx (then drop the flag and record that the cache was not controlled)")
    args = ap.parse_args()
    base = args.base_url.rstrip("/")
    root = base[:-3] if base.endswith("/v1") else base  # /tokenize lives at the server root

    def tokenize(text):
        r = post(root, "/tokenize", {"model": args.model, "prompt": text, "add_special_tokens": True})
        return r["tokens"]

    def apply_template(messages):
        raise SystemExit("this tool uses the fixed ChatML text; --template hf is not supported here")

    prompts = gl.load_prompts(args.prompts)
    models = json.load(urllib.request.urlopen(base + "/models"))["data"]
    max_len = models[0].get("max_model_len")
    print(f"server max_model_len={max_len}", flush=True)
    tensors, meta_prompts, skipped = {}, [], []
    t0 = time.time()
    i = 0
    for p in prompts:
        ids, text = gl.prompt_ids(p, tokenize, "chatml", apply_template)
        n_prompt = len(ids)
        want = int(p.get("continue", 0))
        if max_len is not None and n_prompt + want + 2 > max_len:
            msg = f"{p['name']}: {n_prompt}+{want} tokens do not fit the server's max_model_len {max_len}; skipped"
            print("SKIPPED " + msg, flush=True)
            skipped.append(msg)
            continue
        split = gl.reuse_split(p, ids, tokenize)
        if want:
            # long requests are heavy on a shared server: make sure it is still healthy before each one
            try:
                get_text(root + "/health")
            except Exception as e:
                raise SystemExit(f"server /health failed before {p['name']}: {e}")
            cont = continuation_ids(base, args.model, ids, want)
            if len(cont) != want:
                raise SystemExit(f"{p['name']}: asked for {want} continuation tokens, got {len(cont)}")
            ids = ids + cont
        n = len(ids)
        body = {"model": args.model, "prompt": ids + [ids[-1]], "max_tokens": 1, "temperature": 0,
                "prompt_logprobs": args.topk}
        if args.cache_salt:
            body["cache_salt"] = f"{args.cache_salt}-{len(meta_prompts)}-{n}"
        resp = post(base, "/completions", body)
        if args.dump_first:
            print(json.dumps(resp)[:3000])
            return
        pl = resp["choices"][0]["prompt_logprobs"]
        if pl is None or len(pl) != n + 1 or pl[0] is not None:
            raise SystemExit(f"{p['name']}: expected {n + 1} prompt_logprobs entries with a null first one, "
                             f"got {None if pl is None else len(pl)}")
        K = args.topk
        tid = np.full((n, K), -1, dtype=np.int32)
        tlp = np.full((n, K), PAD_LP, dtype=np.float32)
        for t in range(n):
            entry = pl[t + 1]
            # entries: the top-K plus (if outside it) the actual prompt token, which has rank > K; drop that one
            items = sorted(((int(k), v["logprob"], v.get("rank", 0)) for k, v in entry.items()), key=lambda x: -x[1])
            items = [x for x in items if x[2] <= K][:K]
            if not items:
                raise SystemExit(f"{p['name']} row {t}: empty logprob entry")
            for j, (k, lp, _) in enumerate(items):
                tid[t, j], tlp[t, j] = k, lp
        tensors[f"ids_{i}"] = np.array(ids, dtype=np.int32)
        tensors[f"topk_ids_{i}"] = tid
        tensors[f"topk_logprobs_{i}"] = tlp
        meta_prompts.append(gl.meta_prompt(p, text, n, n_prompt, split))
        print(f"[{i}] {p['name']:<16} {n:5d} tokens ({n_prompt}+{n - n_prompt})  top1_last={int(tid[-1, 0])} "
              f"lp={float(tlp[-1, 0]):.3f}  {time.time() - t0:.0f}s", flush=True)
        i += 1
    if not tensors:
        raise SystemExit("no prompt was built (all skipped?)")

    try:
        version = json.loads(get_text(root + "/version")).get("version", "unknown")
    except Exception:
        version = "unknown"
    cci = cache_config_info(root)
    mroot = models[0].get("root") or args.model
    meta = {
        "format": gl.FORMAT_TOPK,
        "model": mroot,
        "model_revision": gl.snapshot_hash(mroot) or "not in the model root path",
        "source": "vllm",
        "engine": "vllm",
        "engine_version": version,
        "server": args.base_url,
        "served_model": args.model,
        "dtype": "server",
        "device": "server",
        "kv_dtype": cci.get("cache_dtype", "unknown"),
        "cache_config_info": json.dumps(cci),
        "max_model_len": str(max_len),
        "launch_flags": "not observable through the API" + (f"; note: {args.note}" if args.note else ""),
        "template": "chatml",
        "add_special_tokens": "true",
        "sampling": json.dumps({"prompt_logprobs": args.topk, "temperature": 0,
                                "continuation": "greedy, ignore_eos, from the same server",
                                "cache_salt": args.cache_salt or "not used (prefix cache not controlled: unverified whether "
                                "prompt_logprobs requests read cached fp8 KV blocks)"}),
        "prompt_files": json.dumps([pathlib.Path(x).name for x in args.prompts]),
        "skipped": json.dumps(skipped),
        "topk": str(args.topk),
        "prompts": json.dumps(meta_prompts),
        "built_unix": str(int(time.time())),
        "note": "row t = distribution over token t+1 given ids[..=t]; logprobs are full-vocab as reported by vLLM",
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out), metadata=meta)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB, {len(meta_prompts)} prompts, {len(skipped)} skipped) "
          f"in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

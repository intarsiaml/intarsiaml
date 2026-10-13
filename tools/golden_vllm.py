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

Only the standard library, numpy and safetensors are needed. Requests to a fleet vLLM server must run inside
`fleetq run --uses <role> ...`.
"""
import argparse
import json
import pathlib
import time
import urllib.request

import numpy as np
from safetensors.numpy import save_file

import goldenlib as gl

HERE = pathlib.Path(__file__).resolve().parent
PAD_LP = -1e30


def post(base, path, body, timeout=600):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://spark-4e77:8000/v1", help="OpenAI-compatible base URL (ends in /v1)")
    ap.add_argument("--model", default="guidance", help="served model name")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", nargs="+",
                    default=[str(HERE / "prompts" / "golden.jsonl"), str(HERE / "prompts" / "golden_long.jsonl")])
    ap.add_argument("--topk", type=int, default=20, help="vLLM prompt_logprobs (top-K per position, max 20 by default)")
    ap.add_argument("--dump-first", action="store_true", help="print the raw response of the first prompt and exit")
    args = ap.parse_args()
    base = args.base_url.rstrip("/")
    root = base[:-3] if base.endswith("/v1") else base  # /tokenize lives at the server root

    def tokenize(text):
        r = post(root, "/tokenize", {"model": args.model, "prompt": text, "add_special_tokens": True})
        return r["tokens"]

    def apply_template(messages):
        raise SystemExit("this tool uses the fixed ChatML text; --template hf is not supported here")

    prompts = gl.load_prompts(args.prompts)
    tensors, meta_prompts = {}, []
    t0 = time.time()
    for i, p in enumerate(prompts):
        ids, text = gl.prompt_ids(p, tokenize, "chatml", apply_template)
        n = len(ids)
        body = {"model": args.model, "prompt": ids + [ids[-1]], "max_tokens": 1, "temperature": 0,
                "prompt_logprobs": args.topk}
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
        meta_prompts.append({"name": p["name"], "kind": p["kind"], "n_tokens": n, "text": text})
        print(f"[{i}] {p['name']:<16} {n:4d} tokens  top1_last={int(tid[-1, 0])} lp={float(tlp[-1, 0]):.3f}", flush=True)

    models = json.load(urllib.request.urlopen(base + "/models"))["data"]
    meta = {
        "format": gl.FORMAT_TOPK,
        "model": models[0].get("root") or args.model,
        "source": "vllm",
        "server": args.base_url,
        "served_model": args.model,
        "dtype": "server",
        "device": "server",
        "template": "chatml",
        "topk": str(args.topk),
        "prompts": json.dumps(meta_prompts),
        "note": "row t = distribution over token t+1 given ids[..=t]; logprobs are full-vocab as reported by vLLM",
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out), metadata=meta)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB, {len(prompts)} prompts) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

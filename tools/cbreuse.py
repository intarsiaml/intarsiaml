#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["tokenizers"]
# ///
"""Concurrent identity WITH prefix reuse of decode-written KV (gate A of docs/results/2026-10-16-metal-cb-gates-reuse-spec.md).

cbidentity.py runs with INTARSIA_CB_NO_REUSE=1: no request starts from reused KV. In production an idle slot's KV is
reused (`pick_slot`: longest common prefix of slot tokens and prompt, capped at plen - 1), and the slot tokens hold the
previous request's prompt AND its generated tokens (all but the last, which is never fed). This tool builds *chains*:

    P  (cold prompt)  ->  F  (prompt = P.prompt + P.output [+ new tokens])  ->  F2  (prompt = F.prompt + F.output + ...)

Contract (docs/research/continuous-batching-design.md 4.5): a batched request equals its solo run at the same prefill
chunk and the same cached length, with KV written the same way. The reference is therefore a *reproduced predecessor*:

  gen   a solo server (--max-batch 1, prefix cache on) runs each chain in order (flush, P, F, F2); F's prompt is built
        from the solo P's output ids. Writes chains.jsonl (the exact prompt ids) and ref.jsonl (the solo records).
  run   a batched server (--max-batch 8, reuse on) is driven with B concurrent clients, one chain per client at a time,
        each chain in order with the prompts from chains.jsonl.
  compare  request by request: token ids, text, deltas, finish_reason, usage against the solo record. A request is
        *comparable* only if it and every earlier request of its chain ran with the reference's prefill chunk and cached
        length (otherwise the KV it started from was not written the way the solo run wrote it). Not comparable
        requests are counted and listed, never silently dropped. Exit 1 on any mismatch or error.

Usage (stdlib + tokenizers for `gen`):
  uv run tools/cbreuse.py gen     --url http://127.0.0.1:8101 --chains 130 --out-dir DIR     (solo server)
  uv run tools/cbreuse.py run     --url http://127.0.0.1:8101 --chains DIR/chains.jsonl --b 8 --out DIR/conc.jsonl
  uv run tools/cbreuse.py compare --ref DIR/ref.jsonl --chains DIR/chains.jsonl --test DIR/conc.jsonl [--report R.json]
"""
import argparse
import collections
import json
import pathlib
import random
import sys
import threading
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cbidentity as cb  # noqa: E402

DEFAULT_TOKENIZER = pathlib.Path.home() / "models" / "Qwen3-8B" / "tokenizer.json"
KINDS = [("ext", 0.40), ("ext2", 0.20), ("branch", 0.25), ("exact", 0.15)]


def body_of(st, model):
    b = {"model": model, "prompt": st["prompt"], "max_tokens": st["max_tokens"], "temperature": st["temperature"],
         "stream": st["stream"], "return_token_ids": True}
    for k in ("top_p", "top_k", "seed"):
        if st.get(k) is not None:
            b[k] = st[k]
    if st["stream"]:
        b["stream_options"] = {"include_usage": True}
    return b


def chunk_of(rec):
    return cb.rec_chunk(rec)


def cached_of(rec):
    return ((rec.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens")


def read_jsonl(path):
    return [json.loads(ln) for ln in pathlib.Path(path).read_text().splitlines() if ln.strip()]


# ----------------------------------------------------------------------------------------------------------------------
# gen: build chains on a solo server
# ----------------------------------------------------------------------------------------------------------------------

def cmd_gen(args):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(args.tokenizer))
    rng = random.Random(args.seed)
    pdir = HERE / "prompts"
    corp = {f: (pdir / f).read_text() for f in ("corpus_docs.txt", "corpus_code.txt")}
    vocab = sorted(w for w in tok.get_vocab() if w.isascii() and w.isalpha() and 4 <= len(w) <= 9)
    used_first = {cb.FLUSH_ID}

    def nonce_ids():
        while True:
            w = rng.choice(vocab)
            ids = tok.encode(w + " notes: ", add_special_tokens=False).ids
            if ids[0] not in used_first:
                used_first.add(ids[0])
                return ids

    def slice_ids(n):
        f = rng.choice(["corpus_docs.txt", "corpus_docs.txt", "corpus_code.txt"])
        text = corp[f]
        while True:
            start = rng.randrange(0, max(1, len(text) - 12 * n))
            nl = text.find("\n", start)
            start = nl + 1 if 0 <= nl < start + 200 else start
            ids = tok.encode(text[start:start + 12 * n], add_special_tokens=False).ids
            if len(ids) >= n:
                return ids[:n]

    def copy_task_ids():
        """Input-grounded cold prompt (as tools/cbspec_cases.py `copy`): '--spec ngram' drafts from the text in the prompt."""
        f = rng.choice(["corpus_docs.txt", "corpus_code.txt"])
        text = corp[f]
        clen = rng.randint(500, 1000)
        start = rng.randrange(0, len(text) - clen - 1)
        nl = text.find("\n", start)
        start = nl + 1 if 0 <= nl < start + 200 else start
        sl = text[start:start + clen]
        return tok.encode("Repeat the following text word for word.\n\nTEXT:\n" + sl + "\n\nREPEATED TEXT:\n", add_special_tokens=False).ids

    def sampling(st):
        if rng.random() < 0.30:
            st.update(temperature=rng.choice([0.6, 0.8, 1.0]), top_p=rng.choice([1.0, 0.95, 0.9]),
                      top_k=rng.choice([0, 40, 20]), seed=rng.randrange(1, 2**31))
        else:
            st["temperature"] = 0

    def prompt_len():
        r = rng.random()
        return rng.randint(64, 400) if r < 0.6 else rng.randint(400, 1200) if r < 0.9 else rng.randint(1200, 2400)

    info = cb.server_info(args.url)
    model = cb.model_name(info)
    if not cb.probe_token_ids(args.url, model):
        raise SystemExit("server does not return token_ids")
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chains_f = open(out_dir / "chains.jsonl", "w")
    ref_recs = []
    t0 = time.monotonic()
    for n in range(args.chains):
        kind = rng.choices([k for k, _ in KINDS], [w for _, w in KINDS])[0]
        depth = 3 if kind == "ext2" else 2
        # decode-written token ranges (indices into the step's prompt) that the reference history put into the KV
        copy_chain = args.copy_frac > 0 and rng.random() < args.copy_frac
        prompt = nonce_ids() + (copy_task_ids() if copy_chain else slice_ids(prompt_len()))
        dec = []
        flush(args.url, model, args.timeout)
        for k in range(depth):
            st = {"id": f"c{n:03d}s{k}", "chain": n, "step": k, "kind": kind, "prompt": prompt, "dec_ranges": dec,
                  "max_tokens": rng.randint(24, 120) if k == 0 else rng.randint(24, 100), "stream": rng.random() < 0.5}
            sampling(st)
            if copy_chain and k == 0:  # greedy (only greedy requests speculate) and long enough to draft
                for key in ("top_p", "top_k", "seed"):
                    st.pop(key, None)
                st.update(temperature=0, max_tokens=rng.choice([96, 160, 224]))
            rec = cb.one_request(args.url, "/v1/completions", body_of(st, model), st["stream"], args.timeout)
            rec["id"], rec["chain"], rec["step"] = st["id"], n, k
            chains_f.write(json.dumps(st) + "\n")
            chains_f.flush()
            ref_recs.append(rec)
            u = rec.get("usage") or {}
            print(f"  {st['id']} {kind:6s} S={'S' if st['stream'] else 'N'} plen {len(prompt):5d} cached {cached_of(rec)} "
                  f"gen {u.get('completion_tokens')} chunk {chunk_of(rec)} {rec['status']} {rec['finish_reason']} "
                  f"[{time.monotonic() - t0:.0f}s]", flush=True)
            if rec["status"] != "ok" or not rec["ids"]:
                break
            if k + 1 >= depth:
                break
            out = rec["ids"]
            if len(prompt) + len(out) + 200 + st["max_tokens"] > args.max_seq - 64:
                break
            plen0 = len(prompt)
            extra_n = rng.randint(8, 120)
            if kind in ("ext", "ext2"):
                prompt, dec = prompt + out + slice_ids(extra_n), dec + [[plen0, plen0 + len(out)]]
            elif kind == "exact":
                prompt, dec = prompt + out, dec + [[plen0, plen0 + len(out)]]
            else:  # branch: keep a random-length head of the output, then diverge
                keep = rng.randint(1, max(1, len(out) - 1))
                tail = slice_ids(extra_n)
                if keep < len(out) and tail[0] == out[keep]:
                    tail = tail[1:]
                prompt, dec = prompt + out[:keep] + tail, dec + [[plen0, plen0 + keep]]
    chains_f.close()
    hdr = {"mode": "solo-chains", "server": info, "seed": args.seed, "chains": args.chains, "n_requests": len(ref_recs),
           "wall_s": round(time.monotonic() - t0, 1), "started": time.strftime("%Y-%m-%d %H:%M:%S"), "argv": sys.argv}
    cb.write_run(out_dir / "ref.jsonl", hdr, ref_recs)
    bad = [r["id"] for r in ref_recs if r["status"] != "ok"]
    print(f"wrote {out_dir}/chains.jsonl, ref.jsonl: {len(ref_recs)} requests, {len(bad)} errors, wall {hdr['wall_s']}s")
    return 1 if bad else 0


def flush(url, model, timeout):
    """Drop the held KV so the chain's first request starts cold (the solo server keeps one sequence)."""
    cb.one_request(url, "/v1/completions", {"model": model, "prompt": [cb.FLUSH_ID], "max_tokens": 1, "temperature": 0},
                   False, timeout)


# ----------------------------------------------------------------------------------------------------------------------
# run: concurrent chains on the batched server
# ----------------------------------------------------------------------------------------------------------------------

def cmd_run(args):
    steps = read_jsonl(args.chains)
    chains = collections.OrderedDict()
    for s in steps:
        chains.setdefault(s["chain"], []).append(s)
    if args.limit:
        chains = collections.OrderedDict(list(chains.items())[: args.limit])
    info = cb.server_info(args.url)
    model = cb.model_name(info)
    if not cb.probe_token_ids(args.url, model):
        raise SystemExit("server does not return token_ids")
    lock = threading.Lock()
    queue = list(chains.items())
    records = []
    inflight = [0, 0]
    t0 = time.monotonic()

    def worker(widx):
        time.sleep(widx * args.stagger)
        while True:
            with lock:
                if not queue:
                    return
                cid, sts = queue.pop(0)
            rng = random.Random(f"gap-{cid}")
            for st in sts:
                if st["step"] > 0 and rng.random() < args.gap_frac:
                    time.sleep(rng.uniform(0.05, args.gap_max))  # another client may take the predecessor's slot
                with lock:
                    inflight[0] += 1
                    inflight[1] = max(inflight[1], inflight[0])
                rec = cb.one_request(args.url, "/v1/completions", body_of(st, model), st["stream"], args.timeout)
                with lock:
                    inflight[0] -= 1
                    rec["id"], rec["chain"], rec["step"] = st["id"], st["chain"], st["step"]
                    rec["t_start_ms"] = round((time.monotonic() - t0) * 1000, 1)
                    records.append(rec)
                    u = rec.get("usage") or {}
                    print(f"  [{len(records):3d}] {st['id']} {st['kind']:6s} cached {cached_of(rec)} gen {u.get('completion_tokens')} "
                          f"chunk {chunk_of(rec)} {rec['status']} {rec['finish_reason']}", flush=True)
                if rec["status"] != "ok":
                    break

    ths = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(args.b)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    hdr = {"mode": "concurrent-chains", "b": args.b, "gap_frac": args.gap_frac, "gap_max": args.gap_max, "server": info,
           "n_requests": len(records), "peak_client_inflight": inflight[1], "wall_s": round(time.monotonic() - t0, 1),
           "started": time.strftime("%Y-%m-%d %H:%M:%S"), "argv": sys.argv}
    cb.write_run(args.out, hdr, sorted(records, key=lambda r: (r["chain"], r["step"])))
    bad = [r["id"] for r in records if r["status"] != "ok"]
    print(f"wrote {args.out}: {len(records)} requests, {len(bad)} errors, wall {hdr['wall_s']}s, peak in-flight {inflight[1]}")
    return 0


# ----------------------------------------------------------------------------------------------------------------------
# compare
# ----------------------------------------------------------------------------------------------------------------------

def decode_cached(st, cached):
    """Tokens of the reused prefix [0, cached) that were written by decode in the history."""
    if cached is None:
        return 0
    return sum(max(0, min(hi, cached) - lo) for lo, hi in st["dec_ranges"])


def cmd_compare(args):
    steps = {s["id"]: s for s in read_jsonl(args.chains)}
    rh, ref = cb.read_run(args.ref)
    ref_chunk_default = cb.ref_chunk(rh)
    rows, mism, notcomp = [], [], []
    tot = collections.Counter()
    for tf in args.test:
        th, test = cb.read_run(tf)
        by_chain = collections.defaultdict(list)
        for rid in test:
            by_chain[test[rid]["chain"]].append(rid)
        for cid, rids in sorted(by_chain.items()):
            hist_ok, hist_why = True, None
            for rid in sorted(rids, key=lambda r: test[r]["step"]):
                st, t, r = steps[rid], test[rid], ref.get(rid)
                tot["requests"] += 1
                if r is None or r["status"] != "ok":
                    tot["ref_error"] += 1
                    mism.append({"id": rid, "why": ["no usable reference record"]})
                    hist_ok, hist_why = False, "ref_error"
                    continue
                cached_t, cached_r = cached_of(t), cached_of(r)
                chunk_t, chunk_r = chunk_of(t), chunk_of(r) or ref_chunk_default
                dec_t = decode_cached(st, cached_t)
                row = {"id": rid, "kind": st["kind"], "step": st["step"], "plen": len(st["prompt"]), "cached_ref": cached_r,
                       "cached_test": cached_t, "chunk_ref": chunk_r, "chunk_test": chunk_t, "decode_cached_test": dec_t,
                       "stream": st["stream"], "sampled": st["temperature"] != 0, "file": pathlib.Path(tf).name}
                if t["status"] != "ok":
                    tot["test_error"] += 1
                    mism.append({"id": rid, "why": [f"test {t['status']}: {t.get('error')}"]})
                    hist_ok, hist_why = False, "test_error"
                    row["verdict"] = "error"
                    rows.append(row)
                    continue
                here_ok = cached_t == cached_r and chunk_t == chunk_r
                why = None if here_ok else ("cached differs" if cached_t != cached_r else "chunk differs")
                if not (hist_ok and here_ok):
                    reason = why or f"earlier step of the chain not reproducible ({hist_why})"
                    tot["not_comparable"] += 1
                    tot["not_comparable_" + reason.split(" (")[0].replace(" ", "_")] += 1
                    row["verdict"] = "not_comparable"
                    row["why"] = reason
                    row["info_ids_equal"] = (r.get("ids") == t.get("ids"))  # informational, not a gate verdict
                    notcomp.append(row)
                    rows.append(row)
                    hist_ok = False
                    hist_why = hist_why or why
                    continue
                verdict, reasons, _ = cb.compare_case({"id": rid}, r, t)
                row["verdict"] = verdict
                tot["comparable"] += 1
                tot["comparable_decode_reuse"] += 1 if dec_t > 0 else 0
                tot["comparable_reuse_any"] += 1 if (cached_t or 0) > 0 else 0
                tot["decode_cached_tokens_comparable"] += dec_t
                if verdict != "pass":
                    row["why"] = reasons
                    mism.append({"id": rid, "why": reasons})
                    tot["mismatch"] += 1
                else:
                    tot["pass"] += 1
                rows.append(row)
    rep = {"ref": str(args.ref), "test": [str(t) for t in args.test], "totals": dict(tot), "mismatches": mism,
           "not_comparable": notcomp, "rows": rows}
    n = tot["comparable"]
    print(f"requests {tot['requests']}  comparable {n}  pass {tot['pass']}  MISMATCH {tot['mismatch']}  "
          f"errors ref/test {tot['ref_error']}/{tot['test_error']}  not comparable {tot['not_comparable']}")
    print(f"comparable requests that started from reused KV: {tot['comparable_reuse_any']} "
          f"({100 * tot['comparable_reuse_any'] / max(n, 1):.0f}%); from reused DECODE-WRITTEN KV: {tot['comparable_decode_reuse']} "
          f"({100 * tot['comparable_decode_reuse'] / max(n, 1):.0f}%), {tot['decode_cached_tokens_comparable']} decode-written tokens reused")
    for k in sorted(tot):
        if k.startswith("not_comparable_"):
            print(f"  {k}: {tot[k]}")
    kinds = collections.Counter((r["kind"], r["verdict"]) for r in rows)
    for (k, v), c in sorted(kinds.items()):
        print(f"  kind {k:7s} {v:15s} {c}")
    for m in mism[:20]:
        print("MISMATCH", m["id"], "; ".join(m["why"]))
    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(rep, indent=1))
    return 1 if (tot["mismatch"] or tot["ref_error"] or tot["test_error"]) else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--url", required=True)
    g.add_argument("--chains", type=int, default=130)
    g.add_argument("--seed", type=int, default=20261016)
    g.add_argument("--out-dir", required=True)
    g.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    g.add_argument("--timeout", type=int, default=900)
    g.add_argument("--max-seq", type=int, default=8192)
    g.add_argument("--copy-frac", type=float, default=0.0, help="share of chains whose cold prompt is an input-grounded copy task (for --spec ngram runs)")
    r = sub.add_parser("run")
    r.add_argument("--url", required=True)
    r.add_argument("--chains", required=True)
    r.add_argument("--b", type=int, default=8)
    r.add_argument("--out", required=True)
    r.add_argument("--limit", type=int)
    r.add_argument("--gap-frac", type=float, default=0.15, help="share of follow-ups preceded by a random pause")
    r.add_argument("--gap-max", type=float, default=0.5)
    r.add_argument("--stagger", type=float, default=0.2, help="seconds between the first requests of the B clients")
    r.add_argument("--timeout", type=int, default=900)
    c = sub.add_parser("compare")
    c.add_argument("--ref", required=True)
    c.add_argument("--chains", required=True)
    c.add_argument("--test", nargs="+", required=True)
    c.add_argument("--report")
    args = ap.parse_args()
    sys.exit({"gen": cmd_gen, "run": cmd_run, "compare": cmd_compare}[args.cmd](args))


if __name__ == "__main__":
    main()

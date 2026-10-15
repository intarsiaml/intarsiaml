#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Concurrent-identity harness for `intarsia serve` (gate of milestone M1, docs/research/continuous-batching-design.md).

Exact-only batching (design section 8.1): every request served while others run must be byte-identical to
the same request run alone. This tool runs a case set twice, once alone (reference) and once replayed
concurrently with staggered arrivals, and compares the two per request. The reference and the concurrent
run take their own server URL, so they can be different servers with different flags (for example a
reference with `--max-batch 1 --prefill-chunk 256` and a batched server).

Standard library only (`http.client` + threads). `gen-cases` needs the `tokenizers` package:

    uv run --with tokenizers tools/cbidentity.py gen-cases --out tools/prompts/cb_identity_cases.jsonl
    uv run tools/cbidentity.py run --mode solo       --url http://127.0.0.1:8090 --cases CASES --out ref.jsonl
    uv run tools/cbidentity.py run --mode concurrent --url http://127.0.0.1:8090 --cases CASES --b 4 --stream mixed --out b4.jsonl
    uv run tools/cbidentity.py compare --ref ref.jsonl --test b4.jsonl [b8.jsonl ...] --report report.json
    uv run tools/cbidentity.py gate --ref-url U1 --url U2 --cases CASES --b 2,3,4,6,8 --outdir DIR   # all of it
    uv run tools/cbidentity.py selftest        # offline: mock server, injected bug, synthetic records

`compare` exits 1 on any mismatch. What is compared: the response text, the list of streamed deltas (stream
runs against a stream reference), finish_reason, completion_tokens, prompt_tokens, and, as the primary criterion,
the generated token ids. `intarsia serve` returns ids when the request carries the extension field
`"return_token_ids": true` (`choices[0].token_ids` non-streaming; `delta.token_ids` per chunk for chat and
`choices[0].token_ids` per chunk for completions; the stop/EOS token is not included). `--token-ids auto|on|off`
(default auto) probes the server once with a 1-token request: auto requests ids only if the probe returns them, on
fails when it does not, off never asks. Against a server without the extension the ids are simply absent and the
text/deltas/usage criteria apply as before (a ref/test pair compares ids only when both runs have them).
cached_tokens is recorded and reported next to every mismatch, because a different KV prefix reuse shifts the
prefill chunk boundaries.
"""
import argparse
import collections
import hashlib
import http.client
import json
import os
import pathlib
import random
import socket
import sys
import threading
import time
import urllib.parse

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_CASES = HERE / "prompts" / "cb_identity_cases.jsonl"
DEFAULT_TOKENIZER = pathlib.Path.home() / "models" / "Qwen3-8B" / "tokenizer.json"
FLUSH_ID = 151643  # Qwen <|endoftext|>: shares no prefix with any real prompt (raw text or <|im_start|> chat)


# ----------------------------------------------------------------------------------------------------------------------
# cases
# ----------------------------------------------------------------------------------------------------------------------

def resolve_text(spec, prompts_dir):
    """A prompt/content is a string, or {"nonce", "file", "char_start", "char_len", "suffix"} (corpus slice)."""
    if isinstance(spec, str):
        return spec
    text = (prompts_dir / spec["file"]).read_text()
    s = spec["char_start"]
    return spec.get("nonce", "") + text[s : s + spec["char_len"]] + spec.get("suffix", "")


def load_cases(path, prompts_dir=None):
    prompts_dir = pathlib.Path(prompts_dir) if prompts_dir else pathlib.Path(path).resolve().parent
    cases = []
    for ln, line in enumerate(pathlib.Path(path).read_text().splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        c = json.loads(line)
        for k in ("id", "max_tokens", "temperature"):
            if k not in c:
                raise SystemExit(f"{path}:{ln}: case needs '{k}' (greedy cases must say temperature 0 explicitly: the server defaults to 1.0)")
        if c["temperature"] != 0 and "seed" not in c:
            raise SystemExit(f"{path}:{ln}: sampled case {c['id']} needs a seed")
        if ("prompt" in c) == ("messages" in c):
            raise SystemExit(f"{path}:{ln}: case {c['id']} needs exactly one of prompt / messages")
        if "prompt" in c:
            c["_text"] = resolve_text(c["prompt"], prompts_dir)
        else:
            c["_messages"] = [{"role": m["role"], "content": resolve_text(m["content"], prompts_dir)} for m in c["messages"]]
        cases.append(c)
    ids = [c["id"] for c in cases]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{path}: duplicate case ids")
    return cases


def is_cancel_case(c):
    return c.get("cancel_after_ms") is not None or c.get("cancel_after_chunks") is not None


def build_body(c, stream, model, cap_max_tokens=None):
    max_tokens = c["max_tokens"] if cap_max_tokens is None else min(c["max_tokens"], cap_max_tokens)
    b = {"model": model, "max_tokens": max_tokens, "temperature": c["temperature"], "stream": stream}
    if c.get("min_tokens"):
        b["min_tokens"] = c["min_tokens"]
    for k in ("top_p", "top_k", "seed"):
        if c.get(k) is not None:
            b[k] = c[k]
    if c.get("stop"):
        b["stop"] = c["stop"]
    if c.get("ignore_eos"):
        b["ignore_eos"] = True
    if stream:
        b["stream_options"] = {"include_usage": True}
    if "_messages" in c:
        b["messages"] = c["_messages"]
        b["chat_template_kwargs"] = {"enable_thinking": bool(c.get("enable_thinking", False))}
        return "/v1/chat/completions", b
    b["prompt"] = c["_text"]
    return "/v1/completions", b


# ----------------------------------------------------------------------------------------------------------------------
# HTTP client
# ----------------------------------------------------------------------------------------------------------------------

class Cancel(Exception):
    pass


def split_url(url):
    u = urllib.parse.urlparse(url)
    return u.hostname, u.port or 80


def get_json(url, path, timeout=10):
    host, port = split_url(url)
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        conn.request("GET", path)
        r = conn.getresponse()
        data = r.read()
        return json.loads(data) if r.status == 200 else {"_http_status": r.status}
    except Exception as e:  # noqa: BLE001
        return {"_error": repr(e)}
    finally:
        conn.close()


def one_request(url, path, body, stream, timeout, cancel_ms=None, cancel_chunks=None):
    """Run one request. Returns a record dict. Streams are read event by event; a cancel closes the socket."""
    host, port = split_url(url)
    rec = {"stream": stream, "status": "ok", "text": "", "deltas": [] if stream else None, "finish_reason": None,
           "usage": None, "http_status": None, "error": None, "cancel": None, "ids": None}
    want_ids = bool(body.get("return_token_ids"))
    t_send = time.monotonic()
    rec["t_send"] = time.time()
    deadline = t_send + cancel_ms / 1000.0 if cancel_ms is not None else None
    conn = http.client.HTTPConnection(host, port, timeout=timeout)

    def arm():
        if conn.sock is None:
            return
        if deadline is None:
            conn.sock.settimeout(timeout)
        else:
            left = deadline - time.monotonic()
            if left <= 0:
                raise Cancel("ms")
            conn.sock.settimeout(left)

    try:
        payload = json.dumps(body).encode()
        conn.request("POST", path, body=payload, headers={"Content-Type": "application/json", "Connection": "close"})
        arm()
        resp = conn.getresponse()
        rec["http_status"] = resp.status
        rec["t_headers_ms"] = round((time.monotonic() - t_send) * 1000, 1)
        if resp.status != 200:
            rec["status"] = "error"
            rec["error"] = f"HTTP {resp.status}: {resp.read(2000).decode(errors='replace')}"
        elif not stream:
            arm()
            obj = json.loads(resp.read())
            ch = obj["choices"][0]
            rec["text"] = ch["message"]["content"] if "message" in ch else ch["text"]
            rec["finish_reason"] = ch.get("finish_reason")
            rec["usage"] = obj.get("usage")
            if want_ids and isinstance(ch.get("token_ids"), list):
                rec["ids"] = ch["token_ids"]
        else:
            while True:
                arm()
                line = resp.readline()
                if not line:
                    break
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    rec["done"] = True
                    break
                obj = json.loads(data)
                if "error" in obj:
                    rec["status"] = "error"
                    rec["error"] = json.dumps(obj["error"])
                    break
                if obj.get("usage"):
                    rec["usage"] = obj["usage"]
                for ch in obj.get("choices") or []:
                    tid = (ch["delta"] if "delta" in ch else ch).get("token_ids")
                    if want_ids and isinstance(tid, list):
                        if rec["ids"] is None:
                            rec["ids"] = []
                        rec["ids"].extend(tid)
                    d = ch["delta"].get("content") if "delta" in ch else ch.get("text")
                    if d:
                        rec["deltas"].append(d)
                        rec["text"] += d
                    if ch.get("finish_reason"):
                        rec["finish_reason"] = ch["finish_reason"]
                if cancel_chunks is not None and len(rec["deltas"]) >= cancel_chunks:
                    raise Cancel("chunks")
            if rec["status"] == "ok" and not rec.get("done"):
                rec["status"] = "error"
                rec["error"] = "stream ended without [DONE]"
    except Cancel as c:
        rec["status"] = "cancelled"
        rec["cancel"] = str(c)
    except socket.timeout:
        if deadline is not None and time.monotonic() >= deadline - 0.05:
            rec["status"] = "cancelled"
            rec["cancel"] = "ms"
        else:
            rec["status"] = "error"
            rec["error"] = f"timeout after {timeout}s"
    except Exception as e:  # noqa: BLE001
        rec["status"] = "error"
        rec["error"] = repr(e)
    finally:
        if rec["status"] == "cancelled" and conn.sock is not None:
            try:
                conn.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        conn.close()
    rec["t_total_ms"] = round((time.monotonic() - t_send) * 1000, 1)
    return rec


def want_stream(case, mode, force=None):
    if force is not None:
        return force
    if is_cancel_case(case):
        return True  # a cancel is only observable on a stream
    if mode == "on":
        return True
    if mode == "off":
        return False
    return int(hashlib.sha256(case["id"].encode()).hexdigest(), 16) % 2 == 0  # mixed: deterministic half/half


# ----------------------------------------------------------------------------------------------------------------------
# run
# ----------------------------------------------------------------------------------------------------------------------

def select_cases(cases, sample=None, ids=None):
    if ids:
        want = set(ids.split(","))
        cases = [c for c in cases if c["id"] in want]
    if sample and sample < len(cases):
        # stratified by the first tag (case family): proportional share, at least 2 per family, evenly spaced inside it
        fams = collections.OrderedDict()
        for c in cases:
            fams.setdefault((c.get("tags") or ["-"])[0], []).append(c)
        keep = set()
        for fam in fams.values():
            q = min(len(fam), max(2, round(sample * len(fam) / len(cases))))
            keep.update(fam[int(i * len(fam) / q)]["id"] for i in range(q))
        cases = [c for c in cases if c["id"] in keep]
    return cases


def server_info(url):
    return {"url": url, "plan": get_json(url, "/v1/intarsia/plan"), "models": get_json(url, "/v1/models"),
            "health": get_json(url, "/health")}


def model_name(info):
    try:
        return info["models"]["data"][0]["id"]
    except Exception:  # noqa: BLE001
        return "intarsia"


def probe_token_ids(url, model, timeout=60):
    """True when the server honours `return_token_ids` (1-token greedy completion carries `token_ids`)."""
    rec = one_request(url, "/v1/completions", {"model": model, "prompt": "Hello", "max_tokens": 1, "temperature": 0,
                                               "return_token_ids": True}, False, timeout)
    return rec["status"] == "ok" and bool(rec["ids"])


def run_cases(args, cases, mode):
    info = server_info(args.url)
    model = model_name(info)
    tmode = getattr(args, "token_ids", "auto")
    use_ids = False
    if tmode != "off":
        use_ids = probe_token_ids(args.url, model)
        if tmode == "on" and not use_ids:
            raise SystemExit(f"--token-ids on: {args.url} did not return token_ids for return_token_ids (server without the extension?)")
    print(f"token ids: {'requested' if use_ids else 'not available/not requested'} ({args.url})", flush=True)
    cap = getattr(args, "cap_max_tokens", None)
    solo = mode == "solo"
    flush = args.flush == "on" or (args.flush == "auto" and solo)
    records = {}
    lock = threading.Lock()
    inflight = [0, 0]  # current, peak
    t0 = time.monotonic()

    def work(c):
        stream = want_stream(c, args.stream, force=True if (solo and args.stream != "off") else None)
        path, body = build_body(c, stream, model, cap)
        if use_ids:
            body["return_token_ids"] = True
        if solo:
            if flush:  # drop the KV prefix so no case is prefilled from a reused prefix
                one_request(args.url, "/v1/completions", {"model": model, "prompt": [args.flush_id], "max_tokens": 1,
                                                          "temperature": 0}, False, args.timeout)
            rec = one_request(args.url, path, body, stream, args.timeout)  # cancels are not applied to the reference
        else:
            with lock:
                inflight[0] += 1
                inflight[1] = max(inflight[1], inflight[0])
            try:
                rec = one_request(args.url, path, body, stream, args.timeout, c.get("cancel_after_ms"), c.get("cancel_after_chunks"))
            finally:
                with lock:
                    inflight[0] -= 1
        rec["id"] = c["id"]
        rec["arrival_ms"] = c.get("arrival_ms", 0)
        rec["t_start_ms"] = round((time.monotonic() - t0) * 1000, 1)
        with lock:
            records[c["id"]] = rec
        tag = rec["status"] if rec["status"] != "ok" else rec["finish_reason"]
        print(f"  [{len(records):3d}/{len(cases)}] {c['id']} {'S' if stream else 'N'} {tag} "
              f"{(rec.get('usage') or {}).get('completion_tokens', len(rec['text']))} {rec['t_total_ms'] / 1000:.1f}s", flush=True)

    if solo:
        for c in cases:
            work(c)
    else:
        sem = threading.Semaphore(args.b)
        threads = []
        for c in sorted(cases, key=lambda c: c.get("arrival_ms", 0)):
            wait = t0 + c.get("arrival_ms", 0) * args.arrival_scale / 1000.0 - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            sem.acquire()

            def go(c=c):
                try:
                    work(c)
                finally:
                    sem.release()

            th = threading.Thread(target=go, daemon=True)
            th.start()
            threads.append(th)
        for th in threads:
            th.join()
    header = {"mode": mode, "b": None if solo else args.b, "stream": args.stream, "arrival_scale": args.arrival_scale,
              "cap_max_tokens": cap, "token_ids": use_ids, "flush": flush, "n_cases": len(cases), "peak_client_inflight": inflight[1],
              "wall_s": round(time.monotonic() - t0, 1), "cases_sha256": getattr(args, "cases_sha", None),
              "server": info, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "argv": sys.argv}
    return header, [records[c["id"]] for c in cases if c["id"] in records]


def write_run(path, header, recs):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(json.dumps({"_header": header}) + "\n")
        for r in recs:
            f.write(json.dumps(r) + "\n")


def read_run(path):
    header, recs = None, {}
    for line in pathlib.Path(path).read_text().splitlines():
        o = json.loads(line)
        if "_header" in o:
            header = o["_header"]
        else:
            recs[o["id"]] = o
    return header or {}, recs


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()[:16]


def cmd_run(args):
    cases = select_cases(load_cases(args.cases), args.sample, args.ids)
    args.cases_sha = sha(args.cases)
    print(f"{args.mode}: {len(cases)} cases -> {args.url}" + ("" if args.mode == "solo" else f", B={args.b}, stream={args.stream}"))
    header, recs = run_cases(args, cases, args.mode)
    write_run(args.out, header, recs)
    bad = [r["id"] for r in recs if r["status"] == "error"]
    print(f"wrote {args.out}: {len(recs)} records, {len(bad)} errors, wall {header['wall_s']}s, peak client in-flight {header['peak_client_inflight']}")
    return 1 if bad and args.mode == "solo" else 0


# ----------------------------------------------------------------------------------------------------------------------
# compare
# ----------------------------------------------------------------------------------------------------------------------

def first_diff(a, b):
    n = min(len(a), len(b))
    i = next((k for k in range(n) if a[k] != b[k]), n)
    return i


def comp_tokens(r):
    return (r.get("usage") or {}).get("completion_tokens")


def compare_case(c, r, t):
    """-> (verdict, reasons, info). verdict: pass | mismatch | ref_error | missing."""
    if r is None:
        return "missing", ["no reference record"], {}
    if r["status"] != "ok":
        return "ref_error", [f"reference {r['status']}: {r.get('error')}"], {}
    info = {"cached_ref": (r.get("usage") or {}).get("prompt_tokens_details", {}).get("cached_tokens"),
            "cached_test": ((t.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens")}
    info["cached_differs"] = None not in (info["cached_ref"], info["cached_test"]) and info["cached_ref"] != info["cached_test"]
    reasons = []
    if t["status"] == "error":
        return "mismatch", [f"test error: {t.get('error')}"], info
    if t["status"] == "cancelled":
        info["cancelled"] = True
        info["emitted_bytes"] = len(t["text"].encode())
        if not r["text"].startswith(t["text"]):
            i = first_diff(r["text"], t["text"])
            reasons.append(f"cancelled prefix is not a prefix of the reference text (first diff at char {i}: ref {r['text'][i:i + 30]!r} vs {t['text'][i:i + 30]!r})")
        ri, ti = r.get("ids"), t.get("ids")
        if ri is not None and ti is not None:
            info["ids_compared"] = True
            if ri[: len(ti)] != ti:
                reasons.append(f"cancelled token ids are not a prefix of the reference ids (first diff at token {first_diff(ri, ti)})")
        rd, td = r.get("deltas"), t.get("deltas")
        if rd is not None and td is not None and rd[: len(td)] != td:
            reasons.append(f"cancelled deltas are not a prefix of the reference deltas (first diff at delta {first_diff(rd, td)})")
        return ("mismatch" if reasons else "pass"), reasons, info
    # completed: exact comparison; token ids first, they are the primary criterion
    ri, ti = r.get("ids"), t.get("ids")
    if ri is not None and ti is not None:
        info["ids_compared"] = True
        if ri != ti:
            i = first_diff(ri, ti)
            reasons.append(f"TOKEN IDS differ at token {i}/{len(ri)} (ref {ri[max(0, i - 2):i + 4]} vs test {ti[max(0, i - 2):i + 4]}; {len(ri)} ref vs {len(ti)} test)")
    if r["text"].encode() != t["text"].encode():
        i = first_diff(r["text"], t["text"])
        reasons.append(f"text differs at char {i}/{len(r['text'])} (ref {r['text'][max(0, i - 10):i + 30]!r} vs test {t['text'][max(0, i - 10):i + 30]!r})")
    if r.get("deltas") is not None and t.get("deltas") is not None and r["deltas"] != t["deltas"]:
        reasons.append(f"streamed deltas differ at delta {first_diff(r['deltas'], t['deltas'])} ({len(r['deltas'])} ref vs {len(t['deltas'])} test)")
    if r["finish_reason"] != t["finish_reason"]:
        reasons.append(f"finish_reason {r['finish_reason']} vs {t['finish_reason']}")
    if comp_tokens(r) != comp_tokens(t):
        reasons.append(f"completion_tokens {comp_tokens(r)} vs {comp_tokens(t)}")
    if (r.get("usage") or {}).get("prompt_tokens") != (t.get("usage") or {}).get("prompt_tokens"):
        reasons.append(f"prompt_tokens {(r.get('usage') or {}).get('prompt_tokens')} vs {(t.get('usage') or {}).get('prompt_tokens')}")
    return ("mismatch" if reasons else "pass"), reasons, info


PROMPT_BUCKETS = [(0, 192, "<=192"), (193, 384, "193-384"), (385, 768, "385-768"), (769, 1536, "769-1.5k"),
                  (1537, 2560, "1.5k-2.5k"), (2561, 3584, "2.5k-3.5k"), (3585, 10**9, ">3.5k")]


def coverage(cases, ref):
    cov = collections.OrderedDict()
    tags = collections.Counter()
    for c in cases:
        for tg in c.get("tags", []):
            tags[tg] += 1
    cov["cases"] = len(cases)
    cov["tags"] = dict(sorted(tags.items()))
    fin = collections.Counter()
    hist = collections.Counter()
    ptoks = []
    for c in cases:
        r = ref.get(c["id"])
        if not r or r["status"] != "ok":
            continue
        pt = (r.get("usage") or {}).get("prompt_tokens")
        if pt is not None:
            ptoks.append(pt)
            hist[next(lbl for lo, hi, lbl in PROMPT_BUCKETS if lo <= pt <= hi)] += 1
        if r["finish_reason"] == "length":
            fin["length"] += 1
        elif c.get("stop"):
            fin["stop (stop string armed: stop string or EOS)"] += 1
        else:
            fin["stop (EOS)"] += 1
    cov["reference_finish"] = dict(fin)
    cov["prompt_tokens_histogram"] = {lbl: hist.get(lbl, 0) for _, _, lbl in PROMPT_BUCKETS}
    cov["prompt_tokens_min_max"] = [min(ptoks), max(ptoks)] if ptoks else None
    return cov


def compare_runs(cases_by_id, ref_hdr, ref, test_files, expect_cap=None):
    out = {"warnings": [], "runs": []}
    for tf in test_files:
        th, tr = read_run(tf)
        run = {"file": str(tf), "header": {k: v for k, v in th.items() if k != "server"}, "rows": [], "mismatches": []}
        rp, tp = (ref_hdr.get("server") or {}).get("plan"), (th.get("server") or {}).get("plan")
        if rp and tp and rp != tp:
            diffs = sorted(k for k in set(rp) | set(tp) if rp.get(k) != tp.get(k))
            out["warnings"].append(f"{tf}: server plan differs from the reference server on {diffs}")
            if "prefill_chunk" in diffs:
                out["warnings"].append(f"{tf}: PREFILL CHUNK DIFFERS ({rp.get('prefill_chunk')} reference vs {tp.get('prefill_chunk')} test): the chunk schedule must match (design 8.1)")
        if ref_hdr.get("cases_sha256") and th.get("cases_sha256") and ref_hdr["cases_sha256"] != th["cases_sha256"]:
            out["warnings"].append(f"{tf}: case file hash differs from the reference run")
        counts = collections.Counter()
        for cid, t in tr.items():
            c = cases_by_id.get(cid, {"id": cid})
            verdict, reasons, info = compare_case(c, ref.get(cid), t)
            row = {"id": cid, "verdict": verdict, "stream": t["stream"], "status": t["status"], "reasons": reasons,
                   "tags": c.get("tags", []), "cancel_case": is_cancel_case(c), **info}
            run["rows"].append(row)
            counts[verdict] += 1
            counts["stream" if t["stream"] else "nonstream"] += 1
            if verdict != "pass":
                run["mismatches"].append(row)
            if info.get("cancelled"):
                counts["cancelled"] += 1
                counts["cancelled_with_output"] += 1 if info.get("emitted_bytes") else 0
            elif is_cancel_case(c):
                counts["cancel_case_completed_first"] += 1
            if info.get("ids_compared"):
                counts["ids_compared"] += 1
            if info.get("cached_differs"):
                counts["cached_differs"] += 1
                if verdict != "pass":
                    counts["mismatch_with_cached_diff"] += 1
        if counts["ids_compared"] == 0:
            out["warnings"].append(f"{tf}: token ids were NOT compared (reference ids: {bool(ref_hdr.get('token_ids'))}, test ids: {bool(th.get('token_ids'))}); text/deltas/usage criteria only")
        counts["exact_compared"] = sum(1 for r in run["rows"] if r["status"] == "ok")
        counts["ref_only"] = len(set(ref) - set(tr))
        run["counts"] = dict(counts)
        if expect_cap is not None:  # perturbation self-test: the mismatch set must be exactly the predicted one
            exp = {cid for cid, r in ref.items() if cid in tr and r["status"] == "ok" and not is_cancel_case(cases_by_id.get(cid, {}))
                   and (comp_tokens(r) or 0) > expect_cap}
            got = {m["id"] for m in run["mismatches"] if not m["cancel_case"]}
            run["expect_cap"] = {"cap": expect_cap, "expected": len(exp), "got": len(got), "equal": exp == got,
                                 "missing": sorted(exp - got), "unexpected": sorted(got - exp)}
        out["runs"].append(run)
    return out


def fmt_run_label(run):
    h = run["header"]
    if h.get("mode") == "solo":
        return "solo"
    return f"B={h.get('b')} {h.get('stream')}"


def print_report(rep, cov, ref_hdr):
    print("\n== coverage (from the reference run) ==")
    print(f"cases {cov['cases']}  prompt tokens min/max {cov['prompt_tokens_min_max']}")
    print("prompt-token histogram:", cov["prompt_tokens_histogram"])
    print("reference finish:", cov["reference_finish"])
    print("tags:", cov["tags"])
    for w in rep["warnings"]:
        print("WARNING:", w)
    print("\n== identity vs reference ==")
    cols = ["run", "cases", "pass", "mismatch", "ref_err", "stream/non", "cancel(out>0)/cases", "ids cmp", "cached!=", "mism+cached!="]
    rows = []
    for run in rep["runs"]:
        c = run["counts"]
        ncancel_cases = sum(1 for r in run["rows"] if r["cancel_case"])
        rows.append([fmt_run_label(run), len(run["rows"]), c.get("pass", 0), c.get("mismatch", 0),
                     c.get("ref_error", 0) + c.get("missing", 0), f"{c.get('stream', 0)}/{c.get('nonstream', 0)}",
                     f"{c.get('cancelled', 0)}({c.get('cancelled_with_output', 0)})/{ncancel_cases}", c.get("ids_compared", 0), c.get("cached_differs", 0),
                     c.get("mismatch_with_cached_diff", 0)])
    w = [max(len(str(x)) for x in [cols[i]] + [r[i] for r in rows]) for i in range(len(cols))]
    print("  ".join(cols[i].ljust(w[i]) for i in range(len(cols))))
    for r in rows:
        print("  ".join(str(r[i]).ljust(w[i]) for i in range(len(cols))))
    for run in rep["runs"]:
        for m in run["mismatches"][:15]:
            print(f"  MISMATCH {fmt_run_label(run)} {m['id']} ({m['verdict']}, {'stream' if m['stream'] else 'non-stream'}, cached ref/test "
                  f"{m.get('cached_ref')}/{m.get('cached_test')}): " + "; ".join(m["reasons"]))
        if len(run["mismatches"]) > 15:
            print(f"  ... {len(run['mismatches']) - 15} more in the JSON report")
        if "expect_cap" in run:
            e = run["expect_cap"]
            print(f"  perturbation check (cap max_tokens {e['cap']}): expected {e['expected']} mismatches, got {e['got']}, "
                  f"{'EXACT MATCH' if e['equal'] else 'DIFFERENT: missing ' + str(e['missing']) + ' unexpected ' + str(e['unexpected'])}")


def cmd_compare(args):
    cases = load_cases(args.cases) if pathlib.Path(args.cases).exists() else []
    cbi = {c["id"]: c for c in cases}
    ref_hdr, ref = read_run(args.ref)
    rep = compare_runs(cbi, ref_hdr, ref, args.test, args.expect_cap)
    cov = coverage([cbi.get(cid, {"id": cid}) for cid in ref], ref)
    rep["coverage"] = cov
    rep["ref"] = {"file": args.ref, "header": {k: v for k, v in ref_hdr.items() if k != "server"}}
    print_report(rep, cov, ref_hdr)
    if args.report:
        pathlib.Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.report).write_text(json.dumps(rep, indent=1))
        print("report:", args.report)
    nbad = sum(len(r["mismatches"]) for r in rep["runs"])
    if args.expect_cap is not None:
        ok = all(r["expect_cap"]["equal"] for r in rep["runs"])
        print("PERTURBATION SELF-TEST:", "PASS (mismatch set equals the predicted set)" if ok else "FAIL")
        return 0 if ok else 1
    print("RESULT:", "IDENTICAL, 0 mismatches" if nbad == 0 else f"FAIL, {nbad} mismatches")
    return 1 if nbad else 0


def cmd_gate(args):
    """reference run + one concurrent run per (B, stream mode) + compare."""
    outdir = pathlib.Path(args.outdir)
    cases = select_cases(load_cases(args.cases), args.sample, args.ids)
    args.cases_sha = sha(args.cases)
    ref_path = pathlib.Path(args.ref_file) if args.ref_file else outdir / "ref.jsonl"
    if not args.ref_file:
        ra = argparse.Namespace(**vars(args))
        ra.url = args.ref_url or args.url
        print(f"reference: {len(cases)} cases alone -> {ra.url}")
        h, recs = run_cases(ra, cases, "solo")
        write_run(ref_path, h, recs)
        if any(r["status"] == "error" for r in recs):
            print("reference run has errors; fix the server/case set first")
            return 2
    tests = []
    for b in [int(x) for x in args.b.split(",")]:
        for sm in args.streams.split(","):
            ta = argparse.Namespace(**vars(args))
            ta.b, ta.stream = b, sm
            print(f"concurrent B={b} stream={sm}: {len(cases)} cases -> {args.url}")
            h, recs = run_cases(ta, cases, "concurrent")
            p = outdir / f"conc-b{b}-{sm}.jsonl"
            write_run(p, h, recs)
            tests.append(str(p))
    ca = argparse.Namespace(cases=args.cases, ref=str(ref_path), test=tests, report=str(outdir / "report.json"), expect_cap=None)
    return cmd_compare(ca)


# ----------------------------------------------------------------------------------------------------------------------
# gen-cases
# ----------------------------------------------------------------------------------------------------------------------

# frequent strings (fire within a few dozen tokens of corpus text) and rare ones (never fire, exercise the held-back-text path)
STOPS_FREQUENT = [["\n"], [" the"], ["."], ["e "], [", "], [" of the"], [";", "\n\n"]]
STOPS_RARE = [["\n\n\n"], ["def "], ["Q:"]]
STOPS_CHAT = [["."], [" the"], [","]]
YESNO = ["Reply with only the word YES or NO: is the text above written in English?",
         "Reply with one word: what is the topic of the text above?",
         "Answer in a few words only: what language is the text above about?",
         "Reply with the single word OK."]
OPEN_Q = ["Summarize the text above in two sentences.", "List three facts from the text above.",
          "Continue the text above with one more paragraph."]
SIZES = [128, 256, 512, 1024, 2048, 3072, 4000]
SIZE_W = [14, 14, 17, 17, 15, 12, 11]


def cmd_gen_cases(args):
    try:
        from tokenizers import Tokenizer
    except ImportError:
        raise SystemExit("gen-cases needs the tokenizers package: uv run --with tokenizers tools/cbidentity.py gen-cases ...")
    tok = Tokenizer.from_file(str(args.tokenizer))
    rng = random.Random(args.seed)
    pdir = HERE / "prompts"
    corp = {f: (pdir / f).read_text() for f in ("corpus_docs.txt", "corpus_code.txt")}
    used_first = set()
    vocab = sorted(w for w in tok.get_vocab() if w.isascii() and w.isalpha() and 4 <= len(w) <= 9)

    def nonce():
        """A leading word whose first token no other case starts with, so cases share no KV prefix."""
        while True:
            w = rng.choice(vocab)
            first = tok.encode(w + " notes: ", add_special_tokens=False).ids[0]
            if first not in used_first:
                used_first.add(first)
                return f"{w} notes: "

    def slice_spec(target, nonce_s, suffix=""):
        """A corpus slice whose nonce+slice+suffix tokenizes to about `target` tokens."""
        f = rng.choice(["corpus_docs.txt", "corpus_docs.txt", "corpus_code.txt"])
        text = corp[f]
        n0 = len(tok.encode(nonce_s, add_special_tokens=False).ids) + len(tok.encode(suffix, add_special_tokens=False).ids)
        need = max(target - n0, 8)
        while True:
            start = rng.randrange(0, max(1, len(text) - 12 * need))
            nl = text.find("\n", start)
            start = nl + 1 if 0 <= nl < start + 200 else start
            enc = tok.encode(text[start : start + 12 * need], add_special_tokens=False)
            if len(enc.ids) > need:
                return {"nonce": nonce_s, "file": f, "char_start": start, "char_len": enc.offsets[need - 1][1], "suffix": suffix}

    def sampled(c):
        c.update(temperature=rng.choice([0.6, 0.8, 1.0]), top_p=rng.choice([1.0, 0.95, 0.9]), top_k=rng.choice([0, 40, 20]),
                 seed=rng.randrange(1, 2**31))
        c["tags"].append("sampled")

    def greedy(c):
        c["temperature"] = 0
        c["tags"].append("greedy")

    n = args.count
    fams = [("raw_greedy", 0.27), ("raw_sampled", 0.15), ("chat_greedy", 0.08), ("chat_sampled", 0.04), ("eos", 0.08), ("min_tokens", 0.08),
            ("stop", 0.12), ("cancel", 0.10), ("ignore_eos", 0.04)]
    plan = []
    for name, frac in fams:
        plan += [name] * round(n * frac)
    plan += ["raw_greedy"] * (n - len(plan))
    plan = plan[:n]
    rng.shuffle(plan)
    cases = []
    t = 0
    for i, fam in enumerate(plan):
        size = rng.choices(SIZES, SIZE_W)[0]
        c = {"id": f"cb{i:03d}", "tags": [fam], "max_tokens": rng.choice([16, 32, 48, 64, 96, 128]), "min_tokens": 0}
        chat = fam in ("chat_greedy", "chat_sampled", "eos", "min_tokens") or (fam in ("cancel", "stop") and rng.random() < 0.4)
        if chat:
            q = rng.choice(YESNO) if fam in ("eos", "min_tokens") else rng.choice(OPEN_Q)
            ns = nonce()
            c["messages"] = [{"role": "system", "content": "You are a careful assistant. Be brief."},
                             {"role": "user", "content": slice_spec(size, ns, "\n\n" + q)}]
            c["tags"].append("chat")
        else:
            c["prompt"] = slice_spec(size, nonce())
            c["tags"].append("raw")
        if fam in ("raw_sampled", "chat_sampled"):
            sampled(c)
        else:
            greedy(c)
        if fam == "eos":
            c["max_tokens"] = 160
        if fam == "min_tokens":
            c["max_tokens"] = 160
            c["min_tokens"] = rng.choice([12, 24, 40])
        if fam == "ignore_eos":
            c["ignore_eos"] = True
        if fam == "stop":
            c["stop"] = rng.choice(STOPS_CHAT if chat else STOPS_FREQUENT if rng.random() < 0.8 else STOPS_RARE)
            c["max_tokens"] = 96
            if rng.random() < 0.4:
                sampled(c)
                c["tags"].remove("greedy")
        if fam == "cancel":
            c["max_tokens"] = 128
            if rng.random() < 0.75:
                c["cancel_after_chunks"] = rng.choice([1, 2, 5, 12, 30])
            else:
                c["cancel_after_ms"] = rng.choice([30, 150, 600, 1500, 4000])
        t += 0 if rng.random() < 0.3 else int(rng.expovariate(1 / 500))
        c["arrival_ms"] = t
        cases.append(c)
    # the three golden_4k.jsonl prompts (~4k tokens, a multi-turn chat), appended after the random cases so cb000.. do not change
    for g in (json.loads(x) for x in (pdir / "golden_4k.jsonl").read_text().splitlines() if x.strip()):
        gid = f"cb{len(cases):03d}"
        t += int(rng.expovariate(1 / 500))
        c = {"id": gid, "tags": ["golden_4k"], "max_tokens": g.get("continue", 64), "min_tokens": 0, "temperature": 0, "arrival_ms": t}
        if g["kind"] == "raw":
            ns = nonce()
            text = corp[g["file"]]
            n0 = len(tok.encode(ns, add_special_tokens=False).ids)
            enc = tok.encode(text, add_special_tokens=False)
            c["prompt"] = {"nonce": ns, "file": g["file"], "char_start": 0, "char_len": enc.offsets[g["max_tokens"] - n0 - 1][1], "suffix": ""}
            c["tags"].append("raw")
        else:  # chat_reuse_4k: all four messages; reuse_after_message (prefix reuse) is deliberately ignored
            msgs = []
            for i, m in enumerate(g["messages"]):
                content = m.get("content")
                if "content_file" in m:
                    content = {"file": m["content_file"], "char_start": m.get("skip_chars", 0), "char_len": m["max_chars"]}
                elif i == 0:
                    content = nonce() + content
                msgs.append({"role": m["role"], "content": content})
            c["messages"] = msgs
            c["tags"].append("chat")
        c["tags"].append("greedy")
        cases.append(c)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.write("# cb_identity cases: generated by `tools/cbidentity.py gen-cases` (seed %d, tokenizer %s). Prompts are corpus slices\n" % (args.seed, pathlib.Path(args.tokenizer).parent.name))
        f.write("# (file + char offsets resolved against tools/prompts/) with a unique leading nonce word so no two cases share a KV prefix.\n")
        for c in cases:
            f.write(json.dumps(c) + "\n")
    cc = collections.Counter(tag for c in cases for tag in c["tags"])
    print(f"wrote {args.out}: {len(cases)} cases", dict(cc))
    return 0


# ----------------------------------------------------------------------------------------------------------------------
# mock server (offline self-test): deterministic fake model with the real server's response shapes
# ----------------------------------------------------------------------------------------------------------------------

def make_mock(bug=None, serial=False, delay=0.002, ids=True):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    state = {"inflight": 0, "lock": threading.Lock(), "serial": threading.Lock(), "bug_fired": 0}

    def token(key, seed, pos):
        h = hashlib.sha256(f"{key}|{seed}|{pos}".encode()).hexdigest()
        return " " + ["alpha", "beta", "gamma", "delta", "eps", "zeta", "eta", "theta"][int(h[:2], 16) % 8] + h[2:4]

    def token_id(key, seed, pos):
        return int(hashlib.sha256(f"id|{key}|{seed}|{pos}".encode()).hexdigest()[:6], 16)

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _json(self, code, obj):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path == "/v1/intarsia/plan":
                return self._json(200, {"prefill_chunk": 256, "mock": True})
            if self.path == "/v1/models":
                return self._json(200, {"object": "list", "data": [{"id": "mock"}]})
            return self._json(200, {"status": "ok"})

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            chat = self.path.endswith("chat/completions")
            text = json.dumps(req["messages"]) if chat else (req["prompt"] if isinstance(req["prompt"], str) else str(req["prompt"]))
            stream = req.get("stream", False)
            if serial:
                state["serial"].acquire()
            try:
                with state["lock"]:
                    state["inflight"] += 1
                self._gen(req, chat, text, stream)
            finally:
                with state["lock"]:
                    state["inflight"] -= 1
                if serial:
                    state["serial"].release()

        def _gen(self, req, chat, text, stream):
            key = hashlib.sha256(text.encode()).hexdigest()
            seed = req.get("seed") if req.get("temperature", 1.0) else 0
            eos_at = 4 + int(key[:2], 16) % 30 if ("Reply with" in text or "[eos]" in text) else None
            max_t, min_t, stops = req.get("max_tokens", 64), req.get("min_tokens", 0), req.get("stop") or []
            if isinstance(stops, str):
                stops = [stops]
            ptoks = len(text) // 4
            out, deltas, finish, n = "", [], "length", 0
            want_ids = ids and bool(req.get("return_token_ids"))
            tids = []
            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()

            def send(obj):
                self.wfile.write(b"data: " + (obj if isinstance(obj, bytes) else json.dumps(obj).encode()) + b"\n\n")
                self.wfile.flush()

            def chunk(d, fin=None, tid=None):
                extra = {"token_ids": [tid]} if tid is not None else {}
                return {"id": "x", "object": "chat.completion.chunk" if chat else "text_completion",
                        "choices": [{"index": 0, **({"delta": {"content": d, **extra}} if chat else {"text": d, **extra}), "finish_reason": fin}]}

            try:
                for pos in range(max_t):
                    if eos_at is not None and pos >= eos_at and pos >= min_t and not req.get("ignore_eos"):
                        finish = "stop"
                        n += 1  # EOS counts, is not emitted
                        break
                    tk = token(key, seed, pos)
                    tid = token_id(key, seed, pos)
                    if bug == "inflight" and state["inflight"] >= 2 and pos == 3:
                        tk = " BUG"
                        tid = 999999
                        state["bug_fired"] += 1
                    elif bug == "inflight-id" and state["inflight"] >= 2 and pos == 3:
                        tid = 999999
                        state["bug_fired"] += 1
                    time.sleep(delay)
                    n += 1
                    tids.append(tid)
                    cand = out + tk
                    hit = [s for s in stops if s in cand]
                    if hit:
                        cut = min(cand.index(s) for s in hit)
                        tk = cand[len(out):cut] if cut > len(out) else ""
                        out = cand[:cut]
                        finish = "stop"
                        if stream and (tk or want_ids):
                            send(chunk(tk, tid=tid if want_ids else None))
                            if tk:
                                deltas.append(tk)
                        break
                    out = cand
                    deltas.append(tk)
                    if stream:
                        send(chunk(tk, tid=tid if want_ids else None))
                usage = {"prompt_tokens": ptoks, "completion_tokens": n, "total_tokens": ptoks + n, "prompt_tokens_details": {"cached_tokens": 0}}
                if stream:
                    send(chunk("", finish))
                    if (req.get("stream_options") or {}).get("include_usage"):
                        send({"id": "x", "choices": [], "usage": usage})
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                else:
                    ch = {"index": 0, "finish_reason": finish, **({"message": {"role": "assistant", "content": out}} if chat else {"text": out}),
                          **({"token_ids": tids} if want_ids else {})}
                    self._json(200, {"id": "x", "choices": [ch], "usage": usage})
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    srv.daemon_threads = True
    srv.state = state
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def cmd_mock(args):
    srv, url = make_mock(args.bug, args.serial, ids=not args.no_ids)
    print("mock server on", url, flush=True)
    threading.Event().wait()


# ----------------------------------------------------------------------------------------------------------------------
# selftest
# ----------------------------------------------------------------------------------------------------------------------

def mini_cases(n=36):
    rng = random.Random(7)
    cases = []
    for i in range(n):
        fam = ["greedy", "sampled", "chat", "eos", "min", "stop", "cancel", "cancel_ms"][i % 8]
        c = {"id": f"m{i:02d}", "tags": [fam], "max_tokens": rng.choice([8, 16, 40]), "temperature": 0, "arrival_ms": i * 10}
        txt = f"Unique{i} prompt text " + "lorem ipsum " * rng.randint(1, 50)
        if fam in ("chat", "eos", "min"):
            c["messages"] = [{"role": "user", "content": txt + (" Reply with one word." if fam != "chat" else "")}]
        else:
            c["prompt"] = txt
        if fam == "sampled":
            c.update(temperature=0.8, top_p=0.9, seed=1000 + i)
        if fam == "eos":
            c["max_tokens"] = 120
        if fam == "min":
            c.update(min_tokens=30, max_tokens=60)
            c["messages"][0]["content"] += " [eos]"
        if fam == "stop":
            c["stop"] = [" alpha", "eta"]
            c["max_tokens"] = 80
        if fam == "cancel":
            c.update(cancel_after_chunks=3, max_tokens=60)
        if fam == "cancel_ms":
            c.update(cancel_after_ms=40, max_tokens=60)
        cases.append(c)
    return cases


def cmd_selftest(_args):
    import tempfile
    fails = []

    def check(name, cond, detail=""):
        print(("PASS " if cond else "FAIL ") + name + (f"  {detail}" if detail and not cond else ""))
        if not cond:
            fails.append(name)

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="cbid-selftest-"))
    cases = mini_cases()
    cp = tmp / "cases.jsonl"
    cp.write_text("".join(json.dumps(c) + "\n" for c in cases))
    cbi = {c["id"]: c for c in load_cases(cp)}

    # 1. compare_case on synthetic records
    def rec(text="abc def", deltas=("abc", " def"), finish="length", ct=2, status="ok", stream=True, cached=0, ids=None):
        return {"id": "x", "status": status, "stream": stream, "text": text, "deltas": list(deltas) if stream else None, "ids": ids,
                "finish_reason": finish, "usage": {"prompt_tokens": 5, "completion_tokens": ct, "prompt_tokens_details": {"cached_tokens": cached}}}

    ref = rec()
    cc = {"id": "x"}
    check("synthetic: identical passes", compare_case(cc, ref, rec())[0] == "pass")
    check("synthetic: text diff caught", compare_case(cc, ref, rec(text="abc deX", deltas=("abc", " deX")))[0] == "mismatch")
    check("synthetic: delta diff with same text caught", compare_case(cc, ref, rec(deltas=("ab", "c def")))[0] == "mismatch")
    check("synthetic: finish diff caught", compare_case(cc, ref, rec(finish="stop"))[0] == "mismatch")
    check("synthetic: completion_tokens diff caught", compare_case(cc, ref, rec(ct=3))[0] == "mismatch")
    check("synthetic: non-stream vs stream reference compares text only", compare_case(cc, ref, rec(stream=False))[0] == "pass")
    check("synthetic: test error caught", compare_case(cc, ref, rec(status="error"))[0] == "mismatch")
    check("synthetic: reference error flagged", compare_case(cc, rec(status="error"), rec())[0] == "ref_error")
    check("synthetic: missing reference flagged", compare_case(cc, None, rec())[0] == "missing")
    check("synthetic: valid cancelled prefix passes", compare_case(cc, ref, rec(text="abc", deltas=("abc",), status="cancelled", finish=None))[0] == "pass")
    check("synthetic: empty cancelled prefix passes", compare_case(cc, ref, rec(text="", deltas=(), status="cancelled", finish=None))[0] == "pass")
    check("synthetic: cancel prefix violation (text) caught", compare_case(cc, ref, rec(text="abX", deltas=("abX",), status="cancelled", finish=None))[0] == "mismatch")
    check("synthetic: cancel prefix violation (deltas) caught", compare_case(cc, ref, rec(text="abc", deltas=("ab", "c"), status="cancelled", finish=None))[0] == "mismatch")
    v, _, info = compare_case(cc, ref, rec(cached=3))
    check("synthetic: cached_tokens difference reported, not a mismatch", v == "pass" and info["cached_differs"])

    # token ids: primary criterion, compared only when both records have them
    rid = rec(ids=[1, 2])
    check("synthetic ids: identical ids pass and are compared", compare_case(cc, rid, rec(ids=[1, 2]))[0] == "pass" and compare_case(cc, rid, rec(ids=[1, 2]))[2].get("ids_compared"))
    v, why, _ = compare_case(cc, rid, rec(ids=[1, 3]))
    check("synthetic ids: id diff with identical text caught, reported first", v == "mismatch" and why[0].startswith("TOKEN IDS differ"), str(why))
    check("synthetic ids: id count diff caught", compare_case(cc, rid, rec(ids=[1]))[0] == "mismatch")
    v, _, info = compare_case(cc, rid, rec())
    check("synthetic ids: test without ids falls back to text criteria", v == "pass" and not info.get("ids_compared"))
    check("synthetic ids: ref without ids falls back to text criteria", compare_case(cc, rec(), rec(ids=[1, 2]))[0] == "pass")
    check("synthetic ids: valid cancelled id prefix passes", compare_case(cc, rid, rec(text="abc", deltas=("abc",), status="cancelled", finish=None, ids=[1]))[0] == "pass")
    check("synthetic ids: cancelled id prefix violation caught", compare_case(cc, rid, rec(text="abc", deltas=("abc",), status="cancelled", finish=None, ids=[2]))[0] == "mismatch")

    # 2. end to end on the mock server: clean => identical
    ref_srv, ref_url = make_mock(serial=True)
    srv, url = make_mock()
    ns = argparse.Namespace(url=ref_url, b=4, stream="on", arrival_scale=0.2, timeout=30, flush="auto", flush_id=FLUSH_ID, cap_max_tokens=None, cases_sha=None)
    h, recs = run_cases(ns, load_cases(cp), "solo")
    write_run(tmp / "ref.jsonl", h, recs)
    check("mock reference run has no errors", all(r["status"] == "ok" for r in recs))
    tests = []
    for sm in ("on", "off", "mixed"):
        ns.url, ns.stream = url, sm
        h, recs = run_cases(ns, load_cases(cp), "concurrent")
        write_run(tmp / f"c-{sm}.jsonl", h, recs)
        tests.append(str(tmp / f"c-{sm}.jsonl"))
        check(f"mock concurrent run ({sm}) overlapped requests", h["peak_client_inflight"] >= 2, str(h["peak_client_inflight"]))
    rh, rr = read_run(tmp / "ref.jsonl")
    rep = compare_runs(cbi, rh, rr, tests)
    for run in rep["runs"]:
        check(f"clean mock, {run['file'][-8:]}: 0 mismatches", not run["mismatches"], str([m["id"] + ":" + "; ".join(m["reasons"]) for m in run["mismatches"]][:3]))
        check(f"clean mock, {run['file'][-8:]}: every case compared", len(run["rows"]) == len(cases))
    check("mock: reference and concurrent runs carried token ids", rh.get("token_ids") and all(r["ids"] is not None for r in rr.values()))
    check("mock: ids compared on every completed case", all(run["counts"].get("ids_compared", 0) >= len(cases) - 2 * len([c for c in cases if is_cancel_case(c)]) for run in rep["runs"]), str([run["counts"] for run in rep["runs"]]))
    check("mock: ids count == completion_tokens minus EOS", all(len(r["ids"]) == r["usage"]["completion_tokens"] - (1 if r["finish_reason"] == "stop" and not cbi[r["id"]].get("stop") else 0)
                                                                for r in rr.values() if cbi[r["id"]].get("stop") is None))
    cancels = [r for r in rep["runs"][0]["rows"] if r["cancel_case"]]
    check("mock: cancelled cases observed with emitted output", any(r.get("cancelled") and r.get("emitted_bytes") for r in cancels), str(cancels[:2]))

    # 3. injected concurrency bug is detected, and only in overlapped requests
    bug_srv, bug_url = make_mock(bug="inflight")
    ns.url, ns.stream = bug_url, "on"
    h, recs = run_cases(ns, load_cases(cp), "concurrent")
    write_run(tmp / "bug.jsonl", h, recs)
    rep = compare_runs(cbi, rh, rr, [str(tmp / "bug.jsonl")])
    nb = len(rep["runs"][0]["mismatches"])
    check("injected concurrency bug detected", nb > 0 and nb <= bug_srv.state["bug_fired"], f"{nb} mismatches, bug fired {bug_srv.state['bug_fired']}")

    # 3b. a bug that changes only the id (text identical) is caught by the id criterion alone
    idbug_srv, idbug_url = make_mock(bug="inflight-id")
    ns.url, ns.stream, ns.cap_max_tokens = idbug_url, "on", None
    h, recs = run_cases(ns, load_cases(cp), "concurrent")
    write_run(tmp / "idbug.jsonl", h, recs)
    rep = compare_runs(cbi, rh, rr, [str(tmp / "idbug.jsonl")])
    nb = len(rep["runs"][0]["mismatches"])
    check("id-only concurrency bug detected (text identical)", nb > 0 and all(r["reasons"][0].startswith("TOKEN IDS") for r in rep["runs"][0]["mismatches"]) and nb <= idbug_srv.state["bug_fired"], f"{nb} mismatches, fired {idbug_srv.state['bug_fired']}")

    # 3c. server without the extension: probe says no, runs still work, text criteria apply, a warning says ids were skipped
    old_srv, old_url = make_mock(ids=False)
    ns.url, ns.stream = old_url, "mixed"
    h, recs = run_cases(ns, load_cases(cp), "concurrent")
    write_run(tmp / "noids.jsonl", h, recs)
    check("no-extension server: probe finds no ids, nothing requested", h["token_ids"] is False and all(r["ids"] is None for r in recs))
    rep = compare_runs(cbi, rh, rr, [str(tmp / "noids.jsonl")])
    check("no-extension server: text criteria still pass, ids-not-compared warning", not rep["runs"][0]["mismatches"] and any("NOT compared" in w for w in rep["warnings"]), str(rep["warnings"]))
    ns.token_ids = "on"
    try:
        run_cases(ns, load_cases(cp), "concurrent")
        check("--token-ids on against a server without ids fails", False)
    except SystemExit:
        check("--token-ids on against a server without ids fails", True)
    ns.token_ids = "off"
    h, recs = run_cases(ns, load_cases(cp), "concurrent")
    check("--token-ids off never requests ids", h["token_ids"] is False)
    ns.token_ids = "auto"

    # 4. perturbation (max_tokens cap) => predicted mismatch set exactly
    ns.url, ns.stream, ns.cap_max_tokens = url, "mixed", 5
    h, recs = run_cases(ns, load_cases(cp), "concurrent")
    write_run(tmp / "cap.jsonl", h, recs)
    rep = compare_runs(cbi, rh, rr, [str(tmp / "cap.jsonl")], expect_cap=5)
    e = rep["runs"][0]["expect_cap"]
    check("max_tokens perturbation: mismatch set == predicted set", e["equal"] and e["expected"] > 0, str(e))

    # 5. gen/load validation
    bad = tmp / "bad.jsonl"
    bad.write_text(json.dumps({"id": "z", "prompt": "p", "max_tokens": 4}) + "\n")
    try:
        load_cases(bad)
        check("case without explicit temperature rejected", False)
    except SystemExit:
        check("case without explicit temperature rejected", True)
    for s in (ref_srv, srv, bug_srv, idbug_srv, old_srv):
        s.shutdown()
    print(f"\nselftest: {'FAILED ' + str(fails) if fails else 'all checks passed'} (scratch: {tmp})")
    return 1 if fails else 0


# ----------------------------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--cases", default=str(DEFAULT_CASES), help="JSONL case file")
        p.add_argument("--sample", type=int, help="use about N cases: stratified by case family (>= 2 each), evenly spaced inside a family")
        p.add_argument("--ids", help="comma separated case ids to use")
        p.add_argument("--timeout", type=float, default=900, help="per-request timeout, seconds")
        p.add_argument("--flush", choices=["auto", "on", "off"], default="auto",
                       help="solo runs: send a 1-token request with an unrelated prompt before each case so no KV prefix is reused (auto: on for solo)")
        p.add_argument("--flush-id", type=int, default=FLUSH_ID, help="token id used for the flush prompt (default Qwen <|endoftext|>)")
        p.add_argument("--arrival-scale", type=float, default=1.0, help="multiply every arrival_ms (0 = all at once, bounded by B)")
        p.add_argument("--token-ids", choices=["auto", "on", "off"], default="auto",
                       help="request generated token ids (return_token_ids) and compare them: auto = only if a probe shows the server supports it")
        p.add_argument("--cap-max-tokens", type=int, help="perturbation: cap every case's max_tokens (self-test of the detector)")

    g = sub.add_parser("gen-cases", help="write the default case set (needs tokenizers)")
    g.add_argument("--out", default=str(DEFAULT_CASES))
    g.add_argument("--count", type=int, default=200)
    g.add_argument("--seed", type=int, default=20261015)
    g.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    g.set_defaults(fn=cmd_gen_cases)

    r = sub.add_parser("run", help="run the cases against one server (solo = reference, concurrent = replay)")
    common(r)
    r.add_argument("--url", required=True)
    r.add_argument("--mode", choices=["solo", "concurrent"], required=True)
    r.add_argument("--b", type=int, default=4, help="concurrent: max requests in flight")
    r.add_argument("--stream", choices=["on", "off", "mixed"], default="mixed",
                   help="concurrent: streaming for all / none / a deterministic half (solo runs always stream, unless off)")
    r.add_argument("--out", required=True)
    r.set_defaults(fn=cmd_run)

    c = sub.add_parser("compare", help="compare concurrent run(s) against a reference run; exit 1 on any mismatch")
    c.add_argument("--ref", required=True)
    c.add_argument("--test", nargs="+", required=True)
    c.add_argument("--cases", default=str(DEFAULT_CASES))
    c.add_argument("--report", help="JSON report path")
    c.add_argument("--expect-cap", type=int, help="perturbation self-test: the test run used --cap-max-tokens N; pass iff the mismatch set is exactly the predicted one")
    c.set_defaults(fn=cmd_compare)

    ga = sub.add_parser("gate", help="reference run, then concurrent runs for each B and stream mode, then compare")
    common(ga)
    ga.add_argument("--url", required=True, help="server for the concurrent runs")
    ga.add_argument("--ref-url", help="server for the reference run (default: --url)")
    ga.add_argument("--ref-file", help="reuse an existing reference run instead of running one")
    ga.add_argument("--b", default="2,3,4,6,8")
    ga.add_argument("--streams", default="mixed", help="comma list of on,off,mixed")
    ga.add_argument("--stream", default="on", help=argparse.SUPPRESS)
    ga.add_argument("--outdir", required=True)
    ga.set_defaults(fn=cmd_gate)

    m = sub.add_parser("mock", help="run the mock server (offline testing)")
    m.add_argument("--bug", choices=["inflight", "inflight-id"])
    m.add_argument("--serial", action="store_true")
    m.add_argument("--no-ids", action="store_true", help="behave like a server without the return_token_ids extension")
    m.set_defaults(fn=cmd_mock)

    s = sub.add_parser("selftest", help="offline self-test (mock server + synthetic records)")
    s.set_defaults(fn=cmd_selftest)

    args = ap.parse_args()
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()
